# time the model loop, with and without its kv cache
import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from gpt2_engine.generate import ENCODING
from gpt2_engine.model import GPT2


def sync(device):
    # cuda calls are async, so wait before reading the clock
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def run(model, prompt, new_tokens, cache):
    model.reset_cache()
    model.set_caching(cache)
    ids = prompt.clone()
    sync(ids.device)
    start = perf_counter()
    next_id = model(ids)[:, -1, :].argmax(dim=-1, keepdim=True)
    ids = torch.cat((ids, next_id), dim=-1)
    sync(ids.device)
    prefill = perf_counter() - start
    # prefill already predicted the first token, so N - 1 calls remain
    for _ in range(new_tokens - 1):
        inputs = next_id if cache else ids
        next_id = model(inputs)[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat((ids, next_id), dim=-1)
    sync(ids.device)
    elapsed = perf_counter() - start
    return ids, elapsed, prefill


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPT-2 cached vs uncached")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[64, 128, 256, 512, 1024]
    )
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/cpu.json"))
    args = parser.parse_args()
    if args.new_tokens < 2 or args.repeats < 1 or args.warmup < 1 or args.threads < 1:
        parser.error("need at least 2 new tokens, 1 repeat, 1 warmup, and 1 thread")
    if any(length <= args.new_tokens or length > 1024 for length in args.lengths):
        parser.error("total lengths must be greater than new-tokens and at most 1024")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available in this environment")
    torch.set_num_threads(args.threads)
    model = GPT2.from_pretrained().to(device).eval()
    # deterministic text, sliced by token count so every length means the same thing
    text = "The quick brown fox jumps over the lazy dog. In linear algebra, a matrix transforms vectors. "
    source = ENCODING.encode(text * 100)
    rows = []
    for length in args.lengths:
        prompt = torch.tensor([source[: length - args.new_tokens]], device=device)
        for _ in range(args.warmup):
            run(model, prompt, args.new_tokens, False)
            run(model, prompt, args.new_tokens, True)
        samples = {False: [], True: []}
        prefills = {False: [], True: []}
        for repeat in range(args.repeats):
            outputs = {}
            # alternate the order to reduce always-timing-one-mode-first bias
            order = [False, True] if repeat % 2 == 0 else [True, False]
            for cache in order:
                outputs[cache], elapsed, prefill = run(
                    model, prompt, args.new_tokens, cache
                )
                samples[cache].append(elapsed)
                prefills[cache].append(prefill)
            assert torch.equal(outputs[False], outputs[True]), "greedy tokens differ"
        row = {
            "total_tokens": length,
            "prompt_tokens": prompt.shape[1],
            "new_tokens": args.new_tokens,
        }
        for cache, name in [(False, "uncached"), (True, "cached")]:
            row[name] = {
                "seconds": samples[cache],
                "median_seconds": median(samples[cache]),
                "tokens_per_second": args.new_tokens / median(samples[cache]),
                "prefill_seconds": prefills[cache],
            }
        row["speedup"] = median(samples[False]) / median(samples[True])
        rows.append(row)
        print(
            f"{length:4d} total tokens: {row['uncached']['tokens_per_second']:.2f} -> "
            f"{row['cached']['tokens_per_second']:.2f} tok/s, {row['speedup']:.2f}x",
            flush=True,
        )
    model.reset_cache()
    model.set_caching(False)
    result = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        "dtype": str(model.wte.weight.dtype),
        "threads": args.threads,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "prompt_text_repeated": text,
        "method": "batch 1 greedy; prefill + decode; excludes loading, tokenization, reset, and text decoding",
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
