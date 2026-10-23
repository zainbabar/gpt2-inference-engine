# redraw the plot from saved timings, no model run needed
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot saved KV cache benchmark results"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/cpu.png"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    rows = sorted(data["results"], key=lambda row: row["total_tokens"])
    lengths = [row["total_tokens"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name, label, color in [
        ("uncached", "Full sequence each step", "#777777"),
        ("cached", "KV cache", "#1864ab"),
    ]:
        rates = [row[name]["tokens_per_second"] for row in rows]
        # show the full range of repeat timings, not a confidence interval
        low = [min(row["new_tokens"] / t for t in row[name]["seconds"]) for row in rows]
        high = [
            max(row["new_tokens"] / t for t in row[name]["seconds"]) for row in rows
        ]
        axes[0].plot(lengths, rates, "o-", label=label, color=color)
        axes[0].fill_between(lengths, low, high, color=color, alpha=0.15)
    axes[0].set_ylabel("Generated tokens / second")
    axes[0].legend(frameon=False)
    axes[1].plot(lengths, [row["speedup"] for row in rows], "o-", color="#1864ab")
    axes[1].axhline(1, color="#aaaaaa", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Speedup (uncached time / cached time)")
    for ax in axes:
        ax.set_xlabel("Prompt + generated tokens")
        ax.set_xticks(lengths)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    n = rows[0]["new_tokens"]
    fig.suptitle(
        f"GPT-2 small · {data['device_name']} · float32 · batch 1\n"
        f"{n} generated tokens, prefill included · median of {data['repeats']} runs"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix(".svg"))
    plt.close(fig)
    print(f"saved {args.output} and {args.output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
