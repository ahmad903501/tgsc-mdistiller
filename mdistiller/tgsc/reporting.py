"""Human-readable tables and saved training curves derived from the JSON log."""
import csv
import json
from pathlib import Path


def flatten(row, prefix=""):
    result = {}
    for key, value in row.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def write_reports(output, *, plots=True):
    output = Path(output)
    rows = [json.loads(line) for line in (output/"metrics.jsonl").read_text().splitlines() if line.strip()]
    flattened = [flatten(row) for row in rows]
    fields = list(dict.fromkeys(key for row in flattened for key in row))
    with (output/"metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)
    if not plots or not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for key in dict.fromkeys(key for row in rows for key in row["train"]):
        axes[0,0].plot(epochs, [row["train"].get(key, float("nan")) for row in rows], label=key)
    axes[0,0].set(title="Training losses", ylabel="Loss")
    axes[0,0].legend()
    measured = [row for row in rows if row.get("test") is not None]
    if measured:
        for key in ("top1", "top5"):
            axes[0,1].plot([r["epoch"] for r in measured], [r["test"][key] for r in measured], marker="o", label=key)
        axes[0,1].legend()
    axes[0,1].set(title="Measured test accuracy", ylabel="Accuracy (%)")
    axes[1,0].plot(epochs, [row["gate_mean"] for row in rows], label="Mean gate")
    axes[1,0].plot(epochs, [row["diagnostics"].get("q_mean", float("nan")) for row in rows], label="Mean quality")
    axes[1,0].set(title="TGSC quality and gating", ylabel="Value")
    axes[1,0].legend()
    axes[1,1].step(epochs, [row["lr"] for row in rows], where="post")
    axes[1,1].set(title="Learning rate", ylabel="Learning rate", yscale="log")
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=.25)
    figure.savefig(output/"training_curves.png", dpi=160)
    figure.savefig(output/"training_curves.svg")
    plt.close(figure)
