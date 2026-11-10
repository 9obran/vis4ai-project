import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Folder that contains this script (and the datasets).
folder = os.path.dirname(os.path.abspath(__file__))

# Load the two scored datasets.
ai_df = pd.read_csv(os.path.join(folder, "toxigen_processed_scored.csv"))
human_df = pd.read_csv(os.path.join(folder, "mhs_5groups_2500_results.csv"))

groups = ["black", "women", "muslim", "mental_dis", "asian"]
labels = ["All texts", "Black", "Women", "Muslim", "Mental disability",
          "Asian"]


def get_rates(df):
  """Returns the FN rate and FP rate of the model on df."""
  fn_rate = 1 - df[df["true_label"] == 1]["pred_label"].mean()
  fp_rate = df[df["true_label"] == 0]["pred_label"].mean()
  return fn_rate, fp_rate


# Compute the rates for all texts first, then for each group.
ai_fn, ai_fp = [], []
human_fn, human_fp = [], []
for group in [None] + groups:
  if group is None:
    ai_part, human_part = ai_df, human_df
  else:
    ai_part = ai_df[ai_df["target_group"] == group]
    human_part = human_df[human_df["target_group"] == group]
  fn, fp = get_rates(ai_part)
  ai_fn.append(fn)
  ai_fp.append(fp)
  fn, fp = get_rates(human_part)
  human_fn.append(fn)
  human_fp.append(fp)

# Draw the chart: FN rates go to the left (negative), FP rates to the right.
y = np.arange(len(labels))[::-1]  # "All texts" on top.
h = 0.38
fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(y + h / 2, -np.array(ai_fn), height=h * 0.9, color="#2a78d6",
        label="AI-generated (ToxiGen)")
ax.barh(y + h / 2, ai_fp, height=h * 0.9, color="#2a78d6")
ax.barh(y - h / 2, -np.array(human_fn), height=h * 0.9, color="#eb6834",
        label="Human-written (MHS)")
ax.barh(y - h / 2, human_fp, height=h * 0.9, color="#eb6834")

# Write the value next to each bar.
for i in range(len(labels)):
  ax.text(-ai_fn[i] - 0.01, y[i] + h / 2, f"{ai_fn[i]:.2f}", ha="right",
          va="center", fontsize=8, color="#444444")
  ax.text(ai_fp[i] + 0.01, y[i] + h / 2, f"{ai_fp[i]:.2f}", ha="left",
          va="center", fontsize=8, color="#444444")
  ax.text(-human_fn[i] - 0.01, y[i] - h / 2, f"{human_fn[i]:.2f}",
          ha="right", va="center", fontsize=8, color="#444444")
  ax.text(human_fp[i] + 0.01, y[i] - h / 2, f"{human_fp[i]:.2f}",
          ha="left", va="center", fontsize=8, color="#444444")

# Axes, labels and styling.
ax.axvline(0, color="#333333", linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.get_yticklabels()[0].set_fontweight("bold")
ax.tick_params(axis="y", length=0)
ax.axhline(y[0] - 0.5, color="#cccccc", linewidth=0.8)

ax.set_xlim(-0.85, 0.85)
ticks = np.arange(-0.8, 0.81, 0.2)
ax.set_xticks(ticks)
ax.set_xticklabels([f"{abs(t):.1f}" for t in ticks])
ax.set_xlabel("Error rate")
ax.text(-0.42, len(labels) - 0.35, "← Missed hate (FN rate)",
        ha="center", fontsize=11, fontweight="bold")
ax.text(0.42, len(labels) - 0.35, "False alarms (FP rate) →",
        ha="center", fontsize=11, fontweight="bold")

ax.set_title("Same model, opposite errors: AI-generated vs. human-written "
             "text", fontsize=13, pad=28)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2,
          frameon=False)
ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(folder, "butterfly_chart.png"), dpi=200,
            bbox_inches="tight")
plt.show()
