import os

import matplotlib.pyplot as plt
import pandas as pd

# Load the two datasets (they are in the same folder as this script).
folder = os.path.dirname(os.path.abspath(__file__))
ai_df = pd.read_csv(os.path.join(folder, "toxigen_processed_scored.csv"))
human_df = pd.read_csv(os.path.join(folder, "mhs_5groups_2500_results.csv"))


def get_rates(df):
  fn_rate = 1 - df[df["true_label"] == 1]["pred_label"].mean()
  fp_rate = df[df["true_label"] == 0]["pred_label"].mean()
  return fn_rate, fp_rate


groups = ["asian", "mental_dis", "muslim", "women", "black"]
labels = ["Asian", "Mental disability", "Muslim", "Women", "Black",
          "All texts"]

# Rates for each group, and for all texts at the end (top row of the plot).
ai_fn, ai_fp, human_fn, human_fp = [], [], [], []
for group in groups:
  fn, fp = get_rates(ai_df[ai_df["target_group"] == group])
  ai_fn.append(fn)
  ai_fp.append(fp)
  fn, fp = get_rates(human_df[human_df["target_group"] == group])
  human_fn.append(fn)
  human_fp.append(fp)

fn, fp = get_rates(ai_df)
ai_fn.append(fn)
ai_fp.append(fp)
fn, fp = get_rates(human_df)
human_fn.append(fn)
human_fp.append(fp)

# Prepare the bar positions, the left (negative) values and the text labels.
# Each row has two bars: AI slightly above, human slightly below.
rows = range(len(labels))
ai_y, human_y = [], []
ai_fn_left, human_fn_left = [], []
ai_fn_text, ai_fp_text, human_fn_text, human_fp_text = [], [], [], []
for i in rows:
  ai_y.append(i + 0.2)
  human_y.append(i - 0.2)
  ai_fn_left.append(-ai_fn[i])
  human_fn_left.append(-human_fn[i])
  ai_fn_text.append(f"{ai_fn[i]:.2f}")
  ai_fp_text.append(f"{ai_fp[i]:.2f}")
  human_fn_text.append(f"{human_fn[i]:.2f}")
  human_fp_text.append(f"{human_fp[i]:.2f}")

fig, ax = plt.subplots(figsize=(10, 6))

# FN rates are drawn to the left (negative), FP rates to the right.
bars1 = ax.barh(ai_y, ai_fn_left, height=0.35, color="tab:blue",
                label="AI-generated (ToxiGen)")
bars2 = ax.barh(ai_y, ai_fp, height=0.35, color="tab:blue")
bars3 = ax.barh(human_y, human_fn_left, height=0.35, color="tab:orange",
                label="Human-written (MHS)")
bars4 = ax.barh(human_y, human_fp, height=0.35, color="tab:orange")

# Write the rate next to each bar (without the minus sign).
ax.bar_label(bars1, labels=ai_fn_text, padding=3)
ax.bar_label(bars2, labels=ai_fp_text, padding=3)
ax.bar_label(bars3, labels=human_fn_text, padding=3)
ax.bar_label(bars4, labels=human_fp_text, padding=3)

ax.axvline(0, color="black", linewidth=1)
ax.set_yticks(rows)
ax.set_yticklabels(labels)
ax.set_xlim(-0.9, 0.9)
ax.set_xticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
ax.set_xticklabels(["0.8", "0.6", "0.4", "0.2", "0", "0.2", "0.4", "0.6",
                    "0.8"])
ax.set_xlabel("← Missed hate (FN rate)          "
              "False alarms (FP rate) →")
ax.set_title("Same model, opposite errors: AI-generated vs. human-written "
             "text")
ax.legend(loc="lower right")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(folder, "butterfly_chart.png"), dpi=200)
plt.show()

