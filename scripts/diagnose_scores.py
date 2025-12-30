# python - <<'PY'
import pandas as pd

df = pd.read_csv("outputs/results/afozt_scores.csv")
print("rows:", len(df))

print("\nDecision counts:")
print(df["decision"].value_counts(dropna=False))

print("\nConfidence describe:")
print(df["confidence"].describe())

override_thr = 0.55
severe = df["decision"].isin(["restrict","deny"])
override_rate = (severe & (df["confidence"] < override_thr)).mean()
print(f"\nOverride rate (restrict/deny & conf<{override_thr}): {override_rate:.4f}")

