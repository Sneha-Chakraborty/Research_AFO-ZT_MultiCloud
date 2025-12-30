from __future__ import annotations

from pathlib import Path
import pandas as pd


def main() -> None:
    path = Path("outputs/results/afozt_scores.csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path.resolve()}")

    df = pd.read_csv(path)
    print(f"Loaded: {path}  rows={len(df)}  cols={len(df.columns)}")

    # Decisions
    if "decision" in df.columns:
        print("\nDecision counts:")
        print(df["decision"].value_counts(dropna=False))
        restrict_or_deny = int(df["decision"].isin(["restrict", "deny"]).sum())
        print(f"\n#restrict_or_deny = {restrict_or_deny}")
    else:
        print("\n❌ Missing column: decision")

    # Labels
    has_label = "label_attack" in df.columns
    print(f"\nHas label_attack? {has_label}")
    if has_label:
        print("\nlabel_attack counts:")
        print(df["label_attack"].value_counts(dropna=False))

    # Risk + confidence sanity
    for col in ["risk", "confidence"]:
        if col in df.columns:
            print(f"\n{col} describe:")
            print(df[col].describe())
        else:
            print(f"\n⚠️ Missing column: {col}")


if __name__ == "__main__":
    main()
