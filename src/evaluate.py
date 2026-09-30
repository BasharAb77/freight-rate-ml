"""Time-aware validation: rolling-origin folds (train on the past, test on the next 2 months)."""
import sys
import pandas as pd
sys.path.insert(0, "src")
from pipeline import FreightModel, load, metrics

FOLDS = [("2025-05-01", "2025-07-01"), ("2025-07-01", "2025-09-01"), ("2025-09-01", "2025-11-01")]
VARIANTS = {
    "A baseline (keep all labels, no route_dist, no trend)": dict(use_route_dist=False, drop_bad_labels=False, use_trend=False, seeds=(0,)),
    "B + drop corrupted labels": dict(use_route_dist=False, drop_bad_labels=True, use_trend=False, seeds=(0,)),
    "C + route median distance": dict(use_route_dist=True, drop_bad_labels=True, use_trend=False, seeds=(0,)),
    "D + linear time trend (final, 1 seed)": dict(use_route_dist=True, drop_bad_labels=True, use_trend=True, seeds=(0,)),
}

if __name__ == "__main__":
    t = load("data/train_test.csv")
    rows = []
    for name, kw in VARIANTS.items():
        for a, b in FOLDS:
            tr, te = t[t.date < a], t[(t.date >= a) & (t.date < b)]
            m = FreightModel(**kw).fit(tr)
            r = metrics(m.predict(te), te.posted_rate, te.distance)
            rows.append({"variant": name, "fold": f"{a[:7]}..{b[:7]}", **r})
        print("done", name, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("outputs/cv_results.csv", index=False)
    cols = ["MAE", "MAE_clean", "MdAPE", "MAPE", "RMSE"]
    print("\nMEAN OVER 3 FOLDS"); print(df.groupby("variant")[cols].mean().round(3).to_string())
    print("\nLAST FOLD (Sep-Oct, closest to the Nov-Dec test)")
    print(df[df.fold.str.startswith("2025-09")].set_index("variant")[cols].round(3).to_string())
