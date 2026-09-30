"""Fit on all labeled data, predict validation.csv and the fixed December lane, write submission files."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from pipeline import FreightModel, build_features, load, daily_market_index

DATA = "data/"

if __name__ == "__main__":
    train, val = load(DATA + "train_test.csv"), load(DATA + "validation.csv")
    model = FreightModel().fit(train)
    print(f"trend gamma = {model.gamma:.6f} log-points/day")

    # --- validation predictions ---
    tmpl = pd.read_csv(DATA + "validation_predictions_template.csv")
    pred = pd.Series(model.predict(val), index=val.load_id)
    out = pd.DataFrame({"load_id": tmpl.load_id, "predicted_rate": pred.reindex(tmpl.load_id).round(2).values})
    assert out.predicted_rate.notna().all() and (out.predicted_rate > 0).all() and len(out) == 12000
    out.to_csv("validation_predictions.csv", index=False)

    # --- December fixed-lane predictions (only the date changes) ---
    dec = pd.read_csv(DATA + "december_chart_inputs.csv")
    dec["date"] = pd.to_datetime(dec["date"])
    coords = train.drop_duplicates("pickup").set_index("pickup")[["pickup_lat", "pickup_lon"]]
    coords_d = train.drop_duplicates("delivery").set_index("delivery")[["delivery_lat", "delivery_lon"]]
    f = dec.copy()
    f[["pickup_lat", "pickup_lon"]] = coords.loc[f.pickup].values
    f[["delivery_lat", "delivery_lon"]] = coords_d.loc[f.delivery].values
    # market_index / quote_signal are not given for December: take the daily market index observed in
    # validation.csv (features only, no labels) and the average quote_signal of December loads.
    vdaily = daily_market_index(val)
    f["market_index"] = f["date"].map(vdaily)
    f["quote_signal"] = val.loc[val.date >= "2025-12-01", "quote_signal"].mean()
    dec["predicted_rate"] = model.predict(f, build_features(f, model.route_tbl)).round(2)
    dec["date"] = dec["date"].dt.strftime("%Y-%m-%d")
    dec.to_csv(DATA + "december_chart_inputs_predicted.csv", index=False)
    print(dec[["date", "predicted_rate"]].describe())
    print(out.predicted_rate.describe())
