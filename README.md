# Freight Rate Prediction

LightGBM model (log-rate, Huber loss) with a linear time-trend term, validated with time-ordered folds.

## Setup
```bash
python -m pip install -r requirements.txt
```
Place the assessment files in `data/` (they are git-ignored): `train_test.csv`, `validation.csv`,
`validation_predictions_template.csv`, `december_chart_inputs.csv`.

## Run
```bash
python src/evaluate.py     # rolling-origin validation -> outputs/cv_results.csv
python src/predict.py      # fits on all labeled data, writes validation_predictions.csv
                           # and data/december_chart_inputs_predicted.csv
python score.py --predictions validation_predictions.csv \
  --december-predictions data/december_chart_inputs_predicted.csv
```
The scorer writes `scorer_results/candidate_december.png`.

## Approach (short)
- **Split:** labeled data is Jan-Oct 2025, the test set is Nov-Dec 2025, so validation is *time-ordered*:
  3 rolling-origin folds (train on the past, test on the next 2 months). Random K-fold would leak the market regime.
- **Data quality:** negative weights (sign errors) -> abs; weights capped at 47,500 -> flag; missing weight (tree-native NaN);
  missing market_index -> same-day mean; ~1.1% rows with implausible $/mile (<0.8 or >6) dropped from training;
  noisy listed distance -> route-median distance feature + distance/haversine ratio.
- **Model:** LightGBM on log(rate), Huber loss, 3-seed average. A linear time trend (estimated from daily residuals vs
  market index and time) is applied as an offset so the model does not extrapolate flat into Nov-Dec.
- **December chart:** only the date varies; market_index for each December day is taken from validation.csv (features only),
  quote_signal is the December average.
