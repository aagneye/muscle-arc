# muscle-arc

Automated muscle architecture estimation from ultrasound images for the
[UMUD Challenge](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data)
(pennation angle, fascicle length, muscle thickness).

## Task

Predict per image:

| Column   | Parameter        | Unit |
|----------|------------------|------|
| `pa_deg` | Pennation angle  | deg  |
| `fl_mm`  | Fascicle length  | mm   |
| `mt_mm`  | Muscle thickness | mm   |

Metric: **UMUD Score** (normalized MAE across PA / FL / MT; lower is better).

## Approach

Primary pipeline: **segment aponeuroses + fascicles → geometric measurement**.

See [docs/method.md](docs/method.md) for the technical design.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Download competition data (requires [Kaggle API](https://github.com/Kaggle/kaggle-api) credentials):

```bash
python scripts/download_data.py
```

## Train / predict

```bash
python scripts/train.py --config configs/default.yaml
python scripts/predict.py --config configs/default.yaml --out submissions/submission.csv
```

## Layout

```
configs/          experiment configs
data/             raw + processed (gitignored except placeholders)
docs/             challenge notes + method
notebooks/        EDA
scripts/          download / train / predict entrypoints
src/muscle_arc/   library code
submissions/      CSV outputs
experiments/      run artifacts
```

## Citation

Paul Ritsche, Gerardo Romney & Oliver Faude. UMUD Challenge: Muscle Architecture
in Ultrasound Data. https://kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data, 2026. Kaggle.

## License

MIT — required for prize eligibility (OSI-compatible open source).
