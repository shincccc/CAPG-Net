# CAPG-Net

Climate-Adaptive Physics-Guided Network for dUT1 Forecasting

## Project Structure

```
├── data_processor.py # Preprocessing: EOP + EAM + daily Niño‑3.4 alignment
├── train.py # Main training script
├── eval.py # Batch evaluation (outputs annual & global MAE)
├── configs.py # relative data paths
├── dl_models/ # Model implementations
├── data/
│ ├── EOP/processed/ # UT1‑UTC (C04) & rapid Finals
│ ├── ENSO/ # Daily Niño‑3.4 (OISST v2)
│ └── EAM/ # AAM, OAM, HAM, SLAM excitation series
├── weight/
│ └── fluid_model_weights/ # trained checkpoints
├── results/
├── requirements.txt
└── README.md
```

## Data Sources

| Data | Source |
|------|--------|
| EOP 14 C04 | IERS |
| AAM, OAM, HAM, SLAM | GFZ |
| Daily Niño-3.4 | KNMI Climate Explorer (NOAA OISST v2) |

## Usage

Configure data paths in `configs.py`, then:

```bash
# Train
python train.py

# Evaluate
python eval.py
```