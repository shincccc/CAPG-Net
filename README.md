# CAPG-Net

Climate-Adaptive Physics-Guided Network for dUT1 Forecasting

## Project Structure

```
├── data_processor.py         # Data loading and preprocessing
├── train.py                  # Model training
├── eval.py                   # Batch model evaluation
├── configs.py                # Data paths and parameters
├── dl_models/                # Model implementations
└── weight/                   # Pre-trained weights
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