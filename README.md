# CAPG-Net

Climate-Adaptive Physics-Guided Network for dUT1 Forecasting

## Project Structure

```
├── data_processor.py # Preprocessing: EOP + EAM + daily Niño‑3.4 alignment
├── train.py # Main training script
├── eval.py # Batch evaluation (outputs annual & global MAE)
├── configs.py # Relative data paths
├── dl_models/ # Model implementations (6 models used in paper, CAPG-Net is in dl_models/lstm_models.py)
├── data/
│ ├── EOP/
│ │ ├── ori/ # Raw EOP data from IERS (time-filtered only)
│ │ └── processed/ # Processed EOP data (leap seconds & tidal corrections applied)
│ ├── ENSO/ # Daily Niño‑3.4 (NOAA OISST v2 via KNMI)
│ ├── EAM/
│ │ ├── ori/ # Raw EAM data from GFZ (.asc files)
│ │ └── processed/ # Processed EAM data (daily CSV per component)
│ └── utils/ # Data preprocessing scripts
├── weight/
│ └── fluid_model_weights/ # Trained model checkpoints
├── results/ # Evaluation output files
├── logs/ # Training logs
├── requirements.txt
├── LICENSE
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