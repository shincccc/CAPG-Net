# Data directory

This directory contains all datasets used by CAPG-Net.

## Structure

data/
├── EOP/
│   ├── ori/           # Raw EOP data from IERS (txt + csv)
│   └── processed/     # Processed EOP data (leap-second & tidal corrections applied)
├── ENSO/              # Daily Niño-3.4 index (NOAA OISST v2 via KNMI)
├── EAM/
│   ├── ori/           # Raw EAM data from GFZ (annual .asc files)
│   └── processed/     # Processed EAM data (daily CSV per component)
└── utils/             # Data preprocessing scripts

## Data Sources

| Dataset | Source |
|---------|--------|
| EOP 14 C04 | [IERS](https://www.iers.org/) |
| AAM, OAM, HAM, SLAM | [GFZ](https://www.gfz-potsdam.de/en/esmdata) |
| Daily Niño-3.4 | [KNMI Climate Explorer](https://climexp.knmi.nl/) (NOAA OISST v2) |
