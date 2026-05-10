# configs.py
import os

class Config:
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')

        self.OUTPUT_DIR = os.path.join(self.BASE_DIR, 'output')
        self.EOP_FILE = os.path.join(self.DATA_DIR, 'EOP', 'processed', 'eop_ut1r_tai_2000_2025_full.csv')
        self.FINALS_FILE = os.path.join(self.DATA_DIR, 'EOP', 'processed', 'finals2000A_processed.csv')
        self.NINO34_FILE = os.path.join(self.DATA_DIR, 'ENSO', 'nino34_daily_2000_2025.csv')
        self.AAM_FILE = os.path.join(self.DATA_DIR, 'EAM', 'AAM_daily_1976_present.csv')
        self.OAM_FILE = os.path.join(self.DATA_DIR, 'EAM', 'OAM_daily_1976_present.csv')
        self.HAM_FILE = os.path.join(self.DATA_DIR, 'EAM', 'HAM_daily_1976_present.csv')
        self.SLAM_FILE = os.path.join(self.DATA_DIR, 'EAM', 'SLAM_daily_1976_present.csv')

        self.DEEP_LEARNING = False
        self.FORECAST_PERIODS = {'10d': 10}
        self.ROLLING_CONFIG = {'window_years': 10, 'min_train_samples': 100}