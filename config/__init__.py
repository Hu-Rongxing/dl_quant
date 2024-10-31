import configparser
from pathlib import Path

config_path = Path(__file__).parent.parent / "config/config.ini"
# 加载配置文件
config = configparser.ConfigParser()
files_read = config.read(config_path.as_posix())
if not files_read:
    raise FileNotFoundError(f"配置文件未找到: {config_path}")

data_dir = Path(__file__).parent.parent.joinpath('data/precessed_data')
DATA_SAVE_PATHS = {
    'train': data_dir / 'train.pkl',
    'val': data_dir / 'val.pkl',
    'test': data_dir / 'test.pkl',

    'past_covariates': data_dir / 'past_covariates.pkl',
    'future_covariates': data_dir / 'future_covariates.pkl',

    'scaler_train': data_dir / 'scaler_train.pkl',
    'scaler_past': data_dir / 'scaler_past.pkl',
}

# 验证集、训练集、测试集的长度。
TIMESERIES_LENGTH = {
    'val_length': 60,
    'test_length': 60,
    'header_length': 150
}