from pathlib import Path

# 训练数据保存目录
data_dir = Path(__file__).parent.joinpath('data/precessed_data')
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
