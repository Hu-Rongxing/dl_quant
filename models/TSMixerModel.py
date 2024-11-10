import torch
from darts.models import TSMixerModel
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from matplotlib import font_manager
import optuna
from pathlib import Path
from config import TIMESERIES_LENGTH
from load_data.multivariate_timeseries import generate_processed_series_data
from utils.logger import logger
from models.params import get_pl_trainer_kwargs

torch.set_float32_matmul_precision('medium')

MODEL_NAME = "TSMixerModel"
WORK_DIR = Path(f"logs/{MODEL_NAME}_logs").resolve()
PRED_STEPS = TIMESERIES_LENGTH["test_length"]

# 准备训练和验证数据
data = generate_processed_series_data('training')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def define_model(trial):
    parameters = {
        "input_chunk_length": trial.suggest_int("input_chunk_length", 4, 64),
        "output_chunk_length": trial.suggest_int("output_chunk_length", 1, min(20, PRED_STEPS)),
        "hidden_size": trial.suggest_int("hidden_size", 32, 512),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        # "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
        "activation": trial.suggest_categorical("activation", ["ReLU", "GELU", "SELU"]),
        "num_blocks": trial.suggest_int("num_blocks", 1, 4),
    }

    model = TSMixerModel(
        **parameters,
        pl_trainer_kwargs=get_pl_trainer_kwargs(full_training=True),
        work_dir=WORK_DIR,
        save_checkpoints=True,
        force_reset=True,
        model_name=MODEL_NAME,
        batch_size=128,
        n_epochs=50,
        random_state=42,
        log_tensorboard=False,
    )

    return model


def train_and_evaluate(model, data):
    try:
        model.fit(
            series=data['train'][-300:],
            past_covariates=data['past_covariates'],
            future_covariates=data['future_covariates'],
            val_series=data['val'],
            val_past_covariates=data['past_covariates'],
            val_future_covariates=data['future_covariates'],
        )
    except Exception as e:
        logger.error(f"模型训练失败: {str(e)}")
        return 0.0

    backtest_series = model.historical_forecasts(
        series=data['test'],
        past_covariates=data['past_covariates'],
        future_covariates=data['future_covariates'],
        start=data['test'].time_index[-PRED_STEPS],
        forecast_horizon=1,
        stride=1,
        retrain=False
    )

    true_labels = data["test"][-PRED_STEPS:].values().flatten().astype(int)
    probabilities = backtest_series[-PRED_STEPS:].values().flatten()
    binary_predictions = (probabilities > 0.5).astype(int)

    overall_precision = precision_score(true_labels, binary_predictions, zero_division=0)
    logger.info(f"整体精度: {overall_precision:.4%}")

    # 使用用户提供的字体
    font_path = 'C:/Windows/Fonts/msyh.ttc'  # 用户提供的字体文件路径
    my_font = font_manager.FontProperties(fname=font_path)

    # 绘制回测预测值
    backtest_series.pd_dataframe().plot(label='回测预测值', lw=3, alpha=0.25)
    plt.title("TSMixer Model Backtest - Last 20 Steps", fontproperties=my_font)
    plt.legend(prop=my_font)
    plt.show()
    plt.close()

    components_precision = {}
    for component_idx in backtest_series.components:
        component_true_labels = data['test'][component_idx][-PRED_STEPS:].values().flatten().astype(int)
        component_probabilities = backtest_series[component_idx][-PRED_STEPS:].values().flatten()
        component_predictions = (component_probabilities > 0.5).astype(int)

        component_precision = precision_score(component_true_labels, component_predictions, zero_division=0)
        components_precision[component_idx] = component_precision
        logger.info(f"Component {component_idx} 精度: {component_precision:.4%}")

    # 绘制每个组件的精确率
    plt.figure(figsize=(12, 6))
    plt.bar(list(components_precision.keys()), components_precision.values(), color='b', alpha=0.7)
    plt.title(f"每个组件的精确率(总体精确率{overall_precision:.2%})", fontproperties=my_font)
    plt.xlabel("组件索引", fontproperties=my_font)
    plt.ylabel("精确率", fontproperties=my_font)
    plt.ylim(0, 1)
    plt.xticks(ticks=list(components_precision.keys()), labels=list(components_precision.keys()), rotation=45,
               fontproperties=my_font)
    plt.grid(axis='y')
    plt.legend(prop=my_font)
    plt.show()
    plt.close()

    del model
    torch.cuda.empty_cache()

    return overall_precision

def objective(trial):
    model = define_model(trial)
    precision = train_and_evaluate(model, data)

    if precision is None:
        logger.error(f"试验{trial.number}失败，超参数: {trial.params}")
        return 0.0

    return precision

def logging_callback(study, trial):
    logger.info(f"试验{trial.number}: 当前精准率:{trial.value:.4%}; 最佳精准率:{study.best_value:.4%}；\n当前超参数： {trial.params}")

if __name__ == '__main__':
    study_name = 'tsmixermodel-precision-optimization-2'
    try:
        # 尝试加载现有研究
        study = optuna.load_study(
            study_name=study_name,
            storage='sqlite:///data/optuna/optuna_study.db'
        )
    except Exception as e:
        print(f"加载研究时发生错误: {e}")
        # 如果研究不存在，创建新的研究
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage="sqlite:///data/optuna/optuna_study.db"
        )
    # 在 optimize 函数中添加回调函数
    study.optimize(objective, n_trials=100, n_jobs=1, callbacks=[logging_callback])

    logger.info(f"Best hyperparameters: {study.best_params}")
    logger.info(f"Best precision: {study.best_value:.4f}")