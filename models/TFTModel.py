import torch
from darts.models import TFTModel
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from matplotlib import font_manager
import optuna
from pathlib import Path
from config import TIMESERIES_LENGTH
from load_data.multivariate_timeseries import generate_processed_series_data
from utils.logger import logger
from models.params import get_pl_trainer_kwargs, loss_logger

torch.set_float32_matmul_precision('medium')

MODEL_NAME = "TFTModel"
WORK_DIR = Path(f"logs/{MODEL_NAME}_logs").resolve()
PRED_STEPS = TIMESERIES_LENGTH["test_length"]

# 准备训练和验证数据
data = generate_processed_series_data('training')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def define_model(trial):
    parameters = {
        "input_chunk_length": trial.suggest_int("input_chunk_length", 1, 64),
        "output_chunk_length": trial.suggest_int("output_chunk_length", 1, min(20, PRED_STEPS)),
        "hidden_size": trial.suggest_int("hidden_size", 8, 64),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "lstm_layers": trial.suggest_int("lstm_layers", 1, 4),
        "num_attention_heads": trial.suggest_int("num_attention_heads", 1, 4),
        "full_attention": trial.suggest_categorical("full_attention", [True, False]),
        "feed_forward": trial.suggest_categorical("feed_forward", ['GatedResidualNetwork', 'ReLU']),
        "hidden_continuous_size": trial.suggest_int("hidden_continuous_size", 4, 32)
    }

    model = TFTModel(
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

    overall_precision = precision_score(true_labels, binary_predictions)
    logger.info(f"整体精度: {overall_precision:.4%}")

    # 使用用户提供的字体
    font_path = 'C:/Windows/Fonts/msyh.ttc'  # 用户提供的字体文件路径
    my_font = font_manager.FontProperties(fname=font_path)

    # 绘制回测预测值
    backtest_series.pd_dataframe().plot(label='回测预测值', lw=3, alpha=0.25)
    plt.title("TFT Model Backtest - Last 20 Steps", fontproperties=my_font)
    plt.legend(prop=my_font)  # 这里确保图例字体选择
    plt.show()
    plt.close()

    # 绘制训练损失和验证损失
    plt.figure(figsize=(12, 6))
    plt.plot(loss_logger.train_loss, label='训练损失', lw=3, color="red", alpha=1)  # 检查 val_loss
    if hasattr(loss_logger, 'val_loss'):
        plt.plot(loss_logger.val_loss, label='验证损失', lw=3, color="blue", alpha=1)
    plt.title("训练损失和验证损失", fontproperties=my_font)
    plt.legend(prop=my_font)
    plt.show()

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
    plt.legend(prop=my_font)  # 确保图例字体选择
    plt.show()
    plt.close()

    del model
    torch.cuda.empty_cache()

    return overall_precision


def objective(trial):
    model = define_model(trial)
    precision = train_and_evaluate(model, data)

    if precision is None:  # 确保只有在精度非None的情况下才记录
        logger.error(f"试验{trial.number}失败，超参数: {trial.params}")
        return 0.0  # 返回0.0或其他默认值，表示试验失败

    logger.info(f"试验{trial.number}: 当前精准率:{precision:.4%}; 最佳精准率{study.best_value:.4%}；\n当前超参数： {trial.params}")
    return precision


if __name__ == '__main__':
    study_name = 'tftmodel-precision-optimization'
    try:
        # 尝试加载现有研究
        study = optuna.load_study(
            study_name=study_name,
            storage='sqlite:///data/optuna/optuna_study.db'
        )
    except Exception as e:
        print(f"加载研究时发生错误: {e}")
        # 这里可以添加逻辑来处理研究不存在的情况，例如新建研究
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage="sqlite:///data/optuna/optuna_study.db"
        )
    study.optimize(objective, n_trials=100, n_jobs=1)

    logger.info(f"Best hyperparameters: {study.best_params}")
    logger.info(f"Best precision: {study.best_value:.4f}")
