import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib import font_manager
import optuna
from load_data.pandas_object_data import generate_wide_dataframe
from utils.logger import logger


def load_data():
    """加载特征和目标数据"""
    target_df, covariate_df = generate_wide_dataframe()
    return covariate_df, target_df


def prepare_data(covariate_df, target_df):
    """准备训练和测试数据集"""
    X = covariate_df
    y = target_df
    return train_test_split(X, y, test_size=0.2, random_state=42)


def create_model(num_leaves, learning_rate, n_estimators):
    """创建多输出分类器模型"""
    base_model = LGBMClassifier(num_leaves=num_leaves, learning_rate=learning_rate, n_estimators=n_estimators,
                                random_state=42)
    model = MultiOutputClassifier(base_model)
    return model

def preprocess_data(X):
    """数据预处理"""
    # 数值特征和类别特征的列名
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # 创建预处理管道
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # 缺失值填充
        ('scaler', StandardScaler())  # 标准化
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # 缺失值填充
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # 独热编码
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor.fit_transform(X)

def feature_selection_rfe(X_train, y_train):
    """使用递归特征消除进行特征选择"""
    model = RandomForestClassifier(random_state=42)
    rfe = RFE(model, n_features_to_select=10)  # 选择10个特征
    rfe.fit(X_train, y_train)
    return X_train.iloc[:, rfe.support_], X_train.columns[rfe.support_]

def objective(trial):
    """Optuna 超参数优化目标函数"""
    num_leaves = trial.suggest_int("num_leaves", 20, 50)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2)
    n_estimators = trial.suggest_int("n_estimators", 50, 150)

    # 准备数据集
    X_train, X_test, y_train, y_test = prepare_data(covariate_df, target_df)

    # 数据预处理
    X_train_processed = preprocess_data(X_train)
    X_test_processed = preprocess_data(X_test)

    # 特征选择
    X_train_selected, selected_features = feature_selection_rfe(X_train_processed, y_train)

    model = create_model(num_leaves, learning_rate, n_estimators)
    model.fit(X_train_selected, y_train)

    precisions = evaluate_model(model, X_test_processed, y_test)
    mean_precision = sum(precisions) / len(precisions)

    # 获取当前最佳精度
    trials = trial.study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    best_precision = max([t.value for t in trials]) if trials else None

    # 可视化精确率
    plot_precision(precisions, y_test, trial.number, mean_precision, best_precision)

    return mean_precision

def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    precisions = [
        precision_score(y_test[y_test.columns[i]], y_pred[:, i], average='macro', zero_division=0)
        for i in range(y_pred.shape[1])
    ]
    return precisions


def plot_precision(precisions, y_test, trial_number, current_precision, best_precision):
    """可视化各个标签的精确率"""
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(precisions)), precisions)
    plt.xticks(range(len(precisions)), y_test.columns, rotation=90)
    plt.title('Precision for Each Label')
    plt.xlabel('Labels')
    plt.ylabel('Precision')
    plt.grid(axis='y')

    max_precision = max(precisions) + 0.01 if precisions else 0.1

    # 在图上显示调优轮数、本轮精度和最佳精度
    plt.text(0.5, max_precision,
             f'Trial: {trial_number}\nCurrent Precision: {current_precision:.2%}\nBest Precision: {best_precision:.2%}' if best_precision is not None else f'Trial: {trial_number}\nCurrent Precision: {current_precision:.4f}\nBest Precision: N/A',
             horizontalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # 使用用户提供的字体
    font_path = 'C:/Windows/Fonts/msyh.ttc'  # 用户提供的字体文件路径
    my_font = font_manager.FontProperties(fname=font_path)

    plt.xlabel('Labels', fontproperties=my_font)
    plt.ylabel('Precision', fontproperties=my_font)
    plt.title('各标签的精确率', fontproperties=my_font)
    plt.xticks(rotation=90, fontproperties=my_font)

    # 保存图表
    plt.tight_layout()
    # plt.savefig(f'precision_chart_trial_{trial_number}.png')  # 保存每轮的图表
    plt.show()


def main():
    """主函数"""
    global covariate_df, target_df
    covariate_df, target_df = load_data()  # 在全局范围内加载数据

    # 使用Optuna进行超参数调优
    study = optuna.create_study(
        direction="maximize",
        study_name="lgbmclassifier-precision-optimization",
        storage="sqlite:///data/optuna/optuna_study.db",
        load_if_exists=True  # 如果数据库存在则加载
    )
    study.remove_all_trials()
    study.optimize(objective, n_trials=10)

    logger.info("最佳参数: %s", study.best_params)
    logger.info("最佳精确率: %s", study.best_value)


if __name__ == "__main__":
    main()