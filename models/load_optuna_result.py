import optuna



if __name__ == '__main__':
    # 定义要读取的模型
    study = optuna.load_study(
        study_name=f"tftmodel-precision-optimization",
        storage="sqlite:///data/optuna/optuna_study.db"
    )

    print(study.best_params)
    print(study.best_value)
