from pytorch_lightning.callbacks import EarlyStopping
from darts.utils.callbacks import TFMProgressBar
from utils.model import LossLogger

loss_logger = LossLogger()
progress_bar = TFMProgressBar(
        enable_sanity_check_bar=False, enable_validation_bar=False
    )

early_stopper = EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=1e-6,
            mode="min",
    )


def get_pl_trainer_kwargs(full_training=True):
    # 提前停止：若验证损失在10个epoch内没有减少1e-6则停止
    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=1e-6,
        mode="min",
    )

    # PyTorch Lightning Trainer参数配置（可添加自定义回调）
    if full_training:
        limit_train_batches = None
        limit_val_batches = None
        max_epochs = 200
        batch_size = 64
    else:
        limit_train_batches = 2
        limit_val_batches = 2
        max_epochs = 10
        batch_size = 64

        # 仅显示训练和预测进度条
    progress_bar = TFMProgressBar(
        enable_sanity_check_bar=False, enable_validation_bar=False
    )

    pl_trainer_kwargs = {
        "gradient_clip_val": 1,  # 梯度剪裁，通过限制梯度的范围来稳定训练过程
        "max_epochs": max_epochs,
        "limit_train_batches": limit_train_batches,
        "limit_val_batches": limit_val_batches,
        "accelerator": "auto",
        "callbacks": [early_stopper, progress_bar, loss_logger],
    }

    return pl_trainer_kwargs

    # 获取优化器的参数配置


def get_optimizer_kwargs():
    # 优化器配置，默认使用Adam
    optimizer_kwargs = {
        "lr": 1e-4,
        # "optimizer_cls": torch.optim.Adam
    }
    return optimizer_kwargs


