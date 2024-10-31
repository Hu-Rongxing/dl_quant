
def test_torch():
    import torch
    import pytorch_lightning as pl

    print("PyTorch version:", torch.__version__)
    print("Is CUDA available?", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("PyTorch Lightning version:", pl.__version__)