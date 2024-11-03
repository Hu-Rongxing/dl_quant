import os
import sys
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path


def start_jupyter_lab():
    # 设置工作目录
    work_dir = Path(__file__).parent.parent
    os.chdir(work_dir)
    os.environ['JUPYTER_CONFIG_DIR'] = str(work_dir)
    print(work_dir)

    # 设置支持中文的字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    print(f"matplotlib显示字体已设置为中文。")

    subprocess.run([sys.executable, "-m", "jupyter", "lab"])

if __name__ == "__main__":
    start_jupyter_lab()