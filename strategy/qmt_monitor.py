import psutil
import subprocess
from pathlib import Path
from pywinauto import Application, findwindows
from pywinauto.findwindows import ElementNotFoundError
import win32gui
import win32con
import pyautogui
import time
import ctypes

from utils.logger import logger
from config import config
from multiprocessing import Lock, Queue, Process


# 检查并导入 OpenCV
try:
    import cv2
except ImportError:
    raise ImportError("OpenCV 无法导入。请安装它，命令为 'pip install opencv-python'")


class WindowRegexFinder:
    def __init__(self, regex_pattern: str):
        self.regex_pattern = regex_pattern
        self.app = None
        self.window = None
        self.handle = None

    def get_scaling_factor(self):
        # 获取 Windows 的 DPI 缩放系数
        dpi_scale = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100.0
        logger.debug(f"DPI 缩放系数: {dpi_scale}")
        return dpi_scale

    def find_window(self) -> None:
        try:
            # 使用正则表达式查找与窗口标题匹配的所有窗口
            windows = findwindows.find_windows(title_re=self.regex_pattern)
            if windows:
                self.handle = windows[0]
                logger.debug(f"找到窗口句柄: {self.handle}")
                self.app = Application(backend="uia").connect(handle=self.handle)
                self.window = self.app.window(handle=self.handle)
            else:
                raise Exception(f"未找到符合模式的窗口: {self.regex_pattern}")
        except ElementNotFoundError as e:
            raise Exception(f"查找窗口失败: {e}")

    def bring_window_to_top(self) -> None:
        if not self.handle:
            raise Exception("未找到窗口句柄。请先调用 find_window() 方法。")

        try:
            win32gui.SetForegroundWindow(self.handle)
            win32gui.ShowWindow(self.handle, win32con.SW_NORMAL)
            self.app = Application(backend="uia").connect(handle=self.handle)
            self.window = self.app.window(handle=self.handle)
            logger.debug(f"窗口 {self.handle} 已置顶并聚焦。")
        except Exception as e:
            raise Exception(f"无法将窗口置顶或连接：{e}")

    def find_and_click_button(self, button_text: str) -> None:
        """通过文本查找并点击按钮"""
        if not self.window:
            raise Exception("未设置窗口。请先调用 find_window() 和 bring_window_to_top() 方法。")

        try:
            button = self.window.child_window(title=button_text, control_type="Button")
            if button.exists(timeout=5):
                button.click_input()
                logger.debug("按钮点击成功！")
            else:
                raise Exception(f"未找到文本为 '{button_text}' 的按钮！")
        except ElementNotFoundError as e:
            raise Exception(f"未找到文本为 '{button_text}' 的按钮：{e}")

    def find_and_click_image_button(self, image_path: str) -> None:
        """通过图像查找并点击按钮"""


        logger.debug(f"查找路径 {image_path} 中的按钮图像")

        try:
            # 确保图像加载正确
            image = cv2.imread(image_path)
            if image is None:
                raise Exception(f"图像未加载。检查路径: {image_path}")

            # 获取系统 DPI 缩放系数
            scaling_factor = self.get_scaling_factor()

            # 使用调整后的缩放系数在屏幕上定位按钮
            button_location = pyautogui.locateOnScreen(image_path, confidence=0.8)
            logger.debug(f"原始按钮位置: {button_location}")

            if button_location:
                button_point = pyautogui.center(button_location)
                pyautogui.moveTo(button_point)  # 不进行缩放调整
                logger.debug(f"移动到按钮位置 (未缩放): {button_point}")
                time.sleep(5)  # 暂停以验证位置是否正确
                pyautogui.click()
                logger.debug("图像按钮点击成功！")
            else:
                raise Exception(f"屏幕上未找到按钮。图像路径: {image_path}")
        except Exception as e:
            logger.exception(f"点击图像按钮时出错：{e}")



class ProgramMonitor:
    MINIXT_PROCESS_NAME = "XtMiniQmt.exe"
    LOGIN_PROCESS_NAME = "XtItClient.exe"
    _instance = None
    lock = Lock()
    task_queue = Queue()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        # 加载配置文件
        self.program_name = config['xt_client'].get('program_dir')
        if not self.program_name:
            raise KeyError('在"xt_client"节中未找到键"program_dir"。')

        self.check_interval = config.getint('xt_client', 'check_interval', fallback=60)

    def is_program_running(self):
        """检查是否有指定名称的程序正在运行"""
        for proc in psutil.process_iter(['name']):
            try:
                if proc.info['name'] == self.MINIXT_PROCESS_NAME:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return False

    def is_login_progress_running(self):
        """检查是否有指定名称的登录进程正在运行"""
        for proc in psutil.process_iter(['name']):
            try:
                if proc.info['name'] == self.LOGIN_PROCESS_NAME:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return False

    def start_program(self, auto_login=True):
        """启动指定路径的程序，确保操作在同一时刻只被一个进程执行"""
        with self.lock:
            if self.is_program_running():
                logger.info("迅投程序已运行，无需启动。")
                return
            try:
                subprocess.Popen(self.program_name)
                logger.info(f"程序 {self.program_name} 已启动。")
            except Exception as e:
                logger.error(f"无法启动程序 {self.program_name}：{e}")
                return

                # 点击登录
            time.sleep(20)
            if self.is_login_progress_running():
                finder = WindowRegexFinder(r"e海方舟-量化交易版[.\d ]+")
                # 查找窗口句柄
                finder.find_window()
                # 将窗口置顶
                finder.bring_window_to_top()
                # 查找并点击图像按钮
                if not auto_login:
                    path = Path(__file__).parent.parent / "config/xt_login_button.PNG"
                    try:
                        finder.find_and_click_image_button(str(path))
                    except Exception as e:
                        logger.error(e)
                time.sleep(15)

    def stop_program(self):
        """停止指定名称的程序，确保操作在同一时刻只被一个进程执行"""
        with self.lock:
            for proc in psutil.process_iter(['name']):
                try:
                    if proc.info['name'] == self.MINIXT_PROCESS_NAME:
                        proc.terminate()
                        logger.info(f"程序 {self.MINIXT_PROCESS_NAME} 已停止。")
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                    logger.error(f"无法停止程序 {self.MINIXT_PROCESS_NAME}：{e}")
                try:
                    if proc.info['name'] == self.LOGIN_PROCESS_NAME:
                        proc.terminate()
                        logger.info(f"程序 {self.LOGIN_PROCESS_NAME} 已停止。")
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                    logger.error(f"无法停止程序 {self.LOGIN_PROCESS_NAME}：{e}")

    def restart_program(self):
        """重启指定名称的程序，确保操作在同一时刻只被一个进程执行"""
        logger.info("正在重启程序...")
        self.stop_program()
        time.sleep(5)  # 等待进程完全结束
        self.start_program()


    def monitor(self):
        """开始监控程序状态"""
        while True:
            if not self.is_program_running():
                logger.info(f"检测到 {self.MINIXT_PROCESS_NAME} 未启动，正在启动...")
                self.start_program()
            else:
                logger.info(f"{self.MINIXT_PROCESS_NAME} 正在运行。")

                # 每隔指定时间间隔检测一次
            time.sleep(self.check_interval)

    @classmethod
    def add_task(cls, task, *args, **kwargs):
        """将任务添加到队列"""
        cls.task_queue.put((task, args, kwargs))

    @classmethod
    def worker(cls):
        """从队列中获取任务并执行，确保任务的顺序执行"""
        while True:
            task, args, kwargs = cls.task_queue.get()
            logger.info(f"执行任务: {task.__name__}")
            task(*args, **kwargs)
            cls.task_queue.task_done()


def start_xt_client():
    try:
        xt_client = ProgramMonitor()
        xt_client.start_program()
        # xtdata.run()
        return xt_client
    except Exception as e:
        logger.error(e)
        xt_client = ProgramMonitor()
        xt_client.restart_program()
        return xt_client


if __name__ == "__main__":
    monitor = ProgramMonitor()
    monitor.start_program()

    # 创建一个额外的进程以监控队列中的任务执行
    worker_process = Process(target=ProgramMonitor.worker)
    worker_process.daemon = True  # 设置为守护进程
    worker_process.start()

    # 添加任务来模拟多进程环境中的任务调用
    ProgramMonitor.add_task(monitor.monitor)
    ProgramMonitor.add_task(monitor.start_program)
    ProgramMonitor.add_task(monitor.stop_program)
    ProgramMonitor.add_task(monitor.restart_program)

    worker_process.join()  # 等待工作进程执行完队列中的所有任务