# main.py
# -*- coding: utf-8 -*-
import tkinter as tk
from ui import NovelGeneratorGUI
import logging
import sys

log_format = '[%(asctime)s.%(msecs)03d %(filename)s %(lineno)d] %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format,
                    handlers=[logging.StreamHandler(), logging.FileHandler('log_file.log')])
# 获取 logging 的 FileHandler
file_handler = None
for handler in logging.root.handlers:
    if isinstance(handler, logging.FileHandler):
        file_handler = handler
        break


# 自定义一个类，重定向 print 的输出
class DualOutput:
    def __init__(self, file_handler):
        self.console = sys.stdout  # 控制台输出
        self.file_handler = file_handler  # logging 的 FileHandler

    def write(self, message):
        # 写入控制台
        self.console.write(message)
        # 通过 logging 的 FileHandler 写入文件
        if '\r' not in message:
            self.file_handler.stream.write(message)

    def flush(self):
        # 刷新控制台和文件的缓冲区
        self.console.flush()
        self.file_handler.stream.flush()


# 将 sys.stdout 重定向到自定义的 DualOutput 类
if file_handler:
    sys.stdout = DualOutput(file_handler)


def main():
    logging.info("程序启动...")
    root = tk.Tk()
    root.title("Novel Generator")
    app = NovelGeneratorGUI(root)
    log_format = '[%(asctime)s.%(msecs)03d %(filename)s %(lineno)d] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt=date_format)
    root.mainloop()


if __name__ == "__main__":
    main()
