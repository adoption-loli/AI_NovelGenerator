# main.py
# -*- coding: utf-8 -*-
import tkinter as tk
from ui import NovelGeneratorGUI
import logging
import sys
from logging.handlers import RotatingFileHandler

log_format = '[%(asctime)s.%(msecs)03d %(filename)s %(lineno)d] %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt=date_format,
                    handlers=[logging.StreamHandler(),
                              RotatingFileHandler('logs/log_file.log', maxBytes=16384, encoding='utf-8')])
# 获取配置中的 RotatingFileHandler
rotating_file_handler = None
for handler in logging.root.handlers:
    if isinstance(handler, RotatingFileHandler):
        rotating_file_handler = handler
        break


# 自定义一个类，重定向 print 的输出
# 重定向 print 输出的类
class DualOutput:
    def __init__(self, file_handler):
        self.console = sys.stdout  # 原始控制台输出
        self.file_handler = file_handler  # 日志文件处理器

    def write(self, message):
        # 写入控制台
        self.console.write(message)

        # 过滤掉包含回车符的消息，避免日志文件混乱
        if '\r' not in message and message.strip() != '':
            # 使用线程锁确保写入安全
            with self.file_handler.lock:
                self.file_handler.stream.write(message)
                self.file_handler.stream.flush()

    def flush(self):
        # 同步刷新控制台和文件的缓冲区
        self.console.flush()
        with self.file_handler.lock:
            if self.file_handler.stream:
                self.file_handler.stream.flush()


# 执行重定向
if rotating_file_handler:
    sys.stdout = DualOutput(rotating_file_handler)


def main():
    logging.info("程序启动...")
    root = tk.Tk()
    root.title("Novel Generator")
    app = NovelGeneratorGUI(root)
    log_format = '[%(asctime)s.%(msecs)03d %(filename)s %(lineno)d] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format)
    root.mainloop()


if __name__ == "__main__":
    main()
