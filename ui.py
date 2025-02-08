# ui.py
# -*- coding: utf-8 -*-

import logging
import os
import threading
import time

import customtkinter as ctk
from tkinter import filedialog, messagebox
import traceback
from config_manager import load_config, save_config
from utils import read_file, save_string_to_txt, clear_file_content
from novel_generator import (
    Novel_setting_generate,
    Novel_directory_generate,
    generate_chapter_draft,
    finalize_chapter,
    import_knowledge_file,
    clear_vector_store,
    get_last_n_chapters_text,
    summarize_recent_chapters
)
from consistency_checker import check_consistency


def log_error(message: str):
    """
    用于打印详细的错误信息和堆栈信息。
    """
    logging.error(f"{message}\n{traceback.format_exc()}")


# 设置全局主题和颜色
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class NovelGeneratorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Novel Generator GUI")

        # 防止因 icon.ico 不存在导致程序崩溃
        try:
            if os.path.exists("icon.ico"):
                self.master.iconbitmap("icon.ico")
        except Exception:
            pass

        # 配置窗口大小
        self.master.geometry("1350x840")

        # 配置持久化
        self.config_file = "config.json"
        self.loaded_config = load_config(self.config_file)

        # ========== 主要的属性变量 ========== 
        self.api_key_var = ctk.StringVar(value=self.loaded_config.get("api_key", ""))
        self.base_url_var = ctk.StringVar(value=self.loaded_config.get("base_url", "https://api.agicto.cn/v1"))
        self.interface_format_var = ctk.StringVar(value=self.loaded_config.get("interface_format", "OpenAI"))
        self.model_name_var = ctk.StringVar(value=self.loaded_config.get("model_name", "gpt-4o-mini"))

        self.embedding_url_var = ctk.StringVar(value=self.loaded_config.get("embedding_url", ""))
        self.embedding_model_name_var = ctk.StringVar(value=self.loaded_config.get("embedding_model_name", ""))

        self.temperature_var = ctk.DoubleVar(value=self.loaded_config.get("temperature", 0.7))
        self.topic_default = self.loaded_config.get("topic", "")
        self.genre_var = ctk.StringVar(value=self.loaded_config.get("genre", "玄幻"))
        self.num_chapters_var = ctk.IntVar(value=self.loaded_config.get("num_chapters", 10))
        self.word_number_var = ctk.IntVar(value=self.loaded_config.get("word_number", 3000))
        self.filepath_var = ctk.StringVar(value=self.loaded_config.get("filepath", ""))

        self.chapter_num_var = ctk.IntVar(value=1)

        # ========== 主容器使用 TabView ========== 
        self.tabview = ctk.CTkTabview(self.master, width=1200, height=800)
        self.tabview.pack(fill="both", expand=True)

        # 创建各个Tab
        self.main_tab = self.tabview.add("Main Functions")
        self.setting_tab = self.tabview.add("设定")
        self.directory_tab = self.tabview.add("目录")
        self.character_tab = self.tabview.add("章节状态")
        self.summary_tab = self.tabview.add("全文摘要")
        self.chapters_view_tab = self.tabview.add("章节管理")

        # 构建各个 Tab 的布局
        self.build_main_tab()
        self.build_setting_tab()
        self.build_directory_tab()
        self.build_character_tab()
        self.build_summary_tab()
        self.build_chapters_tab()  # 新增

    # ------------------ 统一异常处理方法 ------------------ 
    def handle_exception(self, context: str):
        full_message = f"{context}\n{traceback.format_exc()}"
        logging.error(full_message)
        self.safe_log(full_message)

    # ------------------ 主功能 Tab ------------------ 
    def build_main_tab(self):
        """
        主Tab分为左右两栏：
          左侧：本章内容、Step按钮、日志
          右侧：配置区域(带边框) + 保存/加载配置 + 小说参数 + 可选功能按钮
        """
        self.main_tab.rowconfigure(0, weight=1)
        self.main_tab.columnconfigure(0, weight=1)
        self.main_tab.columnconfigure(1, weight=0)

        # 左侧Frame
        self.left_frame = ctk.CTkFrame(self.main_tab)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        # 右侧Frame
        self.right_frame = ctk.CTkFrame(self.main_tab)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)

        # 左侧布局
        self.build_left_layout()
        # 右侧布局
        self.build_right_layout()

    def build_left_layout(self):
        """
        左侧布局:
         row=0 -> “本章内容”文本框 (chapter_result)
         row=1 -> Step1~4按钮
         row=2 -> “输出日志”标题
         row=3 -> “输出日志”文本框 (log_text)
        """
        self.left_frame.grid_rowconfigure(0, weight=0)
        self.left_frame.grid_rowconfigure(1, weight=2)
        self.left_frame.grid_rowconfigure(2, weight=0)
        self.left_frame.grid_rowconfigure(3, weight=0)
        self.left_frame.grid_rowconfigure(4, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)

        # ========== 本章内容 ========== 
        chapter_label = ctk.CTkLabel(
            self.left_frame,
            text="本章内容 (可编辑)",
            font=("Microsoft YaHei", 14)
        )
        chapter_label.grid(row=0, column=0, padx=5, pady=(5, 0), sticky="w")

        # 章节草稿：可编辑
        self.chapter_result = ctk.CTkTextbox(
            self.left_frame,
            wrap="word",
            font=("Microsoft YaHei", 14)
        )
        self.chapter_result.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))

        # ========== Step1~4按钮 ========== 
        self.build_step_buttons_area()

        # ========== 输出日志 label ========== 
        log_label = ctk.CTkLabel(
            self.left_frame,
            text="输出日志 (只读)",
            font=("Microsoft YaHei", 14)
        )
        log_label.grid(row=3, column=0, padx=5, pady=(5, 0), sticky="w")

        # ========== 日志：只读 ========== 
        self.log_text = ctk.CTkTextbox(
            self.left_frame,
            wrap="word",
            font=("Microsoft YaHei", 12)
        )
        self.log_text.grid(row=4, column=0, sticky="nsew", padx=5, pady=(0, 5))
        self.log_text.configure(state="disabled")

    def build_step_buttons_area(self):
        """
        在左侧，仅放 Step1~Step4 四个按钮
        """
        self.step_buttons_frame = ctk.CTkFrame(self.left_frame)
        self.step_buttons_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        # 四个按钮平分横向空间
        self.step_buttons_frame.columnconfigure((0, 1, 2, 3), weight=1)

        self.btn_generate_setting = ctk.CTkButton(
            self.step_buttons_frame,
            text="Step1. 生成设定",
            command=self.generate_novel_setting_ui,
            font=("Microsoft YaHei", 12)
        )
        self.btn_generate_setting.grid(row=0, column=0, padx=5, pady=2, sticky="ew")

        self.btn_generate_directory = ctk.CTkButton(
            self.step_buttons_frame,
            text="Step2. 生成目录",
            command=self.generate_novel_directory_ui,
            font=("Microsoft YaHei", 12)
        )
        self.btn_generate_directory.grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        self.btn_generate_chapter = ctk.CTkButton(
            self.step_buttons_frame,
            text="Step3. 生成草稿",
            command=self.generate_chapter_draft_ui,
            font=("Microsoft YaHei", 12)
        )
        self.btn_generate_chapter.grid(row=0, column=2, padx=5, pady=2, sticky="ew")

        self.btn_finalize_chapter = ctk.CTkButton(
            self.step_buttons_frame,
            text="Step4. 定稿章节",
            command=self.finalize_chapter_ui,
            font=("Microsoft YaHei", 12)
        )
        self.btn_finalize_chapter.grid(row=0, column=3, padx=5, pady=2, sticky="ew")

    def build_right_layout(self):
        """
        右侧布局，包含：
          row=0 -> 带边框的配置区 (TabView + 保存/加载配置按钮)
          row=1 -> 小说参数区域
          row=2 -> 可选功能按钮 (一致性审校 / 导入知识库 / 清空向量库 / 查看剧情要点)
        """
        self.right_frame.grid_rowconfigure(0, weight=0)
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_rowconfigure(2, weight=0)
        self.right_frame.columnconfigure(0, weight=1)

        # 1) 配置区
        self.config_frame = ctk.CTkFrame(
            self.right_frame,
            corner_radius=10,
            border_width=2,
            border_color="gray"
        )
        self.config_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.config_frame.columnconfigure(0, weight=1)

        self.build_config_tabview()  # LLM、Embedding等配置
        self.build_main_buttons_area()  # 保存/加载配置

        # 2) 小说参数
        self.build_novel_params_area(start_row=1)

        # 3) 可选功能按钮
        self.build_optional_buttons_area(start_row=2)

    # ------------------ 可选功能按钮区域（右下） ------------------ 
    def build_optional_buttons_area(self, start_row=2):
        """
        放在右侧的最下方：包括 一致性审校、导入知识库、清空向量库、查看剧情要点
        """
        self.optional_btn_frame = ctk.CTkFrame(self.right_frame)
        self.optional_btn_frame.grid(row=start_row, column=0, sticky="ew", padx=5, pady=5)
        self.optional_btn_frame.columnconfigure((0, 1, 2, 3, 4), weight=1)

        self.btn_check_consistency = ctk.CTkButton(
            self.optional_btn_frame,
            text="一致性审校",
            command=self.do_consistency_check,
            font=("Microsoft YaHei", 12)
        )
        self.btn_check_consistency.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.btn_import_knowledge = ctk.CTkButton(
            self.optional_btn_frame,
            text="导入知识库",
            command=self.import_knowledge_handler,
            font=("Microsoft YaHei", 12)
        )
        self.btn_import_knowledge.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.btn_clear_vectorstore = ctk.CTkButton(
            self.optional_btn_frame,
            text="清空向量库",
            fg_color="red",
            command=self.clear_vectorstore_handler,
            font=("Microsoft YaHei", 12)
        )
        self.btn_clear_vectorstore.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.plot_arcs_btn = ctk.CTkButton(
            self.optional_btn_frame,
            text="查看剧情要点",
            command=self.show_plot_arcs_ui,
            font=("Microsoft YaHei", 12)
        )
        self.plot_arcs_btn.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        self.one_key_gen = ctk.CTkButton(
            self.optional_btn_frame,
            text="一键从头生成全文",
            command=self.on_key_gen_handler,
            font=("Microsoft YaHei", 12)
        )
        self.one_key_gen.grid(row=0, column=4, padx=5, pady=5, sticky="ew")

    # ========== 配置区域（TabView） ==========
    def build_config_tabview(self):
        self.config_tabview = ctk.CTkTabview(self.config_frame, width=600, height=200)
        self.config_tabview.grid(row=0, column=0, sticky="we", padx=5, pady=5)

        self.ai_config_tab = self.config_tabview.add("LLM Model settings")
        self.embeddings_config_tab = self.config_tabview.add("Embedding settings")

        self.build_ai_config_tab()
        self.build_embeddings_config_tab()

    def build_ai_config_tab(self):
        def on_interface_format_changed(new_value):
            if new_value == "Ollama":
                self.base_url_var.set("http://localhost:11434/v1")
                self.embedding_url_var.set("http://localhost:11434/api")
            elif new_value == "ML Studio":
                self.base_url_var.set("http://localhost:1234/v1")
                self.embedding_url_var.set("http://localhost:1234/api")
            elif new_value == "OpenAI":
                self.base_url_var.set("https://api.openai.com/v1")
                self.embedding_url_var.set("https://api.openai.com/v1")

        for i in range(5):
            self.ai_config_tab.grid_rowconfigure(i, weight=0)
        self.ai_config_tab.grid_columnconfigure(0, weight=0)
        self.ai_config_tab.grid_columnconfigure(1, weight=1)
        self.ai_config_tab.grid_columnconfigure(2, weight=0)  # for temp label

        api_key_label = ctk.CTkLabel(
            self.ai_config_tab,
            text="API Key:",
            font=("Microsoft YaHei", 12)
        )
        api_key_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        api_key_entry = ctk.CTkEntry(
            self.ai_config_tab,
            textvariable=self.api_key_var,
            font=("Microsoft YaHei", 12)
        )
        api_key_entry.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        base_url_label = ctk.CTkLabel(
            self.ai_config_tab,
            text="Base URL:",
            font=("Microsoft YaHei", 12)
        )
        base_url_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        base_url_entry = ctk.CTkEntry(
            self.ai_config_tab,
            textvariable=self.base_url_var,
            font=("Microsoft YaHei", 12)
        )
        base_url_entry.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        interface_label = ctk.CTkLabel(
            self.ai_config_tab,
            text="接口格式:",
            font=("Microsoft YaHei", 12)
        )
        interface_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
        interface_options = ["OpenAI", "Ollama", "ML Studio"]
        interface_dropdown = ctk.CTkOptionMenu(
            self.ai_config_tab,
            values=interface_options,
            variable=self.interface_format_var,
            command=on_interface_format_changed,
            font=("Microsoft YaHei", 12)
        )
        interface_dropdown.grid(row=2, column=1, padx=5, pady=5, sticky="nsew")

        model_name_label = ctk.CTkLabel(
            self.ai_config_tab,
            text="Model Name:",
            font=("Microsoft YaHei", 12)
        )
        model_name_label.grid(row=3, column=0, padx=5, pady=5, sticky="e")
        model_name_entry = ctk.CTkEntry(
            self.ai_config_tab,
            textvariable=self.model_name_var,
            font=("Microsoft YaHei", 12)
        )
        model_name_entry.grid(row=3, column=1, padx=5, pady=5, sticky="nsew")

        temp_label = ctk.CTkLabel(
            self.ai_config_tab,
            text="Temperature:",
            font=("Microsoft YaHei", 12)
        )
        temp_label.grid(row=4, column=0, padx=5, pady=5, sticky="e")

        def update_temp_label(value):
            self.temp_value_label.configure(text=f"{float(value):.2f}")

        temp_scale = ctk.CTkSlider(
            self.ai_config_tab,
            from_=0.0, to=2.0,
            number_of_steps=100,
            command=update_temp_label,
            variable=self.temperature_var
        )
        temp_scale.grid(row=4, column=1, padx=5, pady=5, sticky="we")

        self.temp_value_label = ctk.CTkLabel(
            self.ai_config_tab,
            text=f"{self.temperature_var.get():.2f}",
            font=("Microsoft YaHei", 12)
        )
        self.temp_value_label.grid(row=4, column=2, padx=1, pady=1, sticky="w")

    def build_embeddings_config_tab(self):
        for i in range(2):
            self.embeddings_config_tab.grid_rowconfigure(i, weight=0)
        self.embeddings_config_tab.grid_columnconfigure(0, weight=0)
        self.embeddings_config_tab.grid_columnconfigure(1, weight=1)

        embedding_url_label = ctk.CTkLabel(
            self.embeddings_config_tab,
            text="Embedding URL:",
            font=("Microsoft YaHei", 12)
        )
        embedding_url_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        embedding_url_entry = ctk.CTkEntry(
            self.embeddings_config_tab,
            textvariable=self.embedding_url_var,
            font=("Microsoft YaHei", 12)
        )
        embedding_url_entry.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        emb_model_name_label = ctk.CTkLabel(
            self.embeddings_config_tab,
            text="Embedding Model Name:",
            font=("Microsoft YaHei", 12)
        )
        emb_model_name_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        emb_model_name_entry = ctk.CTkEntry(
            self.embeddings_config_tab,
            textvariable=self.embedding_model_name_var,
            font=("Microsoft YaHei", 12)
        )
        emb_model_name_entry.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

    # ========== 保存/加载 配置按钮区域 ==========
    def build_main_buttons_area(self):
        """
        放置在带边框配置区(config_frame)内部，位于TabView下方
        """
        self.btn_frame_config = ctk.CTkFrame(self.config_frame)
        self.btn_frame_config.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.btn_frame_config.columnconfigure(0, weight=1)
        self.btn_frame_config.columnconfigure(1, weight=1)

        save_config_btn = ctk.CTkButton(
            self.btn_frame_config,
            text="保存配置",
            command=self.save_config_btn,
            font=("Microsoft YaHei", 12)
        )
        save_config_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        load_config_btn = ctk.CTkButton(
            self.btn_frame_config,
            text="加载配置",
            command=self.load_config_btn,
            font=("Microsoft YaHei", 12)
        )
        load_config_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    # ========== 小说参数区域 ==========
    def build_novel_params_area(self, start_row=1):
        """
        右侧下方区域: 输入主题, 类型, 章节数, 字数, 保存路径, 指导信息等
        """
        self.params_frame = ctk.CTkScrollableFrame(
            self.right_frame,
            orientation="vertical"  # 默认垂直滚动
        )
        self.params_frame.grid(row=start_row, column=0, sticky="nsew", padx=5, pady=5)
        self.params_frame.columnconfigure(1, weight=1)

        # 主题(Topic)
        topic_label = ctk.CTkLabel(
            self.params_frame,
            text="主题(Topic):",
            font=("Microsoft YaHei", 12)
        )
        topic_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.topic_text = ctk.CTkTextbox(
            self.params_frame,
            width=200,
            height=80,
            wrap="word",
            font=("Microsoft YaHei", 12)
        )
        self.topic_text.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        if self.topic_default:
            self.topic_text.insert("0.0", self.topic_default)

        # 类型(Genre)
        genre_label = ctk.CTkLabel(
            self.params_frame,
            text="类型(Genre):",
            font=("Microsoft YaHei", 12)
        )
        genre_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        genre_entry = ctk.CTkEntry(
            self.params_frame,
            textvariable=self.genre_var,
            font=("Microsoft YaHei", 12)
        )
        genre_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # 章节数
        num_chapters_label = ctk.CTkLabel(
            self.params_frame,
            text="章节数:",
            font=("Microsoft YaHei", 12)
        )
        num_chapters_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
        num_chapters_entry = ctk.CTkEntry(
            self.params_frame,
            textvariable=self.num_chapters_var,
            width=80,
            font=("Microsoft YaHei", 12)
        )
        num_chapters_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # 每章字数
        word_number_label = ctk.CTkLabel(
            self.params_frame,
            text="每章字数:",
            font=("Microsoft YaHei", 12)
        )
        word_number_label.grid(row=3, column=0, padx=5, pady=5, sticky="e")
        word_number_entry = ctk.CTkEntry(
            self.params_frame,
            textvariable=self.word_number_var,
            width=80,
            font=("Microsoft YaHei", 12)
        )
        word_number_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # 保存路径
        filepath_label = ctk.CTkLabel(
            self.params_frame,
            text="保存路径:",
            font=("Microsoft YaHei", 12)
        )
        filepath_label.grid(row=4, column=0, padx=5, pady=5, sticky="e")

        self.filepath_frame = ctk.CTkFrame(self.params_frame)
        self.filepath_frame.grid(row=4, column=1, padx=5, pady=5, sticky="nsew")
        self.filepath_frame.columnconfigure(0, weight=1)

        filepath_entry = ctk.CTkEntry(
            self.filepath_frame,
            textvariable=self.filepath_var,
            font=("Microsoft YaHei", 12)
        )
        filepath_entry.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        browse_btn = ctk.CTkButton(
            self.filepath_frame,
            text="浏览...",
            command=self.browse_folder,
            width=60,
            font=("Microsoft YaHei", 12)
        )
        browse_btn.grid(row=0, column=1, padx=5, pady=5, sticky="e")

        # 章节号
        chapter_num_label = ctk.CTkLabel(
            self.params_frame,
            text="章节号:",
            font=("Microsoft YaHei", 12)
        )
        chapter_num_label.grid(row=5, column=0, padx=5, pady=5, sticky="e")
        chapter_num_entry = ctk.CTkEntry(
            self.params_frame,
            textvariable=self.chapter_num_var,
            width=80,
            font=("Microsoft YaHei", 12)
        )
        chapter_num_entry.grid(row=5, column=1, padx=5, pady=5, sticky="w")

        # 用户指导
        guide_label = ctk.CTkLabel(
            self.params_frame,
            text="本章指导:",
            font=("Microsoft YaHei", 12)
        )
        guide_label.grid(row=6, column=0, padx=5, pady=5, sticky="ne")
        self.user_guide_text = ctk.CTkTextbox(
            self.params_frame,
            width=200,
            height=80,
            wrap="word",
            font=("Microsoft YaHei", 12),
        )
        self.user_guide_text.insert("1.0", "模仿刘慈欣的相关作品里的文风进行编写，内容要通俗易懂，不要出现太多科学术语")
        self.user_guide_text.grid(row=6, column=1, padx=5, pady=5, sticky="nsew")

    # ---- -------------- 其他Tab的构建 ------------------
    def build_setting_tab(self):
        self.setting_tab.rowconfigure(0, weight=0)
        self.setting_tab.rowconfigure(1, weight=1)
        self.setting_tab.columnconfigure(0, weight=1)

        load_btn = ctk.CTkButton(
            self.setting_tab,
            text="加载 Novel_setting.txt",
            command=self.load_novel_setting,
            font=("Microsoft YaHei", 12)
        )
        load_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        save_btn = ctk.CTkButton(
            self.setting_tab,
            text="保存修改",
            command=self.save_novel_setting,
            font=("Microsoft YaHei", 12)
        )
        save_btn.grid(row=0, column=0, padx=5, pady=5, sticky="e")

        self.setting_text = ctk.CTkTextbox(
            self.setting_tab,
            wrap="word",
            font=("Microsoft YaHei", 12)
        )
        self.setting_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    def build_directory_tab(self):
        self.directory_tab.rowconfigure(0, weight=0)
        self.directory_tab.rowconfigure(1, weight=1)
        self.directory_tab.columnconfigure(0, weight=1)

        load_btn = ctk.CTkButton(
            self.directory_tab,
            text="加载 Novel_directory.txt",
            command=self.load_novel_directory,
            font=("Microsoft YaHei", 12)
        )
        load_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        save_btn = ctk.CTkButton(
            self.directory_tab,
            text="保存修改",
            command=self.save_novel_directory,
            font=("Microsoft YaHei", 12)
        )
        save_btn.grid(row=0, column=0, padx=5, pady=5, sticky="e")

        self.directory_text = ctk.CTkTextbox(
            self.directory_tab,
            wrap="word",
            font=("Microsoft YaHei", 12)
        )
        self.directory_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    def build_character_tab(self):
        self.character_tab.rowconfigure(0, weight=0)
        self.character_tab.rowconfigure(1, weight=1)
        self.character_tab.columnconfigure(0, weight=1)

        load_btn = ctk.CTkButton(
            self.character_tab,
            text="加载 character_state.txt",
            command=self.load_character_state,
            font=("Microsoft YaHei", 12)
        )
        load_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        save_btn = ctk.CTkButton(
            self.character_tab,
            text="保存修改",
            command=self.save_character_state,
            font=("Microsoft YaHei", 12)
        )
        save_btn.grid(row=0, column=0, padx=5, pady=5, sticky="e")

        self.character_text = ctk.CTkTextbox(
            self.character_tab,
            wrap="word",
            font=("Microsoft YaHei", 12)
        )
        self.character_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    def build_summary_tab(self):
        self.summary_tab.rowconfigure(0, weight=0)
        self.summary_tab.rowconfigure(1, weight=1)
        self.summary_tab.columnconfigure(0, weight=1)

        load_btn = ctk.CTkButton(
            self.summary_tab,
            text="加载 global_summary.txt",
            command=self.load_global_summary,
            font=("Microsoft YaHei", 12)
        )
        load_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        save_btn = ctk.CTkButton(
            self.summary_tab,
            text="保存修改",
            command=self.save_global_summary,
            font=("Microsoft YaHei", 12)
        )
        save_btn.grid(row=0, column=0, padx=5, pady=5, sticky="e")

        self.summary_text = ctk.CTkTextbox(
            self.summary_tab,
            wrap="word",
            font=("Microsoft YaHei", 12)
        )
        self.summary_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    def build_chapters_tab(self):
        """
        新增的 Tab，用于查看、编辑和保存已生成的各章节内容。
        """
        self.chapters_view_tab.rowconfigure(0, weight=0)
        self.chapters_view_tab.rowconfigure(1, weight=1)
        self.chapters_view_tab.columnconfigure(0, weight=1)

        top_frame = ctk.CTkFrame(self.chapters_view_tab)
        top_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        top_frame.columnconfigure(0, weight=0)
        top_frame.columnconfigure(1, weight=0)
        top_frame.columnconfigure(2, weight=0)
        top_frame.columnconfigure(3, weight=0)
        top_frame.columnconfigure(4, weight=1)

        prev_btn = ctk.CTkButton(top_frame, text="<< 上一章", command=self.prev_chapter, font=("Microsoft YaHei", 12))
        prev_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        next_btn = ctk.CTkButton(top_frame, text="下一章 >>", command=self.next_chapter, font=("Microsoft YaHei", 12))
        next_btn.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.chapter_select_var = ctk.StringVar(value="")
        self.chapter_select_menu = ctk.CTkOptionMenu(
            top_frame,
            values=[],
            variable=self.chapter_select_var,
            command=self.on_chapter_selected,
            font=("Microsoft YaHei", 12)
        )
        self.chapter_select_menu.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        save_btn = ctk.CTkButton(top_frame, text="保存修改", command=self.save_current_chapter,
                                 font=("Microsoft YaHei", 12))
        save_btn.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        refresh_btn = ctk.CTkButton(top_frame, text="刷新章节列表", command=self.refresh_chapters_list,
                                    font=("Microsoft YaHei", 12))
        refresh_btn.grid(row=0, column=4, padx=5, pady=5, sticky="e")

        self.chapter_view_text = ctk.CTkTextbox(self.chapters_view_tab, wrap="word", font=("Microsoft YaHei", 12))
        self.chapter_view_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.chapters_list = []
        self.refresh_chapters_list()

    # ------------------ 章节管理辅助方法 ------------------
    def refresh_chapters_list(self):
        filepath = self.filepath_var.get().strip()
        chapters_dir = os.path.join(filepath, "chapters")
        if not os.path.exists(chapters_dir):
            self.safe_log("尚未找到 chapters 文件夹，请先生成章节或检查保存路径。")
            self.chapter_select_menu.configure(values=[])
            return

        all_files = os.listdir(chapters_dir)
        chapter_nums = []
        for f in all_files:
            if f.startswith("chapter_") and f.endswith(".txt"):
                number_part = f.replace("chapter_", "").replace(".txt", "")
                if number_part.isdigit():
                    chapter_nums.append(number_part)

        chapter_nums.sort(key=lambda x: int(x))
        self.chapters_list = chapter_nums
        self.chapter_select_menu.configure(values=self.chapters_list)

        current_selected = self.chapter_select_var.get()
        if current_selected not in self.chapters_list:
            if self.chapters_list:
                self.chapter_select_var.set(self.chapters_list[0])
                self.load_chapter_content(self.chapters_list[0])
            else:
                self.chapter_select_var.set("")
                self.chapter_view_text.delete("0.0", "end")

    def on_chapter_selected(self, value):
        self.load_chapter_content(value)

    def load_chapter_content(self, chapter_number_str):
        if not chapter_number_str:
            return

        filepath = self.filepath_var.get().strip()
        chapter_file = os.path.join(filepath, "chapters", f"chapter_{chapter_number_str}.txt")
        if not os.path.exists(chapter_file):
            self.safe_log(f"章节文件 {chapter_file} 不存在！")
            return

        content = read_file(chapter_file)
        self.chapter_view_text.delete("0.0", "end")
        self.chapter_view_text.insert("0.0", content)

    def save_current_chapter(self):
        chapter_number_str = self.chapter_select_var.get()
        if not chapter_number_str:
            messagebox.showwarning("警告", "尚未选择章节，无法保存。")
            return

        filepath = self.filepath_var.get().strip()
        if not filepath:
            messagebox.showwarning("警告", "请先配置保存文件路径")
            return

        chapter_file = os.path.join(filepath, "chapters", f"chapter_{chapter_number_str}.txt")
        content = self.chapter_view_text.get("0.0", "end").strip()

        clear_file_content(chapter_file)
        save_string_to_txt(content, chapter_file)
        self.safe_log(f"已保存对第 {chapter_number_str} 章的修改。")

    def prev_chapter(self):
        if not self.chapters_list:
            return
        current = self.chapter_select_var.get()
        if current not in self.chapters_list:
            return
        idx = self.chapters_list.index(current)
        if idx > 0:
            new_idx = idx - 1
            self.chapter_select_var.set(self.chapters_list[new_idx])
            self.load_chapter_content(self.chapters_list[new_idx])
        else:
            messagebox.showinfo("提示", "已经是第一章了。")

    def next_chapter(self):
        if not self.chapters_list:
            return
        current = self.chapter_select_var.get()
        if current not in self.chapters_list:
            return
        idx = self.chapters_list.index(current)
        if idx < len(self.chapters_list) - 1:
            new_idx = idx + 1
            self.chapter_select_var.set(self.chapters_list[new_idx])
            self.load_chapter_content(self.chapters_list[new_idx])
        else:
            messagebox.showinfo("提示", "已经是最后一章了。")

    # ------------------ 配置管理 ------------------ 
    def load_config_btn(self):
        cfg = load_config(self.config_file)
        if cfg:
            self.api_key_var.set(cfg.get("api_key", ""))
            self.base_url_var.set(cfg.get("base_url", ""))
            self.interface_format_var.set(cfg.get("interface_format", "OpenAI"))
            self.model_name_var.set(cfg.get("model_name", ""))
            self.embedding_url_var.set(cfg.get("embedding_url", ""))
            self.embedding_model_name_var.set(cfg.get("embedding_model_name", ""))
            self.temperature_var.set(cfg.get("temperature", 0.7))
            self.genre_var.set(cfg.get("genre", ""))
            self.num_chapters_var.set(cfg.get("num_chapters", 10))
            self.word_number_var.set(cfg.get("word_number", 3000))
            self.filepath_var.set(cfg.get("filepath", ""))

            # 主题
            self.topic_text.delete("0.0", "end")
            self.topic_text.insert("0.0", cfg.get("topic", ""))

            self.log("已加载配置。")
        else:
            messagebox.showwarning("提示", "未找到或无法读取配置文件。")

    def save_config_btn(self):
        config_data = {
            "api_key": self.api_key_var.get(),
            "base_url": self.base_url_var.get(),
            "interface_format": self.interface_format_var.get(),
            "model_name": self.model_name_var.get(),
            "embedding_url": self.embedding_url_var.get(),
            "embedding_model_name": self.embedding_model_name_var.get(),
            "temperature": self.temperature_var.get(),
            "topic": self.topic_text.get("0.0", "end").strip(),
            "genre": self.genre_var.get(),
            "num_chapters": self.num_chapters_var.get(),
            "word_number": self.word_number_var.get(),
            "filepath": self.filepath_var.get()
        }
        if save_config(config_data, self.config_file):
            messagebox.showinfo("提示", "配置已保存至 config.json")
            self.log("配置已保存。")
        else:
            messagebox.showerror("错误", "保存配置失败。")

    def browse_folder(self):
        selected_dir = filedialog.askdirectory()
        if selected_dir:
            self.filepath_var.set(selected_dir)

    # ------------------ 日志输出（主线程安全） ------------------ 
    def log(self, message: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def safe_log(self, message: str):
        self.master.after(0, lambda: self.log(message))

    def disable_button_safe(self, btn):
        self.master.after(0, lambda: btn.configure(state="disabled"))

    def enable_button_safe(self, btn):
        self.master.after(0, lambda: btn.configure(state="normal"))

    # ------------------ 分步操作：生成设定、目录、章节草稿、定稿 ------------------ 
    def generate_novel_setting_ui(self, thread_return: list[bool] | None = None) -> bool:
        """Step1. 生成小说设定（Novel_setting.txt）"""
        filepath = self.filepath_var.get().strip()
        if not filepath:
            # messagebox.showwarning("警告", "请先选择保存文件路径")
            logging.error("未选择文件保存路径")
            return False

        def task(over_state: list[bool] | None = None):
            self.disable_button_safe(self.btn_generate_setting)
            try:
                api_key = self.api_key_var.get().strip()
                base_url = self.base_url_var.get().strip()
                model_name = self.model_name_var.get().strip()
                topic = self.topic_text.get("0.0", "end").strip()
                genre = self.genre_var.get().strip()
                num_chapters = self.num_chapters_var.get()
                word_number = self.word_number_var.get()
                temperature = self.temperature_var.get()

                self.safe_log("开始生成小说设定...")
                Novel_setting_generate(
                    api_key=api_key,
                    base_url=base_url,
                    llm_model=model_name,
                    topic=topic,
                    genre=genre,
                    number_of_chapters=num_chapters,
                    word_number=word_number,
                    filepath=filepath,
                    temperature=temperature
                )
                self.safe_log("✅ 小说设定生成完成。请在 'Novel Settings' 标签页进行查看或编辑。")
                if over_state is not None:
                    over_state[0] = True
            except Exception:
                self.handle_exception("生成小说设定时出错")
                if over_state is not None:
                    over_state[0] = False
            finally:
                self.enable_button_safe(self.btn_generate_setting)

        threading.Thread(target=task, daemon=True, args=(thread_return,)).start()
        return True

    def generate_novel_directory_ui(self, thread_return: list[bool] | None = None) -> bool:
        """Step2. 基于已有 Novel_setting.txt 生成 Novel_directory.txt"""
        filepath = self.filepath_var.get().strip()
        if not filepath:
            # messagebox.showwarning("警告", "请先选择保存文件路径")
            logging.error("未选择文件保存路径")
            return False

        def task(over_state: list[bool] | None = None):
            self.disable_button_safe(self.btn_generate_directory)
            try:
                api_key = self.api_key_var.get().strip()
                base_url = self.base_url_var.get().strip()
                model_name = self.model_name_var.get().strip()
                num_chapters = self.num_chapters_var.get()
                temperature = self.temperature_var.get()

                self.safe_log("开始生成小说目录...")
                Novel_directory_generate(
                    api_key=api_key,
                    base_url=base_url,
                    llm_model=model_name,
                    number_of_chapters=num_chapters,
                    filepath=filepath,
                    temperature=temperature
                )
                self.safe_log("✅ 小说目录生成完成。请在 'Novel Directory' 标签页查看或编辑。")
                if over_state is not None:
                    over_state[0] = True
            except Exception:
                self.handle_exception("生成小说目录时出错")
                if over_state is not None:
                    over_state[0] = False
            finally:
                self.enable_button_safe(self.btn_generate_directory)

        threading.Thread(target=task, daemon=True, args=(thread_return,)).start()
        return True

    def generate_chapter_draft_ui(self, thread_return: list[bool] | None = None) -> bool:
        """Step3. 生成当前章节草稿"""
        filepath = self.filepath_var.get().strip()
        if not filepath:
            # messagebox.showwarning("警告", "请先选择保存文件路径")
            logging.error("未选择文件保存路径")
            return False

        def task(over_state: list[bool] | None = None):
            self.disable_button_safe(self.btn_generate_chapter)
            try:
                api_key = self.api_key_var.get().strip()
                base_url = self.base_url_var.get().strip()
                model_name = self.model_name_var.get().strip()
                temperature = self.temperature_var.get()
                novel_settings_file = os.path.join(filepath, "Novel_setting.txt")
                novel_settings = read_file(novel_settings_file)
                if not novel_settings.strip():
                    self.safe_log("⚠️ 未找到 Novel_setting.txt，请先生成设定。")
                    return

                character_state_file = os.path.join(filepath, "character_state.txt")
                character_state = read_file(character_state_file)
                global_summary_file = os.path.join(filepath, "global_summary.txt")
                global_summary = read_file(global_summary_file)
                novel_directory_file = os.path.join(filepath, "Novel_directory.txt")
                novel_directory = read_file(novel_directory_file)

                chap_num = self.chapter_num_var.get()
                word_number = self.word_number_var.get()
                user_guidance = self.user_guide_text.get("0.0", "end").strip()

                # 获取最近3章文本
                chapters_dir = os.path.join(filepath, "chapters")
                recent_3_texts = get_last_n_chapters_text(chapters_dir, chap_num, n=3)

                # 生成最近章节摘要
                recent_chapters_summary = summarize_recent_chapters(
                    llm_model=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=temperature,
                    chapters_text_list=recent_3_texts
                )

                self.safe_log(f"开始生成第{chap_num}章草稿...")
                draft_text = generate_chapter_draft(
                    novel_settings=novel_settings,
                    global_summary=global_summary,
                    character_state=character_state,
                    recent_chapters_summary=recent_chapters_summary,
                    user_guidance=user_guidance,
                    api_key=api_key,
                    base_url=base_url,
                    model_name=model_name,
                    novel_number=chap_num,
                    word_number=word_number,
                    temperature=temperature,
                    novel_novel_directory=novel_directory,
                    filepath=filepath,
                    interface_format=self.interface_format_var.get().strip(),
                    embedding_model_name=self.embedding_model_name_var.get().strip(),
                    embedding_base_url=self.embedding_url_var.get().strip()
                )
                if draft_text:
                    self.safe_log(f"✅ 第{chap_num}章草稿生成完成。请在左侧查看或编辑。")
                    self.master.after(0, lambda: self.show_chapter_in_textbox(draft_text))
                    if over_state is not None:
                        over_state[0] = True
                else:
                    self.safe_log("⚠️ 本章草稿生成失败或无内容。")
                    if over_state is not None:
                        over_state[0] = False

            except Exception:
                self.handle_exception("生成章节草稿时出错")
                if over_state is not None:
                    over_state[0] = False
            finally:
                self.enable_button_safe(self.btn_generate_chapter)

        threading.Thread(target=task, daemon=True, args=(thread_return,)).start()
        return True

    def show_chapter_in_textbox(self, text: str):
        self.chapter_result.delete("0.0", "end")
        self.chapter_result.insert("0.0", text)
        self.chapter_result.see("end")

    def finalize_chapter_ui(self, thread_return: list[bool] | None = None) -> bool:
        """Step4. 定稿当前章节：更新全局摘要、角色状态、向量库等"""
        filepath = self.filepath_var.get().strip()
        if not filepath:
            # messagebox.showwarning("警告", "请先选择保存文件路径")
            logging.error("未选择文件保存路径")
            return False

        def task(over_state: list[bool] | None = None):
            self.disable_button_safe(self.btn_finalize_chapter)
            try:
                api_key = self.api_key_var.get().strip()
                base_url = self.base_url_var.get().strip()
                model_name = self.model_name_var.get().strip()
                temperature = self.temperature_var.get()
                interface_format = self.interface_format_var.get().strip()
                embedding_model_name = self.embedding_model_name_var.get().strip()

                chap_num = self.chapter_num_var.get()
                word_number = self.word_number_var.get()

                self.safe_log(f"开始定稿第{chap_num}章...")
                finalize_chapter(
                    novel_number=chap_num,
                    word_number=word_number,
                    api_key=api_key,
                    base_url=base_url,
                    interface_format=interface_format,
                    embedding_model_name=embedding_model_name,
                    model_name=model_name,
                    temperature=temperature,
                    filepath=filepath
                )
                self.safe_log(f"✅ 第{chap_num}章定稿完成（已更新全局摘要、角色状态、剧情要点、向量库）。")

                # 读取定稿后的文本显示
                chap_file = os.path.join(filepath, "chapters", f"chapter_{chap_num}.txt")
                final_text = read_file(chap_file)
                self.master.after(0, lambda: self.show_chapter_in_textbox(final_text))
                if over_state is not None:
                    over_state[0] = True
            except Exception:
                self.handle_exception("定稿章节时出错")
                if over_state is not None:
                    over_state[0] = False
            finally:
                self.enable_button_safe(self.btn_finalize_chapter)

        threading.Thread(target=task, daemon=True, args=(thread_return,)).start()
        return True

    # ------------------ 一致性审校 ------------------ 
    def do_consistency_check(self):
        """使用审校Agent对最新章节进行简单一致性或冲突检查"""
        filepath = self.filepath_var.get().strip()
        if not filepath:
            messagebox.showwarning("警告", "请先配置保存文件路径。")
            return

        def task():
            self.disable_button_safe(self.btn_check_consistency)
            try:
                api_key = self.api_key_var.get().strip()
                base_url = self.base_url_var.get().strip()
                model_name = self.model_name_var.get().strip()
                temperature = self.temperature_var.get()

                novel_settings_file = os.path.join(filepath, "Novel_setting.txt")
                character_state_file = os.path.join(filepath, "character_state.txt")
                global_summary_file = os.path.join(filepath, "global_summary.txt")
                plot_arcs_file = os.path.join(filepath, "plot_arcs.txt")

                novel_setting = read_file(novel_settings_file)
                character_state = read_file(character_state_file)
                global_summary = read_file(global_summary_file)
                plot_arcs = read_file(plot_arcs_file)

                chap_num = self.chapter_num_var.get()
                chap_file = os.path.join(filepath, "chapters", f"chapter_{chap_num}.txt")
                chapter_text = read_file(chap_file)

                if not chapter_text.strip():
                    self.safe_log("⚠️ 当前章节文件为空或不存在，无法审校。")
                    return

                self.safe_log("开始一致性审校...")
                result = check_consistency(
                    novel_setting=novel_setting,
                    character_state=character_state,
                    global_summary=global_summary,
                    chapter_text=chapter_text,
                    api_key=api_key,
                    base_url=base_url,
                    model_name=model_name,
                    temperature=temperature,
                    plot_arcs=plot_arcs
                )
                self.safe_log("审校结果：")
                self.safe_log(result)

            except Exception:
                self.handle_exception("审校时出错")
            finally:
                self.enable_button_safe(self.btn_check_consistency)

        threading.Thread(target=task, daemon=True).start()

    # ------------------ 导入知识库/清空向量库/查看剧情要点 ------------------
    def import_knowledge_handler(self):
        selected_file = filedialog.askopenfilename(
            title="选择要导入的知识库文件",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if selected_file:
            def task():
                self.disable_button_safe(self.btn_import_knowledge)
                try:
                    self.safe_log(f"开始导入知识库文件: {selected_file}")
                    import_knowledge_file(
                        api_key=self.api_key_var.get().strip(),
                        base_url=self.base_url_var.get().strip(),
                        interface_format=self.interface_format_var.get().strip(),
                        embedding_model_name=self.embedding_model_name_var.get().strip(),
                        file_path=selected_file,
                        embedding_base_url=self.embedding_url_var.get().strip()
                    )
                    self.safe_log("✅ 知识库文件导入完成。")
                except Exception:
                    self.handle_exception("导入知识库时出错")
                finally:
                    self.enable_button_safe(self.btn_import_knowledge)

            threading.Thread(target=task, daemon=True).start()

    def on_key_gen_handler(self):
        def task():
            import time
            if not os.path.exists("last_chapter.txt"):
                with open("last_chapter.txt", 'w', encoding='utf-8') as f:
                    pass
            with open("last_chapter.txt", 'r', encoding='utf-8') as f:
                data = f.read()
            logging.info(f"读取到历史记录{data}")
            if len(data) > 0:
                start_chapter = int(data)
            else:
                start_chapter = 1
            if start_chapter == 1:
                while not self.do_until_success(self.generate_novel_setting_ui, []):
                    logging.info("设定失败")
                logging.info("设定成功")
                logging.info("每个环节之间休息30s...")
                time.sleep(30)
                while not self.do_until_success(self.generate_novel_directory_ui, []):
                    logging.info("目录失败")
                logging.info("目录成功")
                logging.info("每个环节之间休息30s...")
                time.sleep(30)
            for i in range(start_chapter, self.num_chapters_var.get() + 1):
                self.gen_chapter(i, start_chapter)
        threading.Thread(target=task, daemon=True).start()

    def gen_chapter(self, i, start_chapter) -> bool:
        self.chapter_num_var.set(i)
        logging.info(f"生成第{self.chapter_num_var.get()}")
        while not self.do_until_success(self.generate_chapter_draft_ui, []):
            logging.info("草稿失败")
            return False
        logging.info("草稿成功")
        while not self.do_until_success(self.finalize_chapter_ui, []):
            logging.info("定稿失败")
            return False
        logging.info("定稿成功")
        with open("last_chapter.txt", 'w', encoding='utf-8') as f:
            f.write(str(start_chapter + 1))
        logging.info("每章休息1分钟...")
        time.sleep(60)
        return True

    def do_until_success(self, func, arg):
        import time
        res = [0]
        if func(res):
            while res[0] == 0:
                time.sleep(5)
            return res[0]
        return False

    def clear_vectorstore_handler(self):
        first_confirm = messagebox.askyesno("警告", "确定要清空本地向量库吗？此操作不可恢复！")
        if first_confirm:
            second_confirm = messagebox.askyesno("二次确认", "你确定真的要删除所有向量数据吗？此操作不可恢复！")
            if second_confirm:
                clear_vector_store()
                self.log("已清空向量库。")

    def show_plot_arcs_ui(self):
        filepath = self.filepath_var.get().strip()
        if not filepath:
            messagebox.showwarning("警告", "请先在主Tab中设置保存文件路径")
            return

        plot_arcs_file = os.path.join(filepath, "plot_arcs.txt")
        if not os.path.exists(plot_arcs_file):
            messagebox.showinfo("剧情要点", "当前还未生成任何剧情要点或未解决冲突。")
            return

        arcs_text = read_file(plot_arcs_file).strip()
        if not arcs_text:
            arcs_text = "当前没有记录的剧情要点或冲突。"

        top = ctk.CTkToplevel(self.master)
        top.title("剧情要点/未解决冲突")
        top.geometry("600x400")

        text_area = ctk.CTkTextbox(top, wrap="word", font=("Microsoft YaHei", 12))
        text_area.pack(fill="both", expand=True, padx=10, pady=10)

        text_area.insert("0.0", arcs_text)
        text_area.configure(state="disabled")

    # ------------------ Novel Settings/Directory/Character/Global Summary 的加载与保存 ------------------
    def load_novel_setting(self):
        filepath = self.filepath_var.get().strip()
        if not filepath:
            messagebox.showwarning("警告", "请先在主Tab中设置保存文件路径")
            return
        setting_file = os.path.join(filepath, "Novel_setting.txt")
        content = read_file(setting_file)
        self.setting_text.delete("0.0", "end")
        self.setting_text.insert("0.0", content)
        self.log("已加载 Novel_setting.txt 内容到编辑区。")

    def save_novel_setting(self):
        filepath = self.filepath_var.get().strip()
        if not filepath:
            messagebox.showwarning("警告", "请先在主Tab中设置保存文件路径")
            return
        content = self.setting_text.get("0.0", "end").strip()
        setting_file = os.path.join(filepath, "Novel_setting.txt")
        clear_file_content(setting_file)
        save_string_to_txt(content, setting_file)
        self.log("已保存对 Novel_setting.txt 的修改。")

    def load_novel_directory(self):
        filepath = self.filepath_var.get().strip()
        if not filepath:
            messagebox.showwarning("警告", "请先在主Tab中设置保存文件路径")
            return
        directory_file = os.path.join(filepath, "Novel_directory.txt")
        content = read_file(directory_file)
        self.directory_text.delete("0.0", "end")
        self.directory_text.insert("0.0", content)
        self.log("已加载 Novel_directory.txt 内容到编辑区。")

    def save_novel_directory(self):
        filepath = self.filepath_var.get().strip()
        if not filepath:
            messagebox.showwarning("警告", "请先在主Tab中设置保存文件路径")
            return
        content = self.directory_text.get("0.0", "end").strip()
        directory_file = os.path.join(filepath, "Novel_directory.txt")
        clear_file_content(directory_file)
        save_string_to_txt(content, directory_file)
        self.log("已保存对 Novel_directory.txt 的修改。")

    def load_character_state(self):
        filepath = self.filepath_var.get().strip()
        if not filepath:
            messagebox.showwarning("警告", "请先在主Tab中设置保存文件路径")
            return
        char_file = os.path.join(filepath, "character_state.txt")
        content = read_file(char_file)
        self.character_text.delete("0.0", "end")
        self.character_text.insert("0.0", content)
        self.log("已加载 character_state.txt 内容到编辑区。")

    def save_character_state(self):
        filepath = self.filepath_var.get().strip()
        if not filepath:
            messagebox.showwarning("警告", "请先在主Tab中设置保存文件路径")
            return
        content = self.character_text.get("0.0", "end").strip()
        char_file = os.path.join(filepath, "character_state.txt")
        clear_file_content(char_file)
        save_string_to_txt(content, char_file)
        self.log("已保存对 character_state.txt 的修改。")

    def load_global_summary(self):
        filepath = self.filepath_var.get().strip()
        if not filepath:
            messagebox.showwarning("警告", "请先在主Tab中设置保存文件路径")
            return
        summary_file = os.path.join(filepath, "global_summary.txt")
        content = read_file(summary_file)
        self.summary_text.delete("0.0", "end")
        self.summary_text.insert("0.0", content)
        self.log("已加载 global_summary.txt 内容到编辑区。")

    def save_global_summary(self):
        filepath = self.filepath_var.get().strip()
        if not filepath:
            messagebox.showwarning("警告", "请先在主Tab中设置保存文件路径")
            return
        content = self.summary_text.get("0.0", "end").strip()
        summary_file = os.path.join(filepath, "global_summary.txt")
        clear_file_content(summary_file)
        save_string_to_txt(content, summary_file)
        self.log("已保存对 global_summary.txt 的修改。")


# 入口
if __name__ == "__main__":
    app = ctk.CTk()
    gui = NovelGeneratorGUI(app)
    app.mainloop()
