import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from analysis import DataAnalyzer
from tkinter import BooleanVar
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import traceback

class YieldAnalysisApp:
    def __init__(self):
        self.root = tk.Tk()
        self.files = []
        self.selected_columns = []
        self.dfs = []
        self.limits = {}  # 初始化limits属性
        self.skiprows = 0  # 初始化skiprows属性
        # 不要在这里初始化UI元素
        self.setup_ui()  # 将setup_ui放在成员变量初始化之后

    def validate_number_input(self, new_value):
        if new_value.strip() == '':
            return True
        try:
            float(new_value)
            return True
        except ValueError:
            return False

    def setup_ui(self):
        print("开始设置UI...")
        # 初始化自动检测变量
        self.auto_detect_var = tk.BooleanVar(value=True)
        
        self.root.title("良率分析工具")
        self.root.geometry("900x700")
        
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # 创建标题
        title_label = ttk.Label(main_frame, text="良率分析工具", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # 创建选项卡控件
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True, padx=5, pady=5)
        self.notebook = notebook  # 保存对notebook的引用
        
        # 数据导入选项卡
        import_tab = ttk.Frame(notebook)
        notebook.add(import_tab, text="数据导入")
        
        # 分析设置选项卡
        analysis_tab = ttk.Frame(notebook)
        notebook.add(analysis_tab, text="分析设置")
        
        # 结果输出选项卡
        result_tab = ttk.Frame(notebook)
        notebook.add(result_tab, text="结果输出")
        
        # ===== 数据导入选项卡内容 =====
        import_frame = ttk.LabelFrame(import_tab, text="CSV文件选择", padding=10)
        import_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 文件选择按钮
        file_button_frame = ttk.Frame(import_frame)
        file_button_frame.pack(fill='x', pady=5)
        
        ttk.Button(file_button_frame, text="选择CSV文件", command=self.select_files).pack(side='left', padx=5)
        ttk.Label(file_button_frame, text="支持多选，请选择格式一致的CSV文件").pack(side='left', padx=5)
        
        # 添加CSV格式设置
        csv_settings_frame = ttk.LabelFrame(import_frame, text="CSV格式设置", padding=5)
        csv_settings_frame.pack(fill='x', pady=5)
        
        # 自动检测选项
        auto_detect_frame = ttk.Frame(csv_settings_frame)
        auto_detect_frame.pack(fill='x', pady=2)
        
        ttk.Checkbutton(auto_detect_frame, text="自动检测标题行", variable=self.auto_detect_var,
                       command=self.toggle_auto_detect).pack(side='left', padx=5)
        
        # 特殊字符输入
        special_char_frame = ttk.Frame(csv_settings_frame)
        special_char_frame.pack(fill='x', pady=2)
        
        ttk.Label(special_char_frame, text="特殊字符:").pack(side='left', padx=5)
        # 重新创建特殊字符输入框
        self.special_char_entry = ttk.Entry(special_char_frame, width=10)
        self.special_char_entry.pack(side='left', padx=5)
        self.special_char_entry.insert(0, ',')  # 默认使用逗号作为特殊字符
        ttk.Label(special_char_frame, text="(用于自动检测标题行的特殊字符)").pack(side='left', padx=5)
        
        # 手动行号输入
        start_row_frame = ttk.Frame(csv_settings_frame)
        start_row_frame.pack(fill='x', pady=2)
        
        ttk.Label(start_row_frame, text="起始行号:").pack(side='left', padx=5)
        # 重新创建起始行号输入框
        self.start_row_entry = ttk.Entry(start_row_frame, width=10)
        self.start_row_entry.pack(side='left', padx=5)
        self.start_row_entry.insert(0, '1')  # 默认从第1行开始（0表示第一行）
        ttk.Label(start_row_frame, text="(当关闭自动检测时使用)").pack(side='left', padx=5)
        
        # 初始状态设置 - 将toggle_auto_detect调用移到特殊字符和起始行号输入框初始化后
        try:
            self.toggle_auto_detect()
        except Exception as e:
            print(f"初始化时toggle_auto_detect出错: {e}")
            traceback.print_exc()
        
        # 文件列表框
        file_list_frame = ttk.Frame(import_frame)
        file_list_frame.pack(fill='both', expand=True, pady=5)
        
        self.file_listbox = tk.Listbox(file_list_frame, width=80, height=10)
        self.file_listbox.pack(side='left', fill='both', expand=True)
        
        # 添加滚动条
        file_scrollbar = ttk.Scrollbar(file_list_frame, orient="vertical", command=self.file_listbox.yview)
        file_scrollbar.pack(side='right', fill='y')
        self.file_listbox.configure(yscrollcommand=file_scrollbar.set)
        
        # ===== 分析设置选项卡内容 =====
        # 参数设置区域
        param_frame = ttk.LabelFrame(analysis_tab, text="统计方法设置", padding=10)
        param_frame.pack(fill='x', padx=10, pady=10)
        
        method_frame = ttk.Frame(param_frame)
        method_frame.pack(fill='x', pady=5)
        
        ttk.Label(method_frame, text="统计方法:").pack(side='left', padx=5)
        self.method_combo = ttk.Combobox(method_frame, values=['3sigma', 'iqr'], width=15)
        self.method_combo.pack(side='left', padx=5)
        self.method_combo.current(0)
        self.method_combo.bind('<<ComboboxSelected>>', self._update_params_ui)
        
        # 参数容器框架
        self.params_frame = ttk.Frame(param_frame)
        self.params_frame.pack(fill='x', pady=5)
        
        # 初始化参数组件
        self._create_param_components()
        
        # 列选择区域
        columns_frame = ttk.LabelFrame(analysis_tab, text="特征项选择", padding=10)
        columns_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 添加批量推荐按钮
        button_frame = ttk.Frame(columns_frame)
        button_frame.pack(fill='x', pady=5)
        
        ttk.Button(button_frame, text="批量推荐", command=self.batch_recommend).pack(side='left', padx=5)
        ttk.Button(button_frame, text="智能推荐", command=self.smart_recommend).pack(side='left', padx=5)
        ttk.Label(button_frame, text="选择下列特征项并使用批量推荐或单击'推荐'按钮").pack(side='left', padx=5)
        
        # 树形视图和滚动条
        tree_frame = ttk.Frame(columns_frame)
        tree_frame.pack(fill='both', expand=True)
        
        # 创建树形视图
        self.column_tree = ttk.Treeview(tree_frame, columns=('select', 'name', 'lower', 'upper', 'recommend'), show='headings', selectmode='none')
        self.column_tree.column('select', width=60, anchor='w')
        self.column_tree.column('name', width=150, anchor='w')
        self.column_tree.column('lower', width=100, anchor='w')
        self.column_tree.column('upper', width=100, anchor='w')
        self.column_tree.column('recommend', width=80, anchor='w')
        self.column_tree.heading('select', text='选择')
        self.column_tree.heading('name', text='特征项')
        self.column_tree.heading('lower', text='下限')
        self.column_tree.heading('upper', text='上限')
        self.column_tree.heading('recommend', text='操作')
        
        # 添加垂直滚动条
        tree_vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.column_tree.yview)
        self.column_tree.configure(yscrollcommand=tree_vsb.set)
        
        # 添加水平滚动条
        tree_hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.column_tree.xview)
        self.column_tree.configure(xscrollcommand=tree_hsb.set)
        
        # 放置树形视图和滚动条
        self.column_tree.grid(row=0, column=0, sticky='nsew')
        tree_vsb.grid(row=0, column=1, sticky='ns')
        tree_hsb.grid(row=1, column=0, sticky='ew')
        
        # 配置树形视图框架的网格权重
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        
        # 添加复选框交互
        self.column_tree.tag_configure('checked', background='#E0F0FF')
        self.column_tree.tag_configure('unchecked', background='white')
        self.column_tree.tag_configure('recommended', background='#E8F5E9')  # 为已推荐项添加背景色
        self.column_tree.bind('<ButtonPress-1>', self._on_treeview_click)
        
        # ===== 结果输出选项卡内容 =====
        result_frame = ttk.LabelFrame(result_tab, text="分析结果", padding=10)
        result_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 操作按钮
        button_frame = ttk.Frame(result_frame)
        button_frame.pack(fill='x', pady=10)
        
        ttk.Button(button_frame, text="开始分析", command=self.analyze).pack(side='left', padx=10)
        ttk.Button(button_frame, text="导出报告", command=self.generate_report).pack(side='left', padx=10)
        
        # Add the new button
        output_button = ttk.Button(button_frame, text="产出分布输出", command=self.generate_distribution_output)
        output_button.pack(side='left', padx=10)
        
        # 结果显示区域
        self.result_text = tk.Text(result_frame, wrap='word', height=15)
        self.result_text.pack(fill='both', expand=True, pady=5)
        self.result_text.insert('1.0', "分析结果将显示在这里...\n\n请先在\"数据导入\"选项卡中选择CSV文件，然后在\"分析设置\"选项卡中设置参数并选择特征项。")
        self.result_text.config(state='disabled')
        
        # 状态栏
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill='x', side='bottom', pady=5)
        
        self.status_label = ttk.Label(status_frame, text="就绪", anchor='w')
        self.status_label.pack(side='left', padx=10)
        
        # 版本信息
        version_label = ttk.Label(status_frame, text="良率分析工具 v1.0", anchor='e')
        version_label.pack(side='right', padx=10)
        
    def _create_param_components(self):
        vcmd = (self.root.register(self.validate_number_input), '%P')
        
        # 3sigma参数
        self.sigma_lower_label = ttk.Label(self.params_frame, text="Sigma下限:")
        self.sigma_lower_entry = ttk.Entry(self.params_frame, width=8, validate="key", validatecommand=vcmd)
        self.sigma_upper_label = ttk.Label(self.params_frame, text="Sigma上限:")
        self.sigma_upper_entry = ttk.Entry(self.params_frame, width=8, validate="key", validatecommand=vcmd)
        
        self.sigma_lower_label.grid(row=0, column=0, padx=5, pady=2)
        self.sigma_lower_entry.grid(row=0, column=1, padx=5, pady=2)
        self.sigma_upper_label.grid(row=0, column=2, padx=5, pady=2)
        self.sigma_upper_entry.grid(row=0, column=3, padx=5, pady=2)
        
        # IQR参数
        self.iqr_lower_label = ttk.Label(self.params_frame, text="IQR下限乘数:")
        self.iqr_lower_entry = ttk.Entry(self.params_frame, width=8, validate="key", validatecommand=vcmd)
        self.iqr_upper_label = ttk.Label(self.params_frame, text="IQR上限乘数:")
        self.iqr_upper_entry = ttk.Entry(self.params_frame, width=8, validate="key", validatecommand=vcmd)
        
        self.iqr_lower_label.grid(row=1, column=0, padx=5, pady=2)
        self.iqr_lower_entry.grid(row=1, column=1, padx=5, pady=2)
        self.iqr_upper_label.grid(row=1, column=2, padx=5, pady=2)
        self.iqr_upper_entry.grid(row=1, column=3, padx=5, pady=2)
        
        # 马氏距离参数
        self.mahalanobis_label = ttk.Label(self.params_frame, text="自由度:")
        self.mahalanobis_entry = ttk.Entry(self.params_frame, width=8, validate="key", validatecommand=vcmd)
        
        self.mahalanobis_label.grid(row=2, column=0, padx=5, pady=2)
        self.mahalanobis_entry.grid(row=2, column=1, padx=5, pady=2)
        
        # Isolation Forest参数
        self.if_contamination_label = ttk.Label(self.params_frame, text="异常比例:")
        self.if_contamination_entry = ttk.Entry(self.params_frame, width=8, validate="key", validatecommand=vcmd)
        
        self.if_contamination_label.grid(row=3, column=0, padx=5, pady=2)
        self.if_contamination_entry.grid(row=3, column=1, padx=5, pady=2)
        
        # 百分位法参数
        self.percentile_lower_label = ttk.Label(self.params_frame, text="下限百分位:")
        self.percentile_lower_entry = ttk.Entry(self.params_frame, width=8, validate="key", validatecommand=vcmd)
        self.percentile_upper_label = ttk.Label(self.params_frame, text="上限百分位:")
        self.percentile_upper_entry = ttk.Entry(self.params_frame, width=8, validate="key", validatecommand=vcmd)
        
        self.percentile_lower_label.grid(row=4, column=0, padx=5, pady=2)
        self.percentile_lower_entry.grid(row=4, column=1, padx=5, pady=2)
        self.percentile_upper_label.grid(row=4, column=2, padx=5, pady=2)
        self.percentile_upper_entry.grid(row=4, column=3, padx=5, pady=2)
        
        # 手动设置上下限
        self.limit_frame = ttk.LabelFrame(self.params_frame, text="手动设置上下限")
        self.limit_frame.grid(row=5, column=0, columnspan=4, sticky='ew', pady=5)
        
        ttk.Label(self.limit_frame, text="下限:").grid(row=0, column=0, padx=5, pady=2)
        self.lower_entry = ttk.Entry(self.limit_frame, width=10)
        self.lower_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(self.limit_frame, text="上限:").grid(row=0, column=2, padx=5, pady=2)
        self.upper_entry = ttk.Entry(self.limit_frame, width=10)
        self.upper_entry.grid(row=0, column=3, padx=5, pady=2)
        
        # 初始隐藏所有组件
        self.sigma_lower_label.grid_remove()
        self.sigma_lower_entry.grid_remove()
        self.sigma_upper_label.grid_remove()
        self.sigma_upper_entry.grid_remove()
        self.iqr_lower_label.grid_remove()
        self.iqr_lower_entry.grid_remove()
        self.iqr_upper_label.grid_remove()
        self.iqr_upper_entry.grid_remove()
        self.mahalanobis_label.grid_remove()
        self.mahalanobis_entry.grid_remove()
        self.if_contamination_label.grid_remove()
        self.if_contamination_entry.grid_remove()
        self.percentile_lower_label.grid_remove()
        self.percentile_lower_entry.grid_remove()
        self.percentile_upper_label.grid_remove()
        self.percentile_upper_entry.grid_remove()
    
    def _update_params_ui(self, event):
        # 隐藏所有参数组件
        self.sigma_lower_label.grid_remove()
        self.sigma_lower_entry.grid_remove()
        self.sigma_upper_label.grid_remove()
        self.sigma_upper_entry.grid_remove()
        self.iqr_lower_label.grid_remove()
        self.iqr_lower_entry.grid_remove()
        self.iqr_upper_label.grid_remove()
        self.iqr_upper_entry.grid_remove()
        self.mahalanobis_label.grid_remove()
        self.mahalanobis_entry.grid_remove()
        self.if_contamination_label.grid_remove()
        self.if_contamination_entry.grid_remove()
        self.percentile_lower_label.grid_remove()
        self.percentile_lower_entry.grid_remove()
        self.percentile_upper_label.grid_remove()
        self.percentile_upper_entry.grid_remove()
        
        method = self.method_combo.get()
        if method == '3sigma':
            self.sigma_lower_label.grid()
            self.sigma_lower_entry.grid()
            self.sigma_upper_label.grid()
            self.sigma_upper_entry.grid()
            self.sigma_lower_entry.delete(0, tk.END)
            self.sigma_lower_entry.insert(0, '1.5')
            self.sigma_upper_entry.delete(0, tk.END)
            self.sigma_upper_entry.insert(0, '3.0')
        elif method == 'iqr':
            self.iqr_lower_label.grid()
            self.iqr_lower_entry.grid()
            self.iqr_upper_label.grid()
            self.iqr_upper_entry.grid()
            self.iqr_lower_entry.delete(0, tk.END)
            self.iqr_lower_entry.insert(0, '1.5')
            self.iqr_upper_entry.delete(0, tk.END)
            self.iqr_upper_entry.insert(0, '1.5')
        elif method == 'mahalanobis':
            self.mahalanobis_label.grid()
            self.mahalanobis_entry.grid()
            self.mahalanobis_entry.delete(0, tk.END)
            self.mahalanobis_entry.insert(0, '1')
        elif method == 'isolation_forest':
            self.if_contamination_label.grid()
            self.if_contamination_entry.grid()
            self.if_contamination_entry.delete(0, tk.END)
            self.if_contamination_entry.insert(0, '0.1')
        elif method == 'percentile':
            self.percentile_lower_label.grid()
            self.percentile_lower_entry.grid()
            self.percentile_upper_label.grid()
            self.percentile_upper_entry.grid()
            self.percentile_lower_entry.delete(0, tk.END)
            self.percentile_lower_entry.insert(0, '1')
            self.percentile_upper_entry.delete(0, tk.END)
            self.percentile_upper_entry.insert(0, '99')
    
    def batch_recommend(self):
        method = self.method_combo.get()
        if not method:
            tk.messagebox.showerror("错误", "请先选择统计方法")
            return
        
        # 获取当前选中的列
        selected_columns = []
        for item in self.column_tree.get_children():
            if self.column_tree.item(item, 'values')[0] == 'True':
                selected_columns.append(self.column_tree.item(item, 'values')[1])
        
        if not selected_columns:
            tk.messagebox.showerror("错误", "请至少选择一个特征项")
            return
        
        # 获取当前方法的参数
        current_params = {}
        if method == '3sigma':
            try:
                lower = float(self.sigma_lower_entry.get() or '1.5')
                upper = float(self.sigma_upper_entry.get() or '3.0')
                current_params = {'lower_param': lower, 'upper_param': upper}
            except ValueError:
                tk.messagebox.showerror("错误", "请输入有效的数字参数")
                return
        elif method == 'iqr':
            try:
                lower = float(self.iqr_lower_entry.get() or '1.5')
                upper = float(self.iqr_upper_entry.get() or '1.5')
                current_params = {'lower_multiplier': lower, 'upper_multiplier': upper}
            except ValueError:
                tk.messagebox.showerror("错误", "请输入有效的数字参数")
                return
        else:
            tk.messagebox.showerror("错误", "不支持的统计方法")
            return
        
        # 清除所有项的推荐标记
        for item in self.column_tree.get_children():
            column = self.column_tree.item(item, 'values')[1]
            if column not in selected_columns:
                # 如果当前项不在选中列表中，但之前有推荐值，则清除推荐值
                current_values = list(self.column_tree.item(item, 'values'))
                if current_values[4] == '已推荐':
                    current_values[2] = ''  # 清除下限
                    current_values[3] = ''  # 清除上限
                    current_values[4] = '推荐'  # 重置为"推荐"状态
                    self.column_tree.item(item, values=tuple(current_values))
                    self.column_tree.item(item, tags=('unchecked',))  # 重置背景色
                    
                    # 从limits中移除
                    if column in self.limits:
                        del self.limits[column]
        
        # 为选中的列推荐限制值
        for column in selected_columns:
            self._on_recommend_click(column, method=method, **current_params)
    
    def analyze(self):
        if not self.files:
            tk.messagebox.showerror("错误", "请先选择文件")
            return
        
        # 检查是否有选中的特征项
        if not self.selected_columns:
            tk.messagebox.showerror("错误", "请至少选择一个特征项")
            return
        
        # 直接使用用户设置的limits参数
        for item in self.column_tree.get_children():
            values = self.column_tree.item(item, 'values')
            if values[0] == 'True':
                try:
                    self.limits[values[1]] = (
                        float(values[2]) if values[2] else None,
                        float(values[3]) if values[3] else None
                    )
                except ValueError:
                    continue
        
        # 显示结果
        self._show_analysis_results()
        
        # 切换到结果选项卡
        for i, tab_name in enumerate(self.notebook.tabs()):
            if self.notebook.tab(tab_name, "text") == "结果输出":
                self.notebook.select(i)
                break
        
        # 更新状态栏
        self.status_label.config(text="分析完成")

    def generate_report(self):
        if not self.files:
            messagebox.showerror("错误", "请先选择文件")
            return
            
        if not self.selected_columns:
            messagebox.showerror("错误", "请先选择至少一个特征项")
            return
        
        # 创建进度条窗口
        progress_window = tk.Toplevel(self.root)
        progress_window.title("生成报告进度")
        progress_window.geometry("300x100")
        
        # 创建进度条
        self.progress = ttk.Progressbar(progress_window, orient="horizontal", length=250, mode="determinate")
        self.progress.pack(pady=20)
        
        # 创建进度标签
        self.progress_label = ttk.Label(progress_window, text="0%")
        self.progress_label.pack()
        
        # 更新进度条
        def update_progress(progress):
            self.progress['value'] = progress
            self.progress_label.config(text=f"{progress}%")
            progress_window.update_idletasks()
        
        output_path = filedialog.asksaveasfilename(defaultextension='.xlsx')
        if not output_path:
            progress_window.destroy()
            return
        
        # 确保analyzer已初始化
        if not hasattr(self, 'analyzer') or self.analyzer is None:
            self.analyzer = DataAnalyzer(self.files, analyzer=self)
        
        # 创建PDF文件用于保存直方图
        pdf_path = output_path.replace('.xlsx', '_distribution.pdf')
        
        try:
            with PdfPages(pdf_path) as pdf:
                # 更新进度条
                update_progress(10)
                
                # 生成汇总分布图
                combined_data = pd.concat(self.dfs)
                total_columns = len(self.selected_columns)
                
                for i, col in enumerate(self.selected_columns):
                    update_progress(10 + int(80 * i / total_columns))  # 更新进度，10-90%
                    
                    if col in combined_data.columns:
                        data = combined_data[col].dropna()
                        lower, upper = self.limits.get(col, (None, None))
                        
                        # 创建图表
                        plt.figure(figsize=(10, 6))
                        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
                        plt.rcParams['axes.unicode_minus'] = False
                        
                        if not data.empty:
                            # 处理离群值并计算更合适的直方图区间
                            try:
                                # 使用四分位范围(IQR)检测离群值
                                q1 = np.percentile(data, 25)
                                q3 = np.percentile(data, 75)
                                iqr = q3 - q1
                                
                                # 定义离群值边界 (一般用1.5倍IQR)
                                lower_bound = q1 - 1.5 * iqr
                                upper_bound = q3 + 1.5 * iqr
                                
                                # 找出主要数据分布范围（排除极端离群值）
                                main_data = data[(data >= lower_bound) & (data <= upper_bound)]
                                
                                # 如果过滤后的数据太少，则还是使用原始数据
                                if len(main_data) < len(data) * 0.9:  # 如果过滤掉超过10%的数据，考虑回退
                                    # 尝试使用更宽松的边界
                                    lower_bound = q1 - 3 * iqr
                                    upper_bound = q3 + 3 * iqr
                                    main_data = data[(data >= lower_bound) & (data <= upper_bound)]
                                    
                                    # 如果还是过滤掉太多数据，则使用5%和95%分位数作为边界
                                    if len(main_data) < len(data) * 0.8:
                                        lower_bound = np.percentile(data, 1)
                                        upper_bound = np.percentile(data, 99)
                                        main_data = data[(data >= lower_bound) & (data <= upper_bound)]
                                        
                                # 确保边界包含上下限值（如果有设置）
                                if lower is not None:
                                    lower_bound = min(lower_bound, lower)
                                if upper is not None:
                                    upper_bound = max(upper_bound, upper)
                                    
                                # 计算直方图bin数和范围
                                min_val = lower_bound
                                max_val = upper_bound
                                data_range = max_val - min_val
                                
                                # 如果数据范围很小，调整间距
                                if data_range < 1e-10:
                                    mean_val = np.mean(data)
                                    min_val = mean_val - 1
                                    max_val = mean_val + 1
                                    
                                # 根据数据量自动调整bin数量
                                if len(data) < 100:
                                    n_bins = 10
                                elif len(data) < 1000:
                                    n_bins = 20
                                else:
                                    n_bins = 30
                                    
                                # 生成区间
                                bins = np.linspace(min_val, max_val, n_bins + 1)
                                
                                # 记录离群值的信息
                                outliers = data[(data < lower_bound) | (data > upper_bound)]
                                outlier_info = f"离群值: {len(outliers)}个 ({len(outliers)/len(data)*100:.2f}%)" if len(outliers) > 0 else ""
                            except:
                                # 如果出错就使用默认设置
                                bins = 20
                                outlier_info = ""
                            
                            # 创建左右分栏布局
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})
                            plt.suptitle(f'{col} 分布', fontsize=14)
                            
                            # 在左侧绘制直方图
                            n, bins, patches = ax1.hist(
                                data,
                                bins=bins,
                                alpha=0.7,
                                edgecolor='black',
                                label='数据分布'
                            )
                        
                            # 如果有上下限，标记范围内和范围外的数据
                            if lower is not None and upper is not None:
                                # 获取每个bin的中心点
                                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                                
                                # 为范围外的bin设置不同的颜色
                                for i, patch in enumerate(patches):
                                    if bin_centers[i] < lower or bin_centers[i] > upper:
                                        patch.set_facecolor('lightcoral')  # 范围外的设为红色
                                    else:
                                        patch.set_facecolor('lightgreen')  # 范围内的设为绿色
                                
                                # 添加图例
                                ax1.legend([
                                    plt.Rectangle((0,0),1,1, facecolor='lightgreen', edgecolor='black'),
                                    plt.Rectangle((0,0),1,1, facecolor='lightcoral', edgecolor='black')
                                ], ['范围内数据', '范围外数据'])
                            
                            # 添加上下限线
                            if lower is not None:
                                ax1.axvline(x=lower, color='red', linestyle='--', label='下限')
                            if upper is not None:
                                ax1.axvline(x=upper, color='green', linestyle='--', label='上限')
                            
                            ax1.set_xlabel('测量值')
                            ax1.set_ylabel('频数')
                            ax1.grid(True, linestyle='--', alpha=0.7)
                            
                            # 计算正态性检验结果
                            _, p_value = stats.normaltest(data)
                            is_normal = p_value > 0.05
                            normality_text = "符合正态分布" if is_normal else "不符合正态分布"
                            
                            # 如果不符合正态分布，尝试识别可能的分布类型
                            distribution_type = ""
                            if not is_normal:
                                # 计算偏度和峰度
                                skewness = stats.skew(data)
                                kurtosis = stats.kurtosis(data)
                                
                                # 尝试拟合几种常见分布
                                try:
                                    # 对数正态分布检验
                                    if np.all(data > 0):  # 对数正态分布要求数据全部为正
                                        _, lognorm_p = stats.normaltest(np.log(data))
                                        if lognorm_p > 0.05:
                                            distribution_type = "可能符合对数正态分布"
                                    
                                    # 根据偏度和峰度判断
                                    if not distribution_type:
                                        if abs(skewness) < 0.5:
                                            if kurtosis > 0.5:
                                                distribution_type = "可能符合t分布"
                                            elif kurtosis < -0.5:
                                                distribution_type = "可能符合均匀分布"
                                            else:
                                                distribution_type = "接近正态但有偏差"
                                        elif skewness > 0.5:
                                            distribution_type = "右偏分布(可能为指数族分布)"
                                        elif skewness < -0.5:
                                            distribution_type = "左偏分布"
                                except:
                                    distribution_type = "无法确定分布类型"
                            
                            # 在右侧显示统计信息
                            ax2.axis('off')  # 关闭坐标轴
                            
                            # 统计信息
                            info_text = f"总样本数: {len(data)}\n"
                            info_text += f"平均值: {np.mean(data):.4f}\n"
                            info_text += f"标准差: {np.std(data):.4f}\n"
                            info_text += f"中位数: {np.median(data):.4f}\n"
                            info_text += f"最小值: {np.min(data):.4f}\n"
                            info_text += f"最大值: {np.max(data):.4f}\n\n"
                            info_text += f"正态性检验: {normality_text}\n(p={p_value:.4f})\n"
                            
                            if not is_normal and distribution_type:
                                info_text += f"分布类型: {distribution_type}\n"
                                
                            if outlier_info:
                                info_text += f"\n{outlier_info}\n"
                            if lower is not None and upper is not None:
                                valid_count = len(data[(data >= lower) & (data <= upper)])
                                below_count = len(data[data < lower])
                                above_count = len(data[data > upper])
                                
                                info_text += f"\n范围内样本数: {valid_count}\n({valid_count/len(data)*100:.2f}%)\n"
                                info_text += f"低于下限样本数: {below_count}\n({below_count/len(data)*100:.2f}%)\n"
                                info_text += f"高于上限样本数: {above_count}\n({above_count/len(data)*100:.2f}%)\n"
                                
                                # 添加上下限值
                                info_text += f"\n下限值: {lower:.4f}\n"
                                info_text += f"上限值: {upper:.4f}\n"
                                
                                # 尝试获取推荐方法信息
                                for item in self.column_tree.get_children():
                                    if self.column_tree.item(item, 'values')[1] == col:
                                        method_info = self.column_tree.item(item, 'values')[4]
                                        if method_info and '智能推荐' in method_info:
                                            info_text += f"\n推荐方法: {method_info}\n"
                                            # 添加方法说明
                                            if '3sigma' in method_info:
                                                info_text += "基于均值和标准差计算"
                                            elif 'iqr' in method_info:
                                                info_text += "基于四分位数范围计算"
                                            elif 'lognormal' in method_info:
                                                info_text += "在对数空间中计算后转换"
                                            elif 'percentile' in method_info:
                                                info_text += "基于百分位数计算"
                                            elif 'range' in method_info:
                                                info_text += "基于数据范围计算"
                                        break
                            
                            # 显示统计信息
                            ax2.text(0, 0.95, info_text, verticalalignment='top', fontsize=9)
                            
                            # 调整子图之间的间距
                            plt.tight_layout()
                            plt.subplots_adjust(top=0.9)
                                
                        else:
                            # 创建简单的图表显示没有数据的信息
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.text(0.5, 0.5, '没有有效数据', horizontalalignment='center', 
                                   verticalalignment='center', transform=ax.transAxes, fontsize=14)
                            ax.set_title(f'{col} 分布')
                            ax.axis('off')
                            print(f'没有有效数据用于生成{col}的直方图')
                        
                        try:
                            pdf.savefig()
                        except Exception as e:
                            print(f'保存图表失败: {e}')
                        finally:
                            plt.close()
                
                # 生成Excel报告
                update_progress(90)
                
                try:
                    # 使用DataAnalyzer中的方法生成完整报告
                    self.analyzer.generate_report(
                        self.selected_columns, 
                        output_path, 
                        self.limits
                    )
                    update_progress(100)
                    messagebox.showinfo("成功", f"报告已成功生成:\n{output_path}\n{pdf_path}")
                except Exception as e:
                    messagebox.showerror("错误", f"生成报告失败: {str(e)}")
                
        except Exception as e:
            messagebox.showerror("错误", f"生成报告时出错: {str(e)}")
            
        finally:
            progress_window.destroy()

    def select_files(self):
        try:
            self.files = filedialog.askopenfilenames(filetypes=[("CSV文件", "*.csv")])
            
            # 如果有选择文件
            if self.files:
                try:
                    # 使用设置中的行号或自动检测
                    if not self.auto_detect_var.get():
                        try:
                            start_row_text = self.start_row_entry.get()
                            print(f"获取到起始行文本: '{start_row_text}'")
                            if not start_row_text.strip():
                                self.skiprows = 0
                                messagebox.showwarning("警告", "起始行号为空，将使用默认值0")
                            else:
                                self.skiprows = int(start_row_text) - 1
                                if self.skiprows < 0:
                                    self.skiprows = 0
                        except (ValueError, AttributeError) as e:
                            self.skiprows = 0
                            messagebox.showwarning("警告", f"无效的起始行号: {e}，将使用默认值0")
                            print(f"获取起始行号出错: {e}")
                    else:
                        # 自动检测模式，先使用默认值
                        special_char = self.special_char_entry.get()
                        print(f"自动检测模式，特殊字符: '{special_char}'")
                        self.skiprows = 0
                    
                    print(f"最终使用的skiprows值: {self.skiprows}")
                    
                    # 更新文件列表显示
                    self.file_listbox.delete(0, tk.END)
                    for f in self.files:
                        self.file_listbox.insert(tk.END, f)
                    
                    # 加载列选择
                    self._load_columns(self.files[0])
                    
                except Exception as e:
                    error_msg = f"加载文件出错: {str(e)}"
                    print(error_msg)
                    traceback.print_exc()
                    tk.messagebox.showerror("错误", error_msg)
        except Exception as e:
            error_msg = f"选择文件时出错: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            tk.messagebox.showerror("错误", error_msg)

    def _load_columns(self, file_path):
        try:
            # 清空现有列
            for i in self.column_tree.get_children():
                self.column_tree.delete(i)
            
            # 重置选中列集合
            self.selected_columns = []
            
            # 创建analyzer对象并加载文件
            self.analyzer = DataAnalyzer(self.files, skiprows=self.skiprows, analyzer=self)
            
            # 如果analyzer没有成功创建，可能是因为文件读取错误
            if not hasattr(self, 'analyzer') or self.analyzer is None:
                messagebox.showerror("错误", "无法创建分析器。请检查文件格式和起始行设置。")
                return
            
            # 检查是否成功加载了数据
            if len(self.analyzer.dfs) == 0 or self.analyzer.dfs[0].empty:
                messagebox.showerror("错误", "没有成功加载任何数据。请检查文件格式和起始行设置。")
                return
            
            # 获取第一个文件的列
            df = self.analyzer.dfs[0]
            self.dfs = self.analyzer.dfs
            
            # 将列添加到树形视图
            for col in df.columns:
                self.column_tree.insert('', 'end', values=('False', col, '', '', '推荐'))
            
            # 显示成功加载的列数
            messagebox.showinfo("成功", f"成功加载 {len(df.columns)} 列数据。")
            
        except Exception as e:
            print(f"加载列出错: {e}")
            traceback.print_exc()
            messagebox.showerror("错误", f"加载列时出错: {str(e)}\n请检查文件格式和起始行设置。")

    def _on_treeview_click(self, event):
        item = self.column_tree.identify_row(event.y)
        col = self.column_tree.identify_column(event.x)
        
        if col == '#1':
            self._on_checkbox_click(event)
        elif col in ('#3', '#4'):  # 处理上下限编辑
            self._on_edit_limit(item, col)
        elif col == '#5' and self.column_tree.item(item, 'values')[4] == '推荐':
            method = self.method_combo.get()
            column = self.column_tree.item(item, 'values')[1]
            
            # 获取当前方法的参数
            current_params = {}
            if method == '3sigma':
                try:
                    lower = float(self.sigma_lower_entry.get() or '1.5')
                    upper = float(self.sigma_upper_entry.get() or '3.0')
                    current_params = {'lower_param': lower, 'upper_param': upper}
                except ValueError:
                    return
            elif method == 'iqr':
                try:
                    lower = float(self.iqr_lower_entry.get() or '1.5')
                    upper = float(self.iqr_upper_entry.get() or '1.5')
                    current_params = {'lower_multiplier': lower, 'upper_multiplier': upper}
                except ValueError:
                    return
            elif method == 'mahalanobis':
                try:
                    current_params = {'n_components': int(self.mahalanobis_entry.get() or '1')}
                except ValueError:
                    return
            elif method == 'Isolation Forest':
                try:
                    current_params = {'contamination': float(self.if_contamination_entry.get() or '0.1')}
                except ValueError:
                    return
                    
            self._on_recommend_click(column, method=method, **current_params)

    def _on_checkbox_click(self, event):
        item = self.column_tree.identify_row(event.y)
        if item and self.column_tree.identify_column(event.x) == '#1':
            current_val = self.column_tree.item(item, 'values')[0]
            new_val = 'True' if current_val == 'False' else 'False'
            current_values = list(self.column_tree.item(item, 'values'))
            current_values[0] = new_val
            self.column_tree.item(item, values=tuple(current_values))
            
            # 更新背景色
            self.column_tree.item(item, tags=('checked' if new_val == 'True' else 'unchecked',))
            self.column_tree.tag_configure('checked', background='#E0F0FF')
            
            # 更新选中列
            col_name = self.column_tree.item(item, 'values')[1]
            if new_val == 'True':
                if col_name not in self.selected_columns:
                    self.selected_columns.append(col_name)
            else:
                if col_name in self.selected_columns:
                    self.selected_columns.remove(col_name)

    def _on_edit_limit(self, item, col):
        column = self.column_tree.item(item, 'values')[1]
        entry_window = tk.Toplevel(self.root)
        entry_window.title("编辑限制值")
        
        ttk.Label(entry_window, text=f"编辑 {column} 限制值").pack(padx=10, pady=5)
        
        # 创建输入框
        entry_frame = ttk.Frame(entry_window)
        entry_frame.pack(padx=10, pady=5)
        
        ttk.Label(entry_frame, text="下限:").grid(row=0, column=0)
        lower_entry = ttk.Entry(entry_frame)
        lower_entry.insert(0, self.column_tree.item(item, 'values')[2])
        lower_entry.grid(row=0, column=1)
        
        ttk.Label(entry_frame, text="上限:").grid(row=1, column=0)
        upper_entry = ttk.Entry(entry_frame)
        upper_entry.insert(0, self.column_tree.item(item, 'values')[3])
        upper_entry.grid(row=1, column=1)
        
        def save_values():
            values = list(self.column_tree.item(item, 'values'))
            values[2] = lower_entry.get()
            values[3] = upper_entry.get()
            self.column_tree.item(item, values=tuple(values))
            # 同步到limits参数
            try:
                self.limits[values[1]] = (
                    float(values[2]) if values[2] else None,
                    float(values[3]) if values[3] else None
                )
            except ValueError:
                pass
            entry_window.destroy()
        
        ttk.Button(entry_window, text="保存", command=save_values).pack(pady=10)

    def _on_recommend_click(self, column, method, **params):
        if not method:
            tk.messagebox.showerror("错误", "请先选择统计方法")
            return
        
        # 确保analyzer已初始化
        if not hasattr(self, 'analyzer') or self.analyzer is None:
            if self.files:
                self.analyzer = DataAnalyzer(self.files, analyzer=self)
            else:
                tk.messagebox.showerror("错误", "请先选择文件")
                return
                
        try:
            lower, upper = self.analyzer.calculate_limits(column, method, **params)
            for item in self.column_tree.get_children():
                if self.column_tree.item(item, 'values')[1] == column:
                    current_values = list(self.column_tree.item(item, 'values'))
                    if lower is not None and upper is not None:
                        current_values[2] = f'{lower:.4f}'
                        current_values[3] = f'{upper:.4f}'
                        current_values[4] = '已推荐'
                        # 更新列背景色
                        self.column_tree.item(item, tags=('recommended',))
                        self.column_tree.tag_configure('recommended', background='#E8F5E9')
                        # 同步到输入框
                        self.lower_entry.delete(0, tk.END)
                        self.lower_entry.insert(0, f'{lower:.4f}')
                        self.upper_entry.delete(0, tk.END)
                        self.upper_entry.insert(0, f'{upper:.4f}')
                        # 更新limits参数
                        self.limits[column] = (lower, upper)
                    else:
                        tk.messagebox.showwarning("警告", f"无法计算{column}的限制值，可能是数据不足或分布不适合当前方法")
                    self.column_tree.item(item, values=tuple(current_values))
                    break
        except Exception as e:
            tk.messagebox.showerror("错误", f"计算限制值时出错: {str(e)}")

    def _show_analysis_results(self):
        """在结果文本框中显示分析结果"""
        # 清空并启用文本框
        self.result_text.config(state='normal')
        self.result_text.delete('1.0', tk.END)
        
        # 添加标题
        self.result_text.insert(tk.END, "===== 分析结果 =====\n\n", 'title')
        self.result_text.tag_configure('title', font=('Arial', 12, 'bold'))
        
        # 添加结果表格
        result_data = []
        for col, (lower, upper) in self.limits.items():
            data = pd.concat([df[col] for df in self.dfs if col in df.columns])
            valid = data[(data >= lower) & (data <= upper)] if lower is not None and upper is not None else data
            
            # 计算超限颗粒数
            below_lower = len(data[data < lower]) if lower is not None else 0
            above_upper = len(data[data > upper]) if upper is not None else 0
            
            # 添加到结果列表
            result_data.append({
                '特征项': col,
                '下限': f'{lower:.4f}' if lower is not None else 'N/A',
                '上限': f'{upper:.4f}' if lower is not None else 'N/A',
                '总颗粒数': len(data),
                '有效颗粒数': len(valid),
                '超下限颗粒数': below_lower,
                '超上限颗粒数': above_upper,
                '良率': f'{len(valid)/len(data)*100:.2f}%' if len(data) > 0 else 'N/A'
            })
        
        # 按良率排序
        result_data.sort(key=lambda x: float(x['良率'].replace('%', '')) if x['良率'] != 'N/A' else 0)
        
        # 显示结果
        for item in result_data:
            self.result_text.insert(tk.END, f"特征项: {item['特征项']}\n", 'feature')
            self.result_text.tag_configure('feature', font=('Arial', 10, 'bold'))
            self.result_text.insert(tk.END, f"下限: {item['下限']}, 上限: {item['上限']}\n")
            self.result_text.insert(tk.END, f"总颗粒数: {item['总颗粒数']}, 有效颗粒数: {item['有效颗粒数']}\n")
            self.result_text.insert(tk.END, f"超下限颗粒数: {item['超下限颗粒数']}, 超上限颗粒数: {item['超上限颗粒数']}\n")
            self.result_text.insert(tk.END, f"良率: {item['良率']}\n\n")
        
        # 移除二次推荐按钮
        self.result_text.insert(tk.END, "\n\n")
        
        # 禁用文本框
        self.result_text.config(state='disabled')

    def mahalanobis_outlier_removal(self, data, threshold=0.99):
        # 计算均值向量
        mean_vec = np.mean(data, axis=0)
        # 计算协方差矩阵
        cov_mat = np.cov(data, rowvar=False)
        # 添加正则化确保矩阵可逆
        cov_mat += 1e-6 * np.eye(cov_mat.shape[0])
        # 计算逆协方差矩阵
        inv_cov = np.linalg.inv(cov_mat)
        # 计算每个点的马氏距离
        distances = []
        for i in range(len(data)):
            diff = data[i] - mean_vec
            dist = np.sqrt(diff.dot(inv_cov).dot(diff.T))
            distances.append(dist)
        # 使用卡方分布确定阈值
        from scipy.stats import chi2
        cutoff = np.sqrt(chi2.ppf(threshold, data.shape[1]))
        # 返回非离群点的索引
        return np.where(np.array(distances) <= cutoff)[0]

    def isolation_forest_removal(self, data, contamination=0.05):
        # 创建并训练隔离森林模型
        clf = IsolationForest(
            contamination=contamination,  # 预期的异常点比例
            random_state=42,
            n_estimators=100
        )
        # 拟合数据
        clf.fit(data)
        # 预测异常(-1)和正常(1)
        predictions = clf.predict(data)
        # 返回正常点的索引
        return np.where(predictions == 1)[0]

    def multi_dimensional_analysis(self):
        """多维分析功能，对多个特征项进行共同分析"""
        
        # 获取选中的特征项
        selected_features = []
        for item in self.column_tree.get_children():
            if self.column_tree.item(item, 'values')[0] == 'True':
                selected_features.append(self.column_tree.item(item, 'values')[1])
        
        if len(selected_features) < 2:
            tk.messagebox.showerror("错误", "多维分析需要至少选择2个特征项")
            return
            
        # 创建多维分析窗口
        multi_win = tk.Toplevel(self.root)
        multi_win.title("多维分析设置")
        multi_win.geometry("400x300")
        
        # 方法选择
        method_frame = ttk.Frame(multi_win, padding=10)
        method_frame.pack(fill='x')
        
        ttk.Label(method_frame, text="选择多维分析方法:").grid(row=0, column=0, sticky='w')
        method_var = tk.StringVar(value="Isolation Forest")
        method_combo = ttk.Combobox(method_frame, textvariable=method_var, 
                                   values=["Isolation Forest", "马氏距离", "DBSCAN"], 
                                   state="readonly", width=15)
        method_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # 参数设置
        param_frame = ttk.LabelFrame(multi_win, text="参数设置", padding=10)
        param_frame.pack(fill='x', padx=10, pady=10)
        
        # Isolation Forest参数
        if_frame = ttk.Frame(param_frame)
        if_frame.pack(fill='x')
        
        ttk.Label(if_frame, text="异常比例(0-1):").grid(row=0, column=0, sticky='w')
        contamination_var = tk.StringVar(value="0.05")
        contamination_entry = ttk.Entry(if_frame, textvariable=contamination_var, width=10)
        contamination_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # 添加应用按钮
        def apply_multi_analysis():
            method = method_var.get()
            
            # 收集选中特征的数据
            data_dict = {}
            for feature in selected_features:
                feature_data = []
                for df in self.dfs:
                    if feature in df.columns:
                        feature_data.extend(df[feature].dropna().tolist())
                data_dict[feature] = feature_data
            
            # 检查数据长度是否一致
            lengths = [len(data) for data in data_dict.values()]
            if len(set(lengths)) > 1:
                tk.messagebox.showerror("错误", "所选特征的数据长度不一致，无法进行多维分析")
                return
                
            # 构建数据矩阵
            import numpy as np
            data_matrix = np.array([data_dict[feature] for feature in selected_features]).T
            
            # 应用选定的方法
            try:
                if method == "Isolation Forest":
                    from sklearn.ensemble import IsolationForest
                    contamination = float(contamination_var.get())
                    if contamination <= 0 or contamination >= 1:
                        tk.messagebox.showerror("错误", "异常比例必须在0-1之间")
                        return
                        
                    # 训练模型
                    model = IsolationForest(contamination=contamination, random_state=42)
                    model.fit(data_matrix)
                    
                    # 预测
                    predictions = model.predict(data_matrix)
                    valid_indices = np.where(predictions == 1)[0]
                    
                elif method == "马氏距离":
                    # 计算均值向量
                    mean_vec = np.mean(data_matrix, axis=0)
                    # 计算协方差矩阵
                    cov_mat = np.cov(data_matrix, rowvar=False)
                    # 添加正则化确保矩阵可逆
                    cov_mat += 1e-6 * np.eye(cov_mat.shape[0])
                    # 计算逆协方差矩阵
                    inv_cov = np.linalg.inv(cov_mat)
                    # 计算每个点的马氏距离
                    distances = []
                    for i in range(len(data_matrix)):
                        diff = data_matrix[i] - mean_vec
                        dist = np.sqrt(diff.dot(inv_cov).dot(diff))
                        distances.append(dist)
                    
                    # 使用卡方分布确定阈值
                    from scipy.stats import chi2
                    threshold = 0.99  # 99%置信度
                    cutoff = np.sqrt(chi2.ppf(threshold, data_matrix.shape[1]))
                    
                    # 获取非离群点的索引
                    valid_indices = np.where(np.array(distances) <= cutoff)[0]
                    
                elif method == "DBSCAN":
                    from sklearn.cluster import DBSCAN
                    from sklearn.preprocessing import StandardScaler
                    
                    # 标准化数据
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(data_matrix)
                    
                    # 应用DBSCAN
                    db = DBSCAN(eps=0.5, min_samples=5).fit(scaled_data)
                    
                    # 获取非噪声点的索引
                    valid_indices = np.where(db.labels_ != -1)[0]
                
                # 计算每个特征的范围
                filtered_data = data_matrix[valid_indices]
                
                # 输出结果
                result_text = f"多维分析结果 ({method}):\n"
                result_text += f"总样本数: {len(data_matrix)}\n"
                result_text += f"有效样本数: {len(valid_indices)}\n"
                result_text += f"剔除样本数: {len(data_matrix) - len(valid_indices)}\n"
                result_text += f"良率: {len(valid_indices) / len(data_matrix) * 100:.2f}%\n\n"
                
                # 更新每个特征的上下限
                for i, feature in enumerate(selected_features):
                    feature_data = filtered_data[:, i]
                    min_val = np.min(feature_data)
                    max_val = np.max(feature_data)
                    
                    # 更新到UI和limits字典
                    for item in self.column_tree.get_children():
                        if self.column_tree.item(item, 'values')[1] == feature:
                            current_values = list(self.column_tree.item(item, 'values'))
                            current_values[2] = f'{min_val:.4f}'
                            current_values[3] = f'{max_val:.4f}'
                            current_values[4] = '多维分析'
                            self.column_tree.item(item, values=tuple(current_values))
                            self.column_tree.item(item, tags=('recommended',))
                            
                            # 更新limits参数
                            self.limits[feature] = (min_val, max_val)
                            
                            result_text += f"{feature}: [{min_val:.4f}, {max_val:.4f}]\n"
                
                # 显示结果
                result_win = tk.Toplevel(multi_win)
                result_win.title("多维分析结果")
                result_win.geometry("400x300")
                
                result_text_widget = tk.Text(result_win, wrap='word')
                result_text_widget.pack(fill='both', expand=True, padx=10, pady=10)
                result_text_widget.insert('1.0', result_text)
                result_text_widget.config(state='disabled')
                
                # 更新状态
                self.status_label.config(text=f"多维分析完成: 良率 {len(valid_indices) / len(data_matrix) * 100:.2f}%")
                
                tk.messagebox.showinfo("完成", "多维分析已完成，已更新所选特征的上下限")
                
            except Exception as e:
                tk.messagebox.showerror("错误", f"多维分析出错: {str(e)}")
                
        # 添加按钮
        button_frame = ttk.Frame(multi_win)
        button_frame.pack(fill='x', pady=20)
        
        ttk.Button(button_frame, text="应用", command=apply_multi_analysis).pack(side='right', padx=10)
        ttk.Button(button_frame, text="取消", command=multi_win.destroy).pack(side='right', padx=10)

    def smart_recommend(self):
        """根据数据分布特性智能推荐上下限"""
        # 获取当前选中的列
        selected_columns = []
        for item in self.column_tree.get_children():
            if self.column_tree.item(item, 'values')[0] == 'True':
                selected_columns.append(self.column_tree.item(item, 'values')[1])
        
        if not selected_columns:
            tk.messagebox.showerror("错误", "请至少选择一个特征项")
            return
        
        # 确保analyzer已初始化
        if not hasattr(self, 'analyzer') or self.analyzer is None:
            if self.files:
                self.analyzer = DataAnalyzer(self.files, analyzer=self)
            else:
                tk.messagebox.showerror("错误", "请先选择文件")
                return
        
        # 创建严格度选择对话框
        strictness_win = tk.Toplevel(self.root)
        strictness_win.title("选择推荐严格度")
        strictness_win.geometry("400x300")  # 增加高度以确保按钮可见
        strictness_win.grab_set()  # 使窗口成为模态窗口
        strictness_win.transient(self.root)  # 设置为主窗口的子窗口
        
        # 添加标题
        title_label = ttk.Label(strictness_win, text="请选择推荐严格度", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # 添加说明
        ttk.Label(strictness_win, text="严格度会影响推荐的上下限范围：", font=("Arial", 11)).pack(anchor='w', padx=20, pady=5)
        
        # 创建单选按钮框架
        radio_frame = ttk.Frame(strictness_win, padding=10)
        radio_frame.pack(fill='x', padx=20, pady=5)
        
        # 创建单选按钮变量
        strictness_var = tk.StringVar(value="balanced")
        
        # 创建单选按钮
        ttk.Radiobutton(
            radio_frame, 
            text="严格 - 更窄的范围，可能降低良率但提高质量", 
            variable=strictness_var, 
            value="strict",
            padding=5
        ).pack(anchor='w', pady=5)
        
        ttk.Radiobutton(
            radio_frame, 
            text="平衡 - 适中的范围，平衡良率和质量", 
            variable=strictness_var, 
            value="balanced",
            padding=5
        ).pack(anchor='w', pady=5)
        
        ttk.Radiobutton(
            radio_frame, 
            text="宽松 - 更宽的范围，提高良率但可能降低质量", 
            variable=strictness_var, 
            value="loose",
            padding=5
        ).pack(anchor='w', pady=5)
        
        # 添加按钮 - 确保按钮在窗口底部可见
        button_frame = ttk.Frame(strictness_win)
        button_frame.pack(fill='x', pady=20, side='bottom')
        
        def start_recommendation():
            strictness = strictness_var.get()
            strictness_win.destroy()
            self._perform_smart_recommend(selected_columns, strictness)
        
        # 使用更大、更明显的按钮样式
        style = ttk.Style()
        style.configure("Action.TButton", font=('Arial', 11), padding=8)
        
        # 确保按钮居中显示
        center_frame = ttk.Frame(button_frame)
        center_frame.pack(fill='x')
        
        ttk.Button(center_frame, text="确定", command=start_recommendation, style="Action.TButton").pack(side='right', padx=10)
        ttk.Button(center_frame, text="取消", command=strictness_win.destroy, style="Action.TButton").pack(side='right', padx=10)
    
    def _perform_smart_recommend(self, selected_columns, strictness):
        """执行智能推荐"""
        # 显示进度窗口
        progress_window = tk.Toplevel(self.root)
        progress_window.title("智能推荐进度")
        progress_window.geometry("300x100")
        
        # 创建进度条
        progress = ttk.Progressbar(progress_window, orient="horizontal", length=250, mode="determinate")
        progress.pack(pady=20)
        
        # 创建进度标签
        progress_label = ttk.Label(progress_window, text="0%")
        progress_label.pack()
        
        # 更新进度条
        def update_progress(value):
            progress['value'] = value
            progress_label.config(text=f"{value}%")
            progress_window.update_idletasks()
        
        try:
            # 获取智能推荐结果
            update_progress(10)
            limits, methods = self.analyzer.smart_recommend_limits_for_columns(selected_columns, strictness)
            
            # 关闭进度窗口
            progress_window.destroy()
            
            # 创建确认对话框
            confirm_win = tk.Toplevel(self.root)
            confirm_win.title("确认推荐上下限")
            confirm_win.geometry("800x600")
            confirm_win.grab_set()  # 使窗口成为模态窗口
            
            # 添加标题
            title_label = ttk.Label(confirm_win, text="请确认推荐的上下限", font=("Arial", 14, "bold"))
            title_label.pack(pady=10)
            
            # 显示当前严格度
            strictness_map = {"strict": "严格", "balanced": "平衡", "loose": "宽松"}
            strictness_label = ttk.Label(
                confirm_win, 
                text=f"当前严格度: {strictness_map.get(strictness, strictness)}", 
                font=("Arial", 12)
            )
            strictness_label.pack(pady=5)
            
            # 创建表格显示推荐结果
            frame = ttk.Frame(confirm_win, padding=10)
            frame.pack(fill='both', expand=True)
            
            # 创建表格
            columns = ('feature', 'distribution', 'lower', 'upper', 'method')
            tree = ttk.Treeview(frame, columns=columns, show='headings')
            tree.heading('feature', text='特征项')
            tree.heading('distribution', text='分布类型')
            tree.heading('lower', text='推荐下限')
            tree.heading('upper', text='推荐上限')
            tree.heading('method', text='推荐方法')
            
            tree.column('feature', width=120)
            tree.column('distribution', width=150)
            tree.column('lower', width=100)
            tree.column('upper', width=100)
            tree.column('method', width=100)
            
            # 添加滚动条
            scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            
            # 放置树形视图和滚动条
            tree.pack(side='left', fill='both', expand=True, pady=10)
            scrollbar.pack(side='right', fill='y', pady=10)
            
            # 填充数据
            for col in selected_columns:
                # 获取分布信息
                dist_info = self.analyzer.analyze_distribution(col)
                distribution_type = dist_info['distribution']
                
                # 翻译分布类型为中文
                dist_map = {
                    'normal': '正态分布',
                    'near-normal': '接近正态分布',
                    'right-skewed': '右偏分布',
                    'left-skewed': '左偏分布',
                    'lognormal': '对数正态分布',
                    't-distribution': 't分布',
                    'uniform': '均匀分布',
                    'unknown': '未知分布'
                }
                
                dist_name = dist_map.get(distribution_type, distribution_type)
                
                # 获取推荐的上下限
                lower, upper = limits[col]
                method = methods[col]
                
                # 添加到表格
                tree.insert('', 'end', values=(col, dist_name, f'{lower:.4f}', f'{upper:.4f}', method))
            
            # 添加说明
            ttk.Label(frame, text="您可以双击表格中的上下限值进行调整", font=("Arial", 11)).pack(anchor='w', pady=5)
            
            # 添加双击编辑功能
            def on_double_click(event):
                item = tree.selection()[0]
                column = tree.identify_column(event.x)
                
                # 只允许编辑下限和上限列
                if column == '#3' or column == '#4':  # 下限或上限列
                    x, y, width, height = tree.bbox(item, column)
                    
                    # 创建编辑框
                    entry = ttk.Entry(tree)
                    entry.place(x=x, y=y, width=width, height=height)
                    
                    # 设置当前值
                    value = tree.item(item, 'values')
                    col_idx = int(column[1]) - 1
                    entry.insert(0, value[col_idx])
                    entry.select_range(0, tk.END)
                    entry.focus()
                    
                    def on_entry_return(event):
                        # 获取新值
                        new_value = entry.get()
                        try:
                            # 验证是否为有效数字
                            new_float = float(new_value)
                            
                            # 更新表格
                            values = list(tree.item(item, 'values'))
                            values[col_idx] = f'{new_float:.4f}'
                            tree.item(item, values=tuple(values))
                        except ValueError:
                            pass
                        finally:
                            entry.destroy()
                    
                    entry.bind('<Return>', on_entry_return)
                    entry.bind('<FocusOut>', lambda e: entry.destroy())
            
            tree.bind('<Double-1>', on_double_click)
            
            # 添加按钮
            button_frame = ttk.Frame(confirm_win)
            button_frame.pack(fill='x', pady=20)
            
            def apply_recommendations():
                # 更新UI和limits
                for item in tree.get_children():
                    values = tree.item(item, 'values')
                    col = values[0]
                    lower = float(values[2])
                    upper = float(values[3])
                    method = values[4]
                    dist_type = values[1]
                    
                    for tree_item in self.column_tree.get_children():
                        if self.column_tree.item(tree_item, 'values')[1] == col:
                            current_values = list(self.column_tree.item(tree_item, 'values'))
                            current_values[2] = f'{lower:.4f}'
                            current_values[3] = f'{upper:.4f}'
                            current_values[4] = f'智能推荐({strictness_map.get(strictness, "")})'
                            self.column_tree.item(tree_item, values=tuple(current_values))
                            self.column_tree.item(tree_item, tags=('smart_recommended',))
                            self.column_tree.tag_configure('smart_recommended', background='#E1F5FE')
                            
                            # 更新limits参数
                            self.limits[col] = (lower, upper)
                            break
                
                tk.messagebox.showinfo("完成", f"智能推荐({strictness_map.get(strictness, '')})已应用！已根据数据分布特性为每个特征项选择最合适的方法。")
                confirm_win.destroy()
            
            # 使用更大、更明显的按钮
            style = ttk.Style()
            style.configure("Action.TButton", font=('Arial', 11), padding=8)
            
            ttk.Button(button_frame, text="应用并确认", command=apply_recommendations, style="Action.TButton").pack(side='right', padx=10)
            ttk.Button(button_frame, text="取消", command=confirm_win.destroy, style="Action.TButton").pack(side='right', padx=10)
            
        except Exception as e:
            progress_window.destroy()
            tk.messagebox.showerror("错误", f"智能推荐出错: {str(e)}")

    def generate_distribution_output(self):
        """打开对话框选择分布输出维度和列"""
        # 确保有选中的特征项
        if not self.selected_columns:
            tk.messagebox.showerror("错误", "请先在分析设置中选择特征项")
            return
        
        # 确保analyzer已初始化
        if not hasattr(self, 'analyzer') or self.analyzer is None:
            if self.files:
                self.analyzer = DataAnalyzer(self.files, analyzer=self)
            else:
                tk.messagebox.showerror("错误", "请先选择文件")
                return

        output_win = tk.Toplevel(self.root)
        output_win.title("产出分布输出")
        output_win.geometry("800x600")
        output_win.grab_set()  # 使窗口成为模态窗口
        
        # 创建主框架
        main_frame = ttk.Frame(output_win, padding=10)
        main_frame.pack(fill='both', expand=True)
        
        # 创建左右分栏
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # 左侧 - 维度选择和列选择
        dimension_frame = ttk.LabelFrame(left_frame, text="选择维度", padding=10)
        dimension_frame.pack(fill='x', pady=(0, 10))
        
        dimension_var = tk.StringVar(value="1D")
        
        # 创建单选按钮
        ttk.Radiobutton(
            dimension_frame, 
            text="一维", 
            variable=dimension_var, 
            value="1D",
            command=lambda: self._update_distribution_ui(dimension_var.get(), self.column_tree, right_frame)
        ).pack(anchor='w', pady=2)
        
        ttk.Radiobutton(
            dimension_frame, 
            text="二维", 
            variable=dimension_var, 
            value="2D",
            command=lambda: self._update_distribution_ui(dimension_var.get(), self.column_tree, right_frame)
        ).pack(anchor='w', pady=2)
        
        ttk.Radiobutton(
            dimension_frame, 
            text="三维", 
            variable=dimension_var, 
            value="3D",
            command=lambda: self._update_distribution_ui(dimension_var.get(), self.column_tree, right_frame)
        ).pack(anchor='w', pady=2)
        
        # 列选择框架
        columns_frame = ttk.LabelFrame(left_frame, text="选择列", padding=10)
        columns_frame.pack(fill='both', expand=True)
        
        # 创建树形视图和滚动条
        tree_frame = ttk.Frame(columns_frame)
        tree_frame.pack(fill='both', expand=True)
        
        # 创建树形视图
        column_tree = ttk.Treeview(tree_frame, columns=('select', 'name', 'lower', 'upper', 'interval'), 
                                   show='headings', selectmode='none')
        column_tree.column('select', width=60, anchor='w')
        column_tree.column('name', width=150, anchor='w')
        column_tree.column('lower', width=80, anchor='w')
        column_tree.column('upper', width=80, anchor='w')
        column_tree.column('interval', width=80, anchor='w')
        column_tree.heading('select', text='选择')
        column_tree.heading('name', text='特征项')
        column_tree.heading('lower', text='下限')
        column_tree.heading('upper', text='上限')
        column_tree.heading('interval', text='间隔')
        
        # 添加垂直滚动条
        tree_vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=column_tree.yview)
        column_tree.configure(yscrollcommand=tree_vsb.set)
        
        # 添加水平滚动条
        tree_hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=column_tree.xview)
        column_tree.configure(xscrollcommand=tree_hsb.set)
        
        # 放置树形视图和滚动条
        column_tree.grid(row=0, column=0, sticky='nsew')
        tree_vsb.grid(row=0, column=1, sticky='ns')
        tree_hsb.grid(row=1, column=0, sticky='ew')
        
        # 配置树形视图框架的网格权重
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        
        # 填充树形视图
        for col in self.selected_columns:
            # 获取当前推荐的上下限（如果有）
            lower, upper = self.limits.get(col, (None, None))
            lower_str = f"{lower:.4f}" if lower is not None else ""
            upper_str = f"{upper:.4f}" if upper is not None else ""
            interval_str = "10" # 默认间隔数
            
            item = column_tree.insert('', 'end', values=('False', col, lower_str, upper_str, interval_str))
            column_tree.item(item, tags=('unchecked',))
        
        # 配置颜色标签
        column_tree.tag_configure('checked', background='#E0F0FF')
        column_tree.tag_configure('unchecked', background='white')
        
        # 添加复选框交互
        def on_treeview_click(event):
            region = column_tree.identify_region(event.x, event.y)
            if region == "cell":
                column = column_tree.identify_column(event.x)
                item = column_tree.identify_row(event.y)
                
                # 如果点击了第一列（选择列）
                if column == '#1':
                    current_values = list(column_tree.item(item, 'values'))
                    is_checked = current_values[0] == 'True'
                    
                    # 切换选中状态
                    current_values[0] = 'False' if is_checked else 'True'
                    column_tree.item(item, values=tuple(current_values))
                    
                    # 更新标签
                    column_tree.item(item, tags=('unchecked' if is_checked else 'checked',))
                # 如果点击了下限、上限或间隔列
                elif column in ('#3', '#4', '#5'):
                    edit_cell(item, int(column[1])-1)
        
        def edit_cell(item, col_idx):
            # 获取当前值
            current_values = list(column_tree.item(item, 'values'))
            column_name = current_values[1]
            
            # 创建编辑对话框
            edit_win = tk.Toplevel(output_win)
            edit_win.title(f"编辑 {column_name} 的参数")
            edit_win.geometry("300x150")
            edit_win.grab_set()
            
            # 确定编辑的是哪个参数
            param_name = ""
            if col_idx == 2:
                param_name = "下限"
            elif col_idx == 3:
                param_name = "上限"
            elif col_idx == 4:
                param_name = "间隔数"
            
            # 添加标签和输入框
            ttk.Label(edit_win, text=f"设置 {column_name} 的{param_name}:").pack(pady=(20,5))
            
            entry_var = tk.StringVar(value=current_values[col_idx])
            entry = ttk.Entry(edit_win, textvariable=entry_var, width=20)
            entry.pack(pady=5)
            entry.select_range(0, 'end')
            entry.focus_set()
            
            # 添加按钮
            button_frame = ttk.Frame(edit_win)
            button_frame.pack(pady=20)
            
            def save_value():
                # 尝试验证输入是数字
                try:
                    if col_idx == 4:  # 间隔必须是整数
                        value = int(entry_var.get())
                        if value <= 0:
                            raise ValueError("间隔数必须大于0")
                    else:
                        value = float(entry_var.get())
                    
                    # 更新值
                    current_values[col_idx] = str(value)
                    column_tree.item(item, values=tuple(current_values))
                except ValueError as e:
                    tk.messagebox.showerror("错误", f"请输入有效的数字: {str(e)}")
            
            ttk.Button(button_frame, text="保存", command=save_value).pack(side='left', padx=5)
            ttk.Button(button_frame, text="取消", command=edit_win.destroy).pack(side='left', padx=5)
            
            # 添加回车键支持
            entry.bind('<Return>', lambda event: save_value())
        
        column_tree.bind('<ButtonPress-1>', on_treeview_click)
        
        # 右侧 - 分布设置
        settings_frame = ttk.LabelFrame(right_frame, text="分布设置", padding=10)
        settings_frame.pack(fill='both', expand=True)
        
        # 初始化右侧UI
        self._update_distribution_ui("1D", self.column_tree, right_frame)
        
        # 底部按钮 
        button_frame = ttk.Frame(output_win)
        button_frame.pack(fill='x', pady=20)
        
        def generate_output():
            # 获取选中的列
            selected_columns = []
            for item in column_tree.get_children():
                values = column_tree.item(item, 'values')
                if values[0] == 'True':
                    column_name = values[1]
                    lower = float(values[2]) if values[2] else None
                    upper = float(values[3]) if values[3] else None
                    interval = int(values[4]) if values[4] else 10
                    selected_columns.append((column_name, lower, upper, interval))
            
            if not selected_columns:
                tk.messagebox.showerror("错误", "请至少选择一个列")
                return
            
            # 根据维度执行不同的输出逻辑
            dimension = dimension_var.get()
            
            # 选择保存路径
            output_path = filedialog.asksaveasfilename(
                defaultextension='.xlsx',
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            
            if not output_path:
                return
                
            # 显示进度条
            progress_window = tk.Toplevel(output_win)
            progress_window.title("生成分布输出")
            progress_window.geometry("300x100")
            progress_bar = ttk.Progressbar(progress_window, orient='horizontal', length=250, mode='determinate')
            progress_bar.pack(pady=20)
            progress_label = ttk.Label(progress_window, text="0%")
            progress_label.pack()
            
            def update_progress(value):
                progress_bar['value'] = value
                progress_label.config(text=f"{value}%")
                progress_window.update_idletasks()
            
            try:
                # 获取数据
                combined_data = pd.concat(self.dfs)
                update_progress(10)
                
                # 检查数据是否为空
                if combined_data.empty:
                    raise ValueError("没有可用数据，请确保已加载文件并包含有效数据。")
                
                # 创建一个默认工作表标志，确保至少有一个工作表被创建
                created_at_least_one_sheet = False
                
                # 创建Excel写入器
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    # 根据不同维度执行不同的输出逻辑
                    if dimension == "1D":
                        # 一维分布 - 简化为只输出数量和百分比
                        for i, (col, lower, upper, interval) in enumerate(selected_columns):
                            update_progress(10 + int(80 * i / len(selected_columns)))
                            
                            if col in combined_data.columns:
                                data = combined_data[col].dropna()
                                
                                # 检查数据是否足够
                                if len(data) < 2:
                                    # 如果数据不足，创建一个简单的工作表说明情况
                                    pd.DataFrame({'说明': [f'列 "{col}" 中的有效数据不足，无法生成分布']}).to_excel(
                                        writer, sheet_name=f"{col}_数据不足", index=False)
                                    created_at_least_one_sheet = True
                                    continue
                                
                                # 使用指定的上下限和间隔生成分布
                                if lower is not None and upper is not None and interval > 0:
                                    try:
                                        # 创建均匀间隔的分箱
                                        bins = np.linspace(lower, upper, interval + 1)
                                        
                                        # 创建区间标签
                                        bin_labels = [f"{bins[i]:.4f}-{bins[i+1]:.4f}" for i in range(len(bins) - 1)]
                                        
                                        # 将数据分箱并计算每个区间的数量
                                        cut_data = pd.cut(data, bins, labels=bin_labels)
                                        value_counts = cut_data.value_counts().sort_index()
                                        
                                        # 如果所有数据都被筛选掉，添加说明
                                        if len(value_counts) == 0:
                                            pd.DataFrame({'说明': [f'列 "{col}" 中没有数据在指定范围 [{lower}, {upper}] 内']}).to_excel(
                                                writer, sheet_name=f"{col}_范围无数据", index=False)
                                            created_at_least_one_sheet = True
                                            continue
                                        
                                        # 计算百分比
                                        percentages = value_counts / len(data) * 100
                                        
                                        # 创建结果数据框
                                        result_df = pd.DataFrame({
                                            '区间': value_counts.index,
                                            '频数': value_counts.values,
                                            '百分比(%)': percentages.values,
                                            '累计百分比(%)': np.cumsum(percentages.values)
                                        })
                                        
                                        # 添加总计行
                                        total_row = pd.DataFrame({
                                            '区间': ['总计'],
                                            '频数': [sum(value_counts.values)],
                                            '百分比(%)': [100.0],
                                            '累计百分比(%)': [None]
                                        })
                                        result_df = pd.concat([result_df, total_row], ignore_index=True)
                                        
                                        # 写入Excel
                                        result_df.to_excel(writer, sheet_name=f"{col}_分布", index=False)
                                        created_at_least_one_sheet = True
                                        
                                        # 添加简单统计信息
                                        stats_df = pd.DataFrame({
                                            '统计量': ['总样本数', '平均值', '标准差', '最小值', '最大值'],
                                            '值': [
                                                len(data),
                                                np.mean(data),
                                                np.std(data),
                                                np.min(data),
                                                np.max(data)
                                            ]
                                        })
                                        
                                        # 计算超限信息
                                        below_limit = len(data[data < lower])
                                        above_limit = len(data[data > upper])
                                        within_limit = len(data[(data >= lower) & (data <= upper)])
                                        
                                        limit_df = pd.DataFrame({
                                            '范围': ['低于下限', '在范围内', '高于上限'],
                                            '数量': [below_limit, within_limit, above_limit],
                                            '百分比(%)': [
                                                below_limit / len(data) * 100,
                                                within_limit / len(data) * 100,
                                                above_limit / len(data) * 100
                                            ]
                                        })
                                        
                                        # 将统计信息和超限信息写入单独的工作表
                                        stats_df.to_excel(writer, sheet_name=f"{col}_统计信息", index=False)
                                        limit_df.to_excel(writer, sheet_name=f"{col}_超限信息", index=False)
                                    
                                    except Exception as e:
                                        # 如果处理特定列时出错，记录错误并继续处理其他列
                                        error_df = pd.DataFrame({'错误信息': [f'处理列 "{col}" 时出错: {str(e)}']})
                                        error_df.to_excel(writer, sheet_name=f"{col}_错误", index=False)
                                        created_at_least_one_sheet = True
                                        print(f"处理列 {col} 时出错: {str(e)}")
                                        continue
                                else:
                                    # 如果没有指定上下限，使用数据的最小值和最大值
                                    try:
                                        min_val = np.min(data)
                                        max_val = np.max(data)
                                        interval = interval if interval > 0 else 10  # 默认使用10个区间
                                        
                                        if min_val == max_val:  # 处理所有值相同的情况
                                            min_val = min_val - 0.5
                                            max_val = max_val + 0.5
                                            
                                        bins = np.linspace(min_val, max_val, interval + 1)
                                        bin_labels = [f"{bins[i]:.4f}-{bins[i+1]:.4f}" for i in range(len(bins) - 1)]
                                        
                                        # 将数据分箱并计算每个区间的数量
                                        cut_data = pd.cut(data, bins, labels=bin_labels)
                                        value_counts = cut_data.value_counts().sort_index()
                                        
                                        # 计算百分比
                                        percentages = value_counts / len(data) * 100
                                        
                                        # 创建结果数据框
                                        result_df = pd.DataFrame({
                                            '区间': value_counts.index,
                                            '频数': value_counts.values,
                                            '百分比(%)': percentages.values,
                                            '累计百分比(%)': np.cumsum(percentages.values)
                                        })
                                        
                                        # 添加总计行
                                        total_row = pd.DataFrame({
                                            '区间': ['总计'],
                                            '频数': [sum(value_counts.values)],
                                            '百分比(%)': [100.0],
                                            '累计百分比(%)': [None]
                                        })
                                        result_df = pd.concat([result_df, total_row], ignore_index=True)
                                        
                                        # 写入Excel
                                        result_df.to_excel(writer, sheet_name=f"{col}_自动分箱", index=False)
                                        created_at_least_one_sheet = True
                                        
                                        # 添加简单统计信息到单独的工作表
                                        stats_df = pd.DataFrame({
                                            '统计量': ['总样本数', '平均值', '标准差', '最小值', '最大值'],
                                            '值': [
                                                len(data),
                                                np.mean(data),
                                                np.std(data),
                                                np.min(data),
                                                np.max(data)
                                            ]
                                        })
                                        
                                        stats_df.to_excel(writer, sheet_name=f"{col}_自动分箱_统计", index=False)
                                    except Exception as e:
                                        # 如果自动分箱失败，记录错误
                                        error_df = pd.DataFrame({'错误信息': [f'自动分箱失败: {str(e)}']})
                                        error_df.to_excel(writer, sheet_name=f"{col}_错误", index=False)
                                        created_at_least_one_sheet = True
                                        print(f"自动分箱失败: {str(e)}")
                                        continue
                    
                    elif dimension == "2D":
                        # 确保至少选择了两列
                        if len(selected_columns) < 2:
                            tk.messagebox.showerror("错误", "二维分析需要至少选择两列")
                            progress_window.destroy()
                            return
                            
                        # 取前两列进行二维分析
                        col1, lower1, upper1, intervals1 = selected_columns[0]
                        col2, lower2, upper2, intervals2 = selected_columns[1]
                        
                        if col1 in combined_data.columns and col2 in combined_data.columns:
                            # 筛选有效数据
                            valid_data = combined_data[[col1, col2]].dropna()
                            
                            if len(valid_data) < 2:
                                # 数据不足
                                pd.DataFrame({'说明': [f'列 "{col1}" 和 "{col2}" 的有效配对数据不足']}).to_excel(
                                    writer, sheet_name="数据不足", index=False)
                                created_at_least_one_sheet = True
                            else:
                                try:
                                    # 使用指定的上下限创建二维分箱
                                    if all(x is not None for x in [lower1, upper1, lower2, upper2]):
                                        # 创建分箱边界
                                        bins1 = np.linspace(lower1, upper1, intervals1 + 1)
                                        bins2 = np.linspace(lower2, upper2, intervals2 + 1)
                                        
                                        # 创建区间标签
                                        bin_labels1 = [f"{bins1[i]:.4f}-{bins1[i+1]:.4f}" for i in range(len(bins1) - 1)]
                                        bin_labels2 = [f"{bins2[i]:.4f}-{bins2[i+1]:.4f}" for i in range(len(bins2) - 1)]
                                        
                                        # 将数据分箱
                                        cut_data1 = pd.cut(valid_data[col1], bins1, labels=bin_labels1)
                                        cut_data2 = pd.cut(valid_data[col2], bins2, labels=bin_labels2)
                                        
                                        # 创建带有分箱结果的DataFrame
                                        binned_df = pd.DataFrame({
                                            col1: valid_data[col1],
                                            col2: valid_data[col2],
                                            f'{col1}_bin': cut_data1,
                                            f'{col2}_bin': cut_data2
                                        })
                                        
                                        # 移除可能包含NaN的行（超出范围的值）
                                        binned_df = binned_df.dropna()
                                        
                                        if len(binned_df) > 0:
                                            # 创建透视表 - 行为col1的区间，列为col2的区间，值为数量和百分比
                                            pivot_count = pd.pivot_table(
                                                binned_df,
                                                values=col1,
                                                index=f'{col1}_bin',
                                                columns=f'{col2}_bin',
                                                aggfunc='count',
                                                fill_value=0
                                            )
                                            
                                            # 计算百分比
                                            pivot_percent = pivot_count / len(valid_data) * 100
                                            
                                            # 添加行和列的汇总
                                            pivot_count.loc['总计'] = pivot_count.sum()
                                            pivot_count['总计'] = pivot_count.sum(axis=1)
                                            
                                            pivot_percent.loc['总计'] = pivot_percent.sum()
                                            pivot_percent['总计'] = pivot_percent.sum(axis=1)
                                            
                                            # 写入Excel
                                            pivot_count.to_excel(writer, sheet_name="二维分布_频数")
                                            pivot_percent.to_excel(writer, sheet_name="二维分布_百分比")
                                            created_at_least_one_sheet = True
                                            
                                            # 添加统计信息
                                            stats_df = pd.DataFrame({
                                                '统计量': ['总数', f'{col1}平均值', f'{col2}平均值', f'{col1}标准差', f'{col2}标准差', '相关系数'],
                                                '值': [
                                                    len(valid_data),
                                                    np.mean(valid_data[col1]),
                                                    np.mean(valid_data[col2]),
                                                    np.std(valid_data[col1]),
                                                    np.std(valid_data[col2]),
                                                    np.corrcoef(valid_data[col1], valid_data[col2])[0, 1]
                                                ]
                                            })
                                            
                                            stats_df.to_excel(writer, sheet_name="二维分布_统计信息", index=False)
                                        else:
                                            # 没有数据在指定范围内
                                            pd.DataFrame({
                                                '说明': [f'在指定范围内 [{col1}: {lower1}-{upper1}, {col2}: {lower2}-{upper2}] 没有数据']
                                            }).to_excel(writer, sheet_name="范围无数据", index=False)
                                            created_at_least_one_sheet = True
                                    else:
                                        # 如果未指定上下限，使用自动范围
                                        pd.DataFrame({
                                            '说明': ['请为二维分析指定上下限']
                                        }).to_excel(writer, sheet_name="需要上下限", index=False)
                                        created_at_least_one_sheet = True
                                except Exception as e:
                                    # 二维分析错误
                                    pd.DataFrame({
                                        '错误信息': [f'二维分析出错: {str(e)}']
                                    }).to_excel(writer, sheet_name="二维分析错误", index=False)
                                    created_at_least_one_sheet = True
                                    print(f"二维分析出错: {str(e)}")
                    
                    elif dimension == "3D":
                        # 确保至少选择了三列
                        if len(selected_columns) < 3:
                            tk.messagebox.showerror("错误", "三维分析需要至少选择三列")
                            progress_window.destroy()
                            return
                            
                        # 取前三列进行三维分析
                        col1, lower1, upper1, intervals1 = selected_columns[0]
                        col2, lower2, upper2, intervals2 = selected_columns[1]
                        col3, lower3, upper3, intervals3 = selected_columns[2]
                        
                        if all(col in combined_data.columns for col in [col1, col2, col3]):
                            # 筛选有效数据
                            valid_data = combined_data[[col1, col2, col3]].dropna()
                            
                            if len(valid_data) < 3:
                                # 数据不足
                                pd.DataFrame({'说明': [f'列 "{col1}", "{col2}" 和 "{col3}" 的有效配对数据不足']}).to_excel(
                                    writer, sheet_name="数据不足", index=False)
                                created_at_least_one_sheet = True
                            else:
                                try:
                                    # 如果上下限都设置了，进行分箱处理
                                    if all(x is not None for x in [lower1, upper1, lower2, upper2, lower3, upper3]):
                                        # 创建分箱边界
                                        bins1 = np.linspace(lower1, upper1, intervals1 + 1)
                                        bins2 = np.linspace(lower2, upper2, intervals2 + 1)
                                        bins3 = np.linspace(lower3, upper3, intervals3 + 1)
                                        
                                        # 创建区间标签
                                        bin_labels1 = [f"{bins1[i]:.4f}-{bins1[i+1]:.4f}" for i in range(len(bins1) - 1)]
                                        bin_labels2 = [f"{bins2[i]:.4f}-{bins2[i+1]:.4f}" for i in range(len(bins2) - 1)]
                                        bin_labels3 = [f"{bins3[i]:.4f}-{bins3[i+1]:.4f}" for i in range(len(bins3) - 1)]
                                        
                                        # 将数据分箱
                                        cut_data1 = pd.cut(valid_data[col1], bins1, labels=bin_labels1)
                                        cut_data2 = pd.cut(valid_data[col2], bins2, labels=bin_labels2)
                                        cut_data3 = pd.cut(valid_data[col3], bins3, labels=bin_labels3)
                                        
                                        # 创建带有分箱结果的DataFrame
                                        binned_df = pd.DataFrame({
                                            col1: valid_data[col1],
                                            col2: valid_data[col2],
                                            col3: valid_data[col3],
                                            f'{col1}_bin': cut_data1,
                                            f'{col2}_bin': cut_data2,
                                            f'{col3}_bin': cut_data3
                                        })
                                        
                                        # 移除可能包含NaN的行
                                        binned_df = binned_df.dropna()
                                        
                                        if len(binned_df) > 0:
                                            # 创建组合标签列（列2 + 列3的组合）
                                            binned_df['组合'] = binned_df[f'{col2}_bin'].astype(str) + " × " + binned_df[f'{col3}_bin'].astype(str)
                                            
                                            # 创建一个单一的三维分布表
                                            # 行是列1的区间，列是列2和列3的组合
                                            tri_table = pd.crosstab(
                                                index=binned_df[f'{col1}_bin'],
                                                columns=binned_df['组合'],
                                                values=binned_df[col1],
                                                aggfunc='count',
                                                margins=True,
                                                margins_name='总计',
                                                dropna=False
                                            ).fillna(0).astype(int)
                                            
                                            # 添加百分比表格
                                            total_count = len(binned_df)
                                            percent_table = tri_table.copy()
                                            for col in percent_table.columns:
                                                if col != '总计':
                                                    percent_table[col] = (percent_table[col] / total_count * 100).round(2)
                                            
                                            # 输出到Excel
                                            tri_table.to_excel(writer, sheet_name="三维分布")
                                            percent_table.to_excel(writer, sheet_name="三维分布百分比")
                                            created_at_least_one_sheet = True
                                            
                                            # 添加统计信息
                                            stats_df = pd.DataFrame({
                                                '统计量': [
                                                    '总数', 
                                                    f'{col1}平均值', f'{col2}平均值', f'{col3}平均值',
                                                    f'{col1}标准差', f'{col2}标准差', f'{col3}标准差',
                                                    f'{col1}-{col2}相关系数', f'{col1}-{col3}相关系数', f'{col2}-{col3}相关系数'
                                                ],
                                                '值': [
                                                    len(valid_data),
                                                    np.mean(valid_data[col1]), np.mean(valid_data[col2]), np.mean(valid_data[col3]),
                                                    np.std(valid_data[col1]), np.std(valid_data[col2]), np.std(valid_data[col3]),
                                                    np.corrcoef(valid_data[col1], valid_data[col2])[0, 1],
                                                    np.corrcoef(valid_data[col1], valid_data[col3])[0, 1],
                                                    np.corrcoef(valid_data[col2], valid_data[col3])[0, 1]
                                                ]
                                            })
                                            
                                            stats_df.to_excel(writer, sheet_name="三维分布_统计信息", index=False)
                                        else:
                                            # 没有数据在指定范围内
                                            pd.DataFrame({
                                                '说明': [f'在指定范围内没有有效数据点']
                                            }).to_excel(writer, sheet_name="范围无数据", index=False)
                                            created_at_least_one_sheet = True
                                    else:
                                        # 如果没有设置上下限，使用自动范围进行简单统计
                                        summary_df = pd.DataFrame({
                                            '统计量': ['总数', '相关性分析'],
                                            f'{col1}-{col2}': [len(valid_data), np.corrcoef(valid_data[col1], valid_data[col2])[0, 1]],
                                            f'{col1}-{col3}': [len(valid_data), np.corrcoef(valid_data[col1], valid_data[col3])[0, 1]],
                                            f'{col2}-{col3}': [len(valid_data), np.corrcoef(valid_data[col2], valid_data[col3])[0, 1]]
                                        })
                                        summary_df.to_excel(writer, sheet_name="三维数据_统计", index=False)
                                        
                                        # 输出原始数据的摘要
                                        valid_data.describe().to_excel(writer, sheet_name="三维数据_描述")
                                        created_at_least_one_sheet = True
                                except Exception as e:
                                    # 三维分析错误
                                    pd.DataFrame({
                                        '错误信息': [f'三维分析出错: {str(e)}']
                                    }).to_excel(writer, sheet_name="三维分析错误", index=False)
                                    created_at_least_one_sheet = True
                                    print(f"三维分析出错: {str(e)}")
                    
                    # 如果没有创建任何工作表，添加一个默认工作表
                    if not created_at_least_one_sheet:
                        pd.DataFrame({
                            '说明': ['未能生成任何分析结果，请检查数据和参数设置']
                        }).to_excel(writer, sheet_name="无分析结果", index=False)
                
                update_progress(100)
                tk.messagebox.showinfo("成功", f"分布数据已成功导出到: {output_path}")
            except Exception as e:
                tk.messagebox.showerror("错误", f"导出分布数据时出错: {str(e)}")
                print(f"错误详情: {traceback.format_exc()}")
            finally:
                progress_window.destroy()
        
        # 添加按钮
        button_frame = ttk.Frame(output_win)
        button_frame.pack(fill='x', pady=20)
        
        ttk.Button(button_frame, text="生成", command=generate_output).pack(side='right', padx=10)
        ttk.Button(button_frame, text="取消", command=output_win.destroy).pack(side='right', padx=10)
    
    def _update_distribution_ui(self, dimension, column_tree, right_frame):
        """根据所选维度更新分布设置界面"""
        # 清空右侧框架
        for widget in right_frame.winfo_children():
            widget.destroy()
        
        # 创建新的设置框架
        settings_frame = ttk.LabelFrame(right_frame, text="分布设置", padding=10)
        settings_frame.pack(fill='both', expand=True)
        
        if dimension == "1D":
            ttk.Label(settings_frame, text="一维分布设置", font=("Arial", 12, "bold")).pack(pady=10)
            ttk.Label(settings_frame, text="请在左侧选择要分析的列，可选择多列。").pack(anchor='w', pady=5)
            ttk.Label(settings_frame, text="为每个列设置下限、上限和间隔数。").pack(anchor='w', pady=5)
            ttk.Label(settings_frame, text="将为每个所选列生成频率分布表。").pack(anchor='w', pady=5)
            
        elif dimension == "2D":
            ttk.Label(settings_frame, text="二维分布设置", font=("Arial", 12, "bold")).pack(pady=10)
            ttk.Label(settings_frame, text="请在左侧选择要分析的两列。").pack(anchor='w', pady=5)
            ttk.Label(settings_frame, text="如果选择了多于两列，只有前两列会被用于二维分析。").pack(anchor='w', pady=5)
            ttk.Label(settings_frame, text="将生成热力图数据，显示两列数据的联合分布。").pack(anchor='w', pady=5)
            
        elif dimension == "3D":
            ttk.Label(settings_frame, text="三维分布设置", font=("Arial", 12, "bold")).pack(pady=10)
            ttk.Label(settings_frame, text="请在左侧选择要分析的三列。").pack(anchor='w', pady=5)
            ttk.Label(settings_frame, text="如果选择了多于三列，只有前三列会被用于三维分析。").pack(anchor='w', pady=5)
            ttk.Label(settings_frame, text="将生成三维数据表，可在Excel中进一步分析。").pack(anchor='w', pady=5)

    def toggle_auto_detect(self):
        """根据自动检测选项的状态，控制相关UI元素"""
        print(f"toggle_auto_detect被调用，auto_detect_var = {self.auto_detect_var.get()}")
        try:
            if self.auto_detect_var.get():
                # 启用自动检测
                print("启用自动检测模式")
                self.special_char_entry.configure(state='normal')
                self.start_row_entry.configure(state='disabled')
            else:
                # 禁用自动检测，启用手动输入
                print("启用手动输入模式")
                self.special_char_entry.configure(state='disabled')
                self.start_row_entry.configure(state='normal')
            print(f"设置完成: special_char_entry状态={self.special_char_entry['state']}, start_row_entry状态={self.start_row_entry['state']}")
        except Exception as e:
            print(f"toggle_auto_detect出错: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    app = YieldAnalysisApp()
    app.root.mainloop()