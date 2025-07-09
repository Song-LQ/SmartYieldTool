import pandas as pd
import numpy as np
from scipy.stats import iqr
from scipy.spatial.distance import mahalanobis
from scipy import stats

class DataAnalyzer:
    def __init__(self, files, skiprows=16, analyzer=None):
        self.files = files
        self.skiprows = skiprows
        self.analyzer = analyzer  # 添加analyzer属性
        if analyzer is not None and hasattr(analyzer, 'skiprows'):
            self.skiprows = analyzer.skiprows
        self.dfs = [self._read_file(f) for f in files]
        
    def _read_file(self, file_path):
        try:
            print(f"开始读取文件: {file_path}")
            print(f"self.analyzer = {self.analyzer}")
            
            if self.analyzer is not None and hasattr(self.analyzer, 'auto_detect_var') and self.analyzer.auto_detect_var.get():
                # 自动检测模式
                print("使用自动检测模式")
                if hasattr(self.analyzer, 'special_char_entry'):
                    special_char = self.analyzer.special_char_entry.get()
                    print(f"获取到特殊字符: '{special_char}'")
                else:
                    special_char = ','
                    print(f"未找到special_char_entry，使用默认特殊字符: '{special_char}'")
                
                header_row = 0
                data_start_row = 1
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 100:  # 只检查前100行
                            break
                        if special_char in line:
                            header_row = i
                            data_start_row = header_row + 1
                            print(f"找到特殊字符所在行: {header_row}，数据起始行: {data_start_row}")
                            break
                    else:
                        error_msg = f"自动检测失败: 未找到特殊字符'{special_char}'，请检查设置或使用手动模式"
                        print(error_msg)
                        raise ValueError(error_msg)
            else:
                # 手动模式或者没有提供analyzer
                print("使用手动模式")
                if self.analyzer is not None and hasattr(self.analyzer, 'start_row_entry'):
                    try:
                        start_row_text = self.analyzer.start_row_entry.get()
                        print(f"获取到起始行文本: '{start_row_text}'")
                        data_start_row = int(start_row_text)
                        header_row = data_start_row - 1
                        print(f"转换后的行号: header_row={header_row}, data_start_row={data_start_row}")
                    except (ValueError, AttributeError) as e:
                        print(f"获取起始行出错: {e}，使用默认值")
                        # 如果无法获取，使用默认值
                        header_row = self.skiprows
                        data_start_row = header_row + 1
                else:
                    print(f"未找到start_row_entry，使用skiprows值: {self.skiprows}")
                    # 使用初始化时提供的skiprows值
                    header_row = self.skiprows
                    data_start_row = header_row + 1
            
            # 读取标题行
            print(f"Reading header from row {header_row}")
            header_df = pd.read_csv(file_path, skiprows=header_row, nrows=1, header=None)
            columns = [str(col).strip() for col in header_df.iloc[0].tolist()]
            print(f"Found columns: {columns}")
            
            # 读取数据
            print(f"Reading data from row {data_start_row}")
            full_df = pd.read_csv(file_path, skiprows=data_start_row)
            
            # 如果列数不匹配，可能是CSV格式问题，尝试修复
            if len(full_df.columns) != len(columns):
                print(f"列数不匹配: 标题有{len(columns)}列，数据有{len(full_df.columns)}列。尝试修复...")
                # 尝试使用不同的分隔符
                for sep in [',', '\t', ';', ' ']:
                    try:
                        header_df = pd.read_csv(file_path, skiprows=header_row, nrows=1, header=None, sep=sep)
                        test_columns = [str(col).strip() for col in header_df.iloc[0].tolist()]
                        full_df = pd.read_csv(file_path, skiprows=data_start_row, sep=sep)
                        if len(full_df.columns) == len(test_columns):
                            columns = test_columns
                            print(f"修复成功，使用分隔符: {sep}")
                            break
                    except:
                        continue
            
            # 确保列名和数据列数一致
            if len(full_df.columns) != len(columns):
                # 如果仍然不匹配，使用默认列名
                print(f"警告: 列数仍然不匹配。使用默认列名。")
                columns = [f"Column_{i}" for i in range(len(full_df.columns))]
            
            full_df.columns = columns
            return full_df
        except Exception as e:
            print(f"读取文件时出错: {e}")
            import traceback
            traceback.print_exc()
            # 返回空数据帧
            return pd.DataFrame()

    def calculate_limits(self, column, method='3sigma', **params):
        if method not in ['3sigma', 'iqr']:
            raise ValueError(f"不支持的统计方法: {method}")
        combined_data = pd.concat([df[column] for df in self.dfs if column in df])
        
        # 检查分布类型，为偏态分布提供更好的推荐
        skewness = stats.skew(combined_data)
        is_skewed = abs(skewness) > 0.5  # 判断是否为偏态分布
        
        if method == '3sigma':
            mean = combined_data.mean()
            std = combined_data.std()
            
            if is_skewed:
                # 对于偏态分布，使用非对称的sigma倍数
                if skewness > 0:  # 右偏分布
                    # 右偏分布通常右侧尾部更长，左侧更集中
                    lower_sigma = params.get('lower_param', 2.5)  # 左侧使用较小的倍数
                    upper_sigma = params.get('upper_param', 3.5)  # 右侧使用较大的倍数
                else:  # 左偏分布
                    # 左偏分布通常左侧尾部更长，右侧更集中
                    lower_sigma = params.get('lower_param', 3.5)  # 左侧使用较大的倍数
                    upper_sigma = params.get('upper_param', 2.5)  # 右侧使用较小的倍数
                
                return (mean - lower_sigma*std, mean + upper_sigma*std)
            else:
                # 对于正态分布，使用标准的sigma倍数
                return (mean - params.get('lower_param', 3.0)*std, mean + params.get('upper_param', 3.0)*std)
                
        elif method == 'iqr':
            q1 = np.percentile(combined_data, 25)
            q3 = np.percentile(combined_data, 75)
            iqr_value = q3 - q1
            
            if is_skewed:
                # 对于偏态分布，使用非对称的IQR倍数
                if skewness > 0:  # 右偏分布
                    lower_mult = params.get('lower_multiplier', 1.3)  # 左侧使用较小的倍数
                    upper_mult = params.get('upper_multiplier', 2.0)  # 右侧使用较大的倍数
                else:  # 左偏分布
                    lower_mult = params.get('lower_multiplier', 2.0)  # 左侧使用较大的倍数
                    upper_mult = params.get('upper_multiplier', 1.3)  # 右侧使用较小的倍数
                
                lower = q1 - lower_mult * iqr_value
                upper = q3 + upper_mult * iqr_value
            else:
                # 对于正态分布，使用标准的IQR倍数
                lower = q1 - params.get('lower_multiplier', 1.5) * iqr_value
                upper = q3 + params.get('upper_multiplier', 1.5) * iqr_value
                
            return (lower, upper)

    def generate_report(self, selected_columns, output_path, limits):
        """生成分析报告，确保每个文件的良率都被正确输出"""
        report_data = []
        
        # 分文件统计
        for file_idx, (file_path, df) in enumerate(zip(self.files, self.dfs)):
            for col in selected_columns:
                if col in df.columns:
                    data = df[col].dropna()
                    lower, upper = limits.get(col, (None, None))
                    
                    valid = data[(data >= lower) & (data <= upper)] if lower is not None and upper is not None else data
                    
                    # 计算超限颗粒数
                    below_lower = len(data[data < lower]) if lower is not None else 0
                    above_upper = len(data[data > upper]) if upper is not None else 0
                    
                    # 计算总颗粒数
                    total = len(data)
                    
                    base_data = {
                        '文件编号': file_idx+1,
                        '文件名': file_path.split('/')[-1].split('\\')[-1],  # 处理不同操作系统的路径分隔符
                        '特征项': col,
                        '总颗粒数': total,
                        '有效颗粒数': len(valid),
                        '超下限颗粒数': below_lower,
                        '超上限颗粒数': above_upper,
                        '良率': f"{len(valid)/total*100:.2f}%" if total > 0 else 'N/A',
                        '平均值': valid.mean() if len(valid) > 0 else 0,
                        '标准差': valid.std() if len(valid) > 1 else 0,
                        '下限值': lower if lower is not None else '',
                        '上限值': upper if upper is not None else ''
                    }
                    
                    # 添加到报告数据
                    report_data.append(base_data)
        
        # 添加汇总统计
        for col in selected_columns:
            # 收集所有文件中该列的数据
            all_col_data = []
            for df in self.dfs:
                if col in df.columns:
                    all_col_data.append(df[col].dropna())
            
            if all_col_data:
                combined_data = pd.concat(all_col_data)
                lower, upper = limits.get(col, (None, None))
                
                valid = combined_data[(combined_data >= lower) & (combined_data <= upper)] if lower is not None and upper is not None else combined_data
                
                # 计算超限颗粒数
                below_lower = len(combined_data[combined_data < lower]) if lower is not None else 0
                above_upper = len(combined_data[combined_data > upper]) if upper is not None else 0
                
                # 计算总颗粒数
                total = len(combined_data)
                
                base_data = {
                    '文件编号': '汇总',
                    '文件名': '所有文件',
                    '特征项': col,
                    '总颗粒数': total,
                    '有效颗粒数': len(valid),
                    '超下限颗粒数': below_lower,
                    '超上限颗粒数': above_upper,
                    '良率': f"{len(valid)/total*100:.2f}%" if total > 0 else 'N/A',
                    '平均值': valid.mean() if len(valid) > 0 else 0,
                    '标准差': valid.std() if len(valid) > 1 else 0,
                    '下限值': lower if lower is not None else '',
                    '上限值': upper if upper is not None else ''
                }
                
                # 添加到报告数据
                report_data.append(base_data)
        
        try:
            # 将数据转换为DataFrame并保存
            report_df = pd.DataFrame(report_data)
            
            # 重新排序列，使基本信息有序
            base_columns = ['文件编号', '文件名', '特征项', '总颗粒数', '有效颗粒数', 
                           '超下限颗粒数', '超上限颗粒数', '良率', '平均值', '标准差', 
                           '下限值', '上限值']
            
            # 保存排序后的DataFrame
            report_df = report_df[base_columns]
            report_df.to_excel(output_path, index=False)
            return True
        except Exception as e:
            print(f"保存Excel报告失败: {e}")
            return False

    def calculate_limits_for_columns(self, columns, method, **params):
        results = {}
        for col in columns:
            results[col] = self.calculate_limits(col, method, **params)
        return results

    def analyze_distribution(self, column):
        """分析数据分布特性，返回分布信息"""
        combined_data = pd.concat([df[column] for df in self.dfs if column in df])
        if len(combined_data) < 10:  # 数据太少，无法可靠分析
            return {
                'distribution': 'unknown',
                'skewness': 0,
                'kurtosis': 0,
                'is_normal': True,
                'p_value': 1.0,
                'outlier_ratio': 0
            }
            
        # 计算基本统计量
        skewness = stats.skew(combined_data)
        kurtosis = stats.kurtosis(combined_data)
        
        # 正态性检验
        _, p_value = stats.normaltest(combined_data)
        is_normal = p_value > 0.05
        
        # 检测离群值比例
        q1 = np.percentile(combined_data, 25)
        q3 = np.percentile(combined_data, 75)
        iqr_value = q3 - q1
        lower_bound = q1 - 1.5 * iqr_value
        upper_bound = q3 + 1.5 * iqr_value
        outliers = combined_data[(combined_data < lower_bound) | (combined_data > upper_bound)]
        outlier_ratio = len(outliers) / len(combined_data)
        
        # 判断分布类型
        distribution = 'normal'
        if not is_normal:
            if abs(skewness) < 0.5:
                if kurtosis > 0.5:
                    distribution = 't-distribution'
                elif kurtosis < -0.5:
                    distribution = 'uniform'
                else:
                    distribution = 'near-normal'
            elif skewness > 0.5:
                # 检查是否为对数正态分布
                if np.all(combined_data > 0):
                    try:
                        _, lognorm_p = stats.normaltest(np.log(combined_data))
                        if lognorm_p > 0.05:
                            distribution = 'lognormal'
                        else:
                            distribution = 'right-skewed'
                    except:
                        distribution = 'right-skewed'
                else:
                    distribution = 'right-skewed'
            elif skewness < -0.5:
                distribution = 'left-skewed'
        
        return {
            'distribution': distribution,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'is_normal': is_normal,
            'p_value': p_value,
            'outlier_ratio': outlier_ratio
        }
    
    def smart_recommend_limits(self, column, strictness='balanced'):
        """智能推荐上下限，根据数据分布特性自动选择最合适的方法
        
        参数:
            column: 列名
            strictness: 严格程度，可选值为'strict'(严格)、'balanced'(平衡)、'loose'(宽松)
        """
        # 分析数据分布
        dist_info = self.analyze_distribution(column)
        combined_data = pd.concat([df[column] for df in self.dfs if column in df.columns])
        
        # 检查数据是否包含0点或接近0的值
        has_zero = np.any(np.abs(combined_data) < 1e-10)
        data_range = combined_data.max() - combined_data.min()
        data_magnitude = max(abs(combined_data.max()), abs(combined_data.min()))
        
        # 计算基本统计量，用于所有方法
        mean = combined_data.mean()
        std = combined_data.std()
        q1 = np.percentile(combined_data, 25)
        q3 = np.percentile(combined_data, 75)
        iqr_value = q3 - q1
        
        # 根据严格程度设置系数
        if strictness == 'strict':
            strict_factor = 0.7  # 严格模式
            # 对于大数值，使用更严格的控制
            large_value_factor = 0.5 if data_magnitude > 300 else 0.7
        elif strictness == 'loose':
            strict_factor = 1.0  # 宽松模式
            large_value_factor = 0.8 if data_magnitude > 300 else 1.0
        else:  # balanced
            strict_factor = 0.85  # 平衡模式
            large_value_factor = 0.65 if data_magnitude > 300 else 0.85
        
        # 对于大数值，使用相对范围而非绝对范围
        is_large_value = data_magnitude > 300  # 将大数值定义从100改为300
        
        # 根据分布类型选择合适的方法和参数
        if dist_info['distribution'] == 'normal' or dist_info['distribution'] == 'near-normal':
            # 正态或接近正态分布，使用3sigma方法
            if is_large_value:
                # 对于大数值，使用相对标准差（变异系数）或百分位数
                if has_zero:
                    # 有0点时使用变异系数
                    cv = std / abs(mean) if mean != 0 else 0.1  # 变异系数
                    # 使用更严格的系数
                    sigma_factor = 2.0 * strict_factor * large_value_factor
                    lower = mean * (1 - sigma_factor * cv)
                    upper = mean * (1 + sigma_factor * cv)
                else:
                    # 无0点时使用百分位数
                    p_low = 1.0 if strictness == 'strict' else (0.5 if strictness == 'loose' else 0.75)
                    p_high = 99.0 if strictness == 'strict' else (99.5 if strictness == 'loose' else 99.25)
                    lower = np.percentile(combined_data, p_low)
                    upper = np.percentile(combined_data, p_high)
                
                # 确保下限不会变为负数（除非数据本身有负值）
                if lower < 0 and combined_data.min() >= 0:
                    lower = 0
            else:
                # 使用更严格的sigma倍数
                sigma_factor = 2.5 * strict_factor
                lower = mean - sigma_factor * std
                upper = mean + sigma_factor * std
                
            return (lower, upper), '3sigma'
            
        elif dist_info['distribution'] == 'right-skewed':
            # 右偏分布
            if dist_info['outlier_ratio'] > 0.1:
                # 离群值较多，使用IQR方法
                # 对于大数值，调整IQR倍数
                if is_large_value:
                    if has_zero:
                        # 有0点时使用更严格的控制
                        lower_mult = 0.5 * strict_factor * large_value_factor
                        upper_mult = 1.0 * strict_factor * large_value_factor
                    else:
                        # 无0点时使用百分位数
                        p_low = 1.0 if strictness == 'strict' else (0.5 if strictness == 'loose' else 0.75)
                        p_high = 99.0 if strictness == 'strict' else (99.5 if strictness == 'loose' else 99.25)
                        return (np.percentile(combined_data, p_low), np.percentile(combined_data, p_high)), 'percentile'
                else:
                    lower_mult = 1.0 * strict_factor
                    upper_mult = 1.5 * strict_factor
                    
                lower = q1 - lower_mult * iqr_value
                upper = q3 + upper_mult * iqr_value
                
                # 确保下限不会变为负数（除非数据本身有负值）
                if lower < 0 and combined_data.min() >= 0:
                    lower = 0
                    
                return (lower, upper), 'iqr'
            else:
                # 离群值较少，使用调整后的3sigma
                if is_large_value:
                    if has_zero:
                        # 有0点时使用变异系数
                        cv = std / abs(mean) if mean != 0 else 0.1
                        sigma_lower = 1.5 * strict_factor * large_value_factor
                        sigma_upper = 2.0 * strict_factor * large_value_factor
                        lower = mean * (1 - sigma_lower * cv)
                        upper = mean * (1 + sigma_upper * cv)
                    else:
                        # 无0点时使用百分位数
                        p_low = 1.0 if strictness == 'strict' else (0.5 if strictness == 'loose' else 0.75)
                        p_high = 99.0 if strictness == 'strict' else (99.5 if strictness == 'loose' else 99.25)
                        lower = np.percentile(combined_data, p_low)
                        upper = np.percentile(combined_data, p_high)
                    
                    # 确保下限不会变为负数（除非数据本身有负值）
                    if lower < 0 and combined_data.min() >= 0:
                        lower = 0
                else:
                    lower = mean - 2.0 * std * strict_factor
                    upper = mean + 2.5 * std * strict_factor
                    
                return (lower, upper), '3sigma'
                
        elif dist_info['distribution'] == 'left-skewed':
            # 左偏分布
            if dist_info['outlier_ratio'] > 0.1:
                # 离群值较多，使用IQR方法
                if is_large_value:
                    if has_zero:
                        # 有0点时使用更严格的控制
                        lower_mult = 1.0 * strict_factor * large_value_factor
                        upper_mult = 0.5 * strict_factor * large_value_factor
                    else:
                        # 无0点时使用百分位数
                        p_low = 1.0 if strictness == 'strict' else (0.5 if strictness == 'loose' else 0.75)
                        p_high = 99.0 if strictness == 'strict' else (99.5 if strictness == 'loose' else 99.25)
                        return (np.percentile(combined_data, p_low), np.percentile(combined_data, p_high)), 'percentile'
                else:
                    lower_mult = 1.5 * strict_factor
                    upper_mult = 1.0 * strict_factor
                    
                lower = q1 - lower_mult * iqr_value
                upper = q3 + upper_mult * iqr_value
                
                # 确保下限不会变为负数（除非数据本身有负值）
                if lower < 0 and combined_data.min() >= 0:
                    lower = 0
                    
                return (lower, upper), 'iqr'
            else:
                # 离群值较少，使用调整后的3sigma
                if is_large_value:
                    if has_zero:
                        # 有0点时使用变异系数
                        cv = std / abs(mean) if mean != 0 else 0.1
                        sigma_lower = 2.0 * strict_factor * large_value_factor
                        sigma_upper = 1.5 * strict_factor * large_value_factor
                        lower = mean * (1 - sigma_lower * cv)
                        upper = mean * (1 + sigma_upper * cv)
                    else:
                        # 无0点时使用百分位数
                        p_low = 1.0 if strictness == 'strict' else (0.5 if strictness == 'loose' else 0.75)
                        p_high = 99.0 if strictness == 'strict' else (99.5 if strictness == 'loose' else 99.25)
                        lower = np.percentile(combined_data, p_low)
                        upper = np.percentile(combined_data, p_high)
                    
                    # 确保下限不会变为负数（除非数据本身有负值）
                    if lower < 0 and combined_data.min() >= 0:
                        lower = 0
                else:
                    lower = mean - 2.5 * std * strict_factor
                    upper = mean + 2.0 * std * strict_factor
                    
                return (lower, upper), '3sigma'
                
        elif dist_info['distribution'] == 'lognormal':
            # 对数正态分布，在对数空间中使用3sigma，然后转换回原始空间
            # 确保所有值都为正
            if combined_data.min() <= 0:
                # 如果有非正值，使用IQR方法
                lower = max(0, q1 - 1.2 * iqr_value * strict_factor)  # 确保下限不小于0
                upper = q3 + 1.2 * iqr_value * strict_factor
                return (lower, upper), 'iqr'
            else:
                # 对于大数值，使用百分位数
                if is_large_value:
                    p_low = 1.0 if strictness == 'strict' else (0.5 if strictness == 'loose' else 0.75)
                    p_high = 99.0 if strictness == 'strict' else (99.5 if strictness == 'loose' else 99.25)
                    return (np.percentile(combined_data, p_low), np.percentile(combined_data, p_high)), 'percentile'
                else:
                    # 在对数空间中使用sigma
                    log_sigma = 2.0 * strict_factor
                    log_data = np.log(combined_data)
                    log_mean = log_data.mean()
                    log_std = log_data.std()
                    log_lower = log_mean - log_sigma * log_std
                    log_upper = log_mean + log_sigma * log_std
                    return (np.exp(log_lower), np.exp(log_upper)), 'lognormal'
            
        elif dist_info['distribution'] == 't-distribution':
            # t分布，使用分位数方法
            if strictness == 'strict':
                p_low, p_high = 2.5, 97.5
            elif strictness == 'loose':
                p_low, p_high = 0.5, 99.5
            else:  # balanced
                p_low, p_high = 1.0, 99.0
                
            # 对于大数值，使用更严格的分位数
            if is_large_value:
                p_low = p_low * 2 if p_low < 5 else p_low
                p_high = 100 - (100 - p_high) * 2 if p_high > 95 else p_high
                
            lower = np.percentile(combined_data, p_low)
            upper = np.percentile(combined_data, p_high)
                
            # 确保下限不会变为负数（除非数据本身有负值）
            if lower < 0 and combined_data.min() >= 0:
                lower = 0
                
            return (lower, upper), 'percentile'
            
        elif dist_info['distribution'] == 'uniform':
            # 均匀分布，使用扩展的最小/最大值，但对大数值进行调整
            min_val = combined_data.min()
            max_val = combined_data.max()
            range_val = max_val - min_val
            
            # 对于大数值，使用更小的扩展比例或直接使用分位数
            if is_large_value:
                if strictness == 'strict':
                    p_low, p_high = 2.5, 97.5
                elif strictness == 'loose':
                    p_low, p_high = 0.5, 99.5
                else:  # balanced
                    p_low, p_high = 1.0, 99.0
                    
                lower = np.percentile(combined_data, p_low)
                upper = np.percentile(combined_data, p_high)
            else:
                # 根据严格程度调整扩展比例
                if strictness == 'strict':
                    extension = 0.01 * strict_factor
                elif strictness == 'loose':
                    extension = 0.05 * strict_factor
                else:  # balanced
                    extension = 0.03 * strict_factor
                    
                lower = min_val - extension * range_val
                upper = max_val + extension * range_val
            
            # 确保下限不会变为负数（除非数据本身有负值）
            if lower < 0 and min_val >= 0:
                lower = 0
                
            return (lower, upper), 'range'
            
        else:
            # 未知分布，使用IQR方法（较为稳健），但对大数值进行调整
            if is_large_value:
                # 对于大数值，使用分位数
                if strictness == 'strict':
                    p_low, p_high = 2.5, 97.5
                elif strictness == 'loose':
                    p_low, p_high = 0.5, 99.5
                else:  # balanced
                    p_low, p_high = 1.0, 99.0
                    
                lower = np.percentile(combined_data, p_low)
                upper = np.percentile(combined_data, p_high)
            else:
                # 根据严格程度调整IQR倍数
                if strictness == 'strict':
                    mult = 0.8 * strict_factor
                elif strictness == 'loose':
                    mult = 1.5 * strict_factor
                else:  # balanced
                    mult = 1.2 * strict_factor
                    
                lower = q1 - mult * iqr_value
                upper = q3 + mult * iqr_value
            
            # 确保下限不会变为负数（除非数据本身有负值）
            if lower < 0 and combined_data.min() >= 0:
                lower = 0
                
            return (lower, upper), 'iqr'
            
    def smart_recommend_limits_for_columns(self, columns, strictness='balanced'):
        """为多个列智能推荐上下限
        
        参数:
            columns: 列名列表
            strictness: 严格程度，可选值为'strict'(严格)、'balanced'(平衡)、'loose'(宽松)
        """
        results = {}
        methods = {}
        for col in columns:
            (lower, upper), method = self.smart_recommend_limits(col, strictness)
            results[col] = (lower, upper)
            methods[col] = method
        return results, methods