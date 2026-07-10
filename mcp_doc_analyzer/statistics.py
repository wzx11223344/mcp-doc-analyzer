"""
statistics.py - 数据统计工具模块

提供描述性统计、频率分布、相关性分析、趋势分析和异常值检测工具。
使用 numpy 和 scipy.stats 进行统计计算。
"""

from typing import List, Union, Dict, Any

import numpy as np
from scipy import stats as scipy_stats

from mcp_doc_analyzer.utils import format_table


def describe_numeric(values: List[Union[int, float]]) -> str:
    """计算数值列表的描述性统计。

    包含均值、标准差、最小值、最大值、四分位数等统计指标。

    Args:
        values: 数值列表

    Returns:
        Markdown 格式的描述性统计结果

    Examples:
        >>> result = describe_numeric([1, 2, 3, 4, 5])
        >>> "## 描述性统计" in result
        True
    """
    if not values:
        return "## 描述性统计\n\n[错误] 输入数据为空。"

    arr = np.array(values, dtype=float)

    if arr.size == 0:
        return "## 描述性统计\n\n[错误] 数据为空。"

    # 基本统计量
    count = int(arr.size)
    total = float(arr.sum())
    mean = float(arr.mean())
    median = float(np.median(arr))
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    var = float(arr.var(ddof=1)) if arr.size > 1 else 0.0
    min_val = float(arr.min())
    max_val = float(arr.max())
    data_range = max_val - min_val

    # 四分位数
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1

    # 偏度和峰度
    if arr.size >= 3:
        skewness = float(scipy_stats.skew(arr, bias=False))
        kurtosis = float(scipy_stats.kurtosis(arr, bias=False))
    else:
        skewness = 0.0
        kurtosis = 0.0

    # 变异系数
    cv = (std / mean * 100) if mean != 0 else 0.0

    # 汇总表
    summary_rows = [
        ["计数 (count)", str(count)],
        ["总和 (sum)", f"{total:.4f}"],
        ["均值 (mean)", f"{mean:.4f}"],
        ["中位数 (median)", f"{median:.4f}"],
        ["标准差 (std)", f"{std:.4f}"],
        ["方差 (variance)", f"{var:.4f}"],
        ["最小值 (min)", f"{min_val:.4f}"],
        ["最大值 (max)", f"{max_val:.4f}"],
        ["极差 (range)", f"{data_range:.4f}"],
        ["第一四分位 (Q1)", f"{q1:.4f}"],
        ["第三四分位 (Q3)", f"{q3:.4f}"],
        ["四分位距 (IQR)", f"{iqr:.4f}"],
        ["偏度 (skewness)", f"{skewness:.4f}"],
        ["峰度 (kurtosis)", f"{kurtosis:.4f}"],
        ["变异系数 (CV)", f"{cv:.2f}%"],
    ]

    # 五数概括
    five_number_rows = [
        ["最小值 (Min)", f"{min_val:.4f}"],
        ["第一四分位 (Q1)", f"{q1:.4f}"],
        ["中位数 (Median)", f"{median:.4f}"],
        ["第三四分位 (Q3)", f"{q3:.4f}"],
        ["最大值 (Max)", f"{max_val:.4f}"],
    ]

    result = f"""## 描述性统计

{format_table(
    ["属性", "值"],
    [
        ["数据量", str(count)],
        ["数据类型", "数值型"],
    ]
)}

## 统计汇总

{format_table(["统计量", "值"], summary_rows)}

## 五数概括 (Five-Number Summary)

{format_table(["统计量", "值"], five_number_rows)}
"""
    return result


def frequency_distribution(values: List[Union[int, float]], bins: int = 10) -> str:
    """计算数值的频率分布。

    将数值数据分箱统计，返回频率分布表和直方图描述。

    Args:
        values: 数值列表
        bins: 分箱数量，默认为10

    Returns:
        Markdown 格式的频率分布结果

    Examples:
        >>> result = frequency_distribution([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5)
        >>> "## 频率分布" in result
        True
    """
    if not values:
        return "## 频率分布\n\n[错误] 输入数据为空。"

    arr = np.array(values, dtype=float)

    if arr.size == 0:
        return "## 频率分布\n\n[错误] 数据为空。"

    if bins <= 0:
        bins = 10

    # 自动选择分箱策略
    # 使用 min 和 max 来创建等宽分箱
    data_min = float(arr.min())
    data_max = float(arr.max())

    if data_min == data_max:
        # 所有值相同
        result = f"""## 频率分布

{format_table(
    ["属性", "值"],
    [
        ["数据量", str(arr.size)],
        ["分箱数", "1"],
        ["最小值", f"{data_min:.4f}"],
        ["最大值", f"{data_max:.4f}"],
        ["说明", "所有值相同，无法分箱"],
    ]
)}

## 分布表

{format_table(["区间", "频数", "频率", "累计频率"], [[f"= {data_min:.4f}", str(int(arr.size)), "100.00%", "100.00%"]])}
"""
        return result

    # 计算分箱
    counts, bin_edges = np.histogram(arr, bins=bins)

    # 构建分布表
    total_count = int(arr.size)
    dist_rows = []
    cumulative = 0

    for i in range(len(counts)):
        lower = float(bin_edges[i])
        upper = float(bin_edges[i + 1])
        freq = int(counts[i])
        cumulative += freq
        percentage = (freq / total_count) * 100
        cum_percentage = (cumulative / total_count) * 100

        # 创建区间表示
        interval = f"[{lower:.2f}, {upper:.2f})"
        # 添加柱状图条
        bar_length = int(percentage / 2)
        bar = "#" * bar_length

        dist_rows.append([
            interval,
            str(freq),
            f"{percentage:.2f}%",
            f"{cum_percentage:.2f}%",
            bar,
        ])

    # 计算分箱宽度
    bin_width = (data_max - data_min) / bins

    result = f"""## 频率分布

{format_table(
    ["属性", "值"],
    [
        ["数据量", str(total_count)],
        ["分箱数", str(bins)],
        ["分箱宽度", f"{bin_width:.4f}"],
        ["最小值", f"{data_min:.4f}"],
        ["最大值", f"{data_max:.4f}"],
        ["数据范围", f"{data_max - data_min:.4f}"],
    ]
)}

## 分布表

{format_table(["区间", "频数", "频率", "累计频率", "分布"], dist_rows)}
"""
    return result


def correlation_analysis(x: List[Union[int, float]], y: List[Union[int, float]]) -> str:
    """计算两组数据的相关性分析。

    计算 Pearson 和 Spearman 相关系数及其显著性检验。

    Args:
        x: 第一组数值数据
        y: 第二组数值数据

    Returns:
        Markdown 格式的相关性分析结果

    Examples:
        >>> result = correlation_analysis([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        >>> "## 相关性分析" in result
        True
    """
    if not x or not y:
        return "## 相关性分析\n\n[错误] 输入数据为空。"

    if len(x) != len(y):
        return f"## 相关性分析\n\n[错误] 两组数据长度不匹配（x: {len(x)}, y: {len(y)}）。"

    if len(x) < 2:
        return "## 相关性分析\n\n[错误] 数据量不足，至少需要2个数据点。"

    arr_x = np.array(x, dtype=float)
    arr_y = np.array(y, dtype=float)

    n = len(x)

    # Pearson 相关系数
    if n >= 2:
        pearson_r, pearson_p = scipy_stats.pearsonr(arr_x, arr_y)
        pearson_r = float(pearson_r)
        pearson_p = float(pearson_p)
    else:
        pearson_r = 0.0
        pearson_p = 1.0

    # Spearman 相关系数
    if n >= 2:
        spearman_r, spearman_p = scipy_stats.spearmanr(arr_x, arr_y)
        spearman_r = float(spearman_r)
        spearman_p = float(spearman_p)
    else:
        spearman_r = 0.0
        spearman_p = 1.0

    # Kendall 相关系数
    if n >= 2:
        kendall_tau, kendall_p = scipy_stats.kendalltau(arr_x, arr_y)
        kendall_tau = float(kendall_tau)
        kendall_p = float(kendall_p)
    else:
        kendall_tau = 0.0
        kendall_p = 1.0

    # 判断相关性强度
    def interpret_r(r):
        abs_r = abs(r)
        if abs_r >= 0.8:
            return "强相关"
        elif abs_r >= 0.5:
            return "中等相关"
        elif abs_r >= 0.3:
            return "弱相关"
        else:
            return "极弱或无相关"

    def interpret_p(p):
        if p < 0.01:
            return "高度显著 (p<0.01)"
        elif p < 0.05:
            return "显著 (p<0.05)"
        elif p < 0.1:
            return "边际显著 (p<0.1)"
        else:
            return "不显著 (p>=0.1)"

    # 方向判断
    direction = "正相关" if pearson_r > 0 else ("负相关" if pearson_r < 0 else "无相关")

    correlation_rows = [
        ["Pearson", f"{pearson_r:.4f}", f"{pearson_p:.4f}", interpret_r(pearson_r), interpret_p(pearson_p)],
        ["Spearman", f"{spearman_r:.4f}", f"{spearman_p:.4f}", interpret_r(spearman_r), interpret_p(spearman_p)],
        ["Kendall", f"{kendall_tau:.4f}", f"{kendall_p:.4f}", interpret_r(kendall_tau), interpret_p(kendall_p)],
    ]

    result = f"""## 相关性分析

{format_table(
    ["属性", "值"],
    [
        ["数据量 (n)", str(n)],
        ["X 均值", f"{float(arr_x.mean()):.4f}"],
        ["Y 均值", f"{float(arr_y.mean()):.4f}"],
        ["X 标准差", f"{float(arr_x.std(ddof=1)) if n > 1 else 0:.4f}"],
        ["Y 标准差", f"{float(arr_y.std(ddof=1)) if n > 1 else 0:.4f}"],
        ["相关方向", direction],
    ]
)}

## 相关系数

{format_table(["方法", "相关系数", "p值", "强度", "显著性"], correlation_rows)}

## 解读

- **Pearson 相关系数**：衡量线性相关性，范围 -1 到 1。当前值为 **{pearson_r:.4f}**，表示 **{interpret_r(pearson_r)}** 的 {direction}，显著性检验结果为 **{interpret_p(pearson_p)}**。
- **Spearman 相关系数**：衡量单调相关性（非参数），当前值为 **{spearman_r:.4f}**，表示 **{interpret_r(spearman_r)}**。
- **Kendall 相关系数**：衡量秩相关性，当前值为 **{kendall_tau:.4f}**，表示 **{interpret_r(kendall_tau)}**。
"""
    return result


def trend_analysis(values: List[Union[int, float]]) -> str:
    """分析数值序列的趋势。

    使用线性回归计算趋势斜率、R 平方值，判断数据是上升、下降还是平稳趋势。

    Args:
        values: 数值序列（按时间或顺序排列）

    Returns:
        Markdown 格式的趋势分析结果

    Examples:
        >>> result = trend_analysis([1, 2, 3, 4, 5])
        >>> "## 趋势分析" in result
        True
    """
    if not values:
        return "## 趋势分析\n\n[错误] 输入数据为空。"

    arr = np.array(values, dtype=float)

    if arr.size < 2:
        return "## 趋势分析\n\n[错误] 数据量不足，至少需要2个数据点。"

    n = arr.size
    x = np.arange(n, dtype=float)

    # 线性回归
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, arr)

    slope = float(slope)
    intercept = float(intercept)
    r_value = float(r_value)
    r_squared = r_value ** 2
    p_value = float(p_value)
    std_err = float(std_err)

    # 趋势判断
    if slope > 0:
        if p_value < 0.05:
            trend_direction = "显著上升趋势"
        else:
            trend_direction = "轻微上升趋势（不显著）"
    elif slope < 0:
        if p_value < 0.05:
            trend_direction = "显著下降趋势"
        else:
            trend_direction = "轻微下降趋势（不显著）"
    else:
        trend_direction = "无明显趋势"

    # 计算预测值
    predicted = slope * x + intercept

    # 计算残差
    residuals = arr - predicted
    residual_std = float(np.std(residuals, ddof=2)) if n > 2 else 0.0

    # 移动平均（如果数据量足够）
    ma_rows = []
    if n >= 3:
        window = min(3, n)
        moving_avg = np.convolve(arr, np.ones(window) / window, mode="valid")
        for i, ma in enumerate(moving_avg):
            ma_rows.append([str(i + 1), str(i + window), f"{float(ma):.4f}"])

    # 计算变化率
    first_val = float(arr[0])
    last_val = float(arr[-1])
    change_rate = ((last_val - first_val) / abs(first_val) * 100) if first_val != 0 else 0.0

    # 数据点表格（最多显示20个）
    display_n = min(20, n)
    data_rows = []
    for i in range(display_n):
        data_rows.append([
            str(i + 1),
            f"{float(arr[i]):.4f}",
            f"{float(predicted[i]):.4f}",
            f"{float(residuals[i]):.4f}",
        ])

    result = f"""## 趋势分析

{format_table(
    ["属性", "值"],
    [
        ["数据量 (n)", str(n)],
        ["起始值", f"{first_val:.4f}"],
        ["结束值", f"{last_val:.4f}"],
        ["变化率", f"{change_rate:.2f}%"],
        ["趋势方向", trend_direction],
    ]
)}

## 线性回归结果

{format_table(
    ["属性", "值"],
    [
        ["斜率 (slope)", f"{slope:.6f}"],
        ["截距 (intercept)", f"{intercept:.6f}"],
        ["相关系数 (r)", f"{r_value:.4f}"],
        ["决定系数 (R²)", f"{r_squared:.4f}"],
        ["p 值", f"{p_value:.4f}"],
        ["标准误差 (std_err)", f"{std_err:.6f}"],
        ["残差标准差", f"{residual_std:.4f}"],
        ["回归方程", f"y = {slope:.4f} * x + {intercept:.4f}"],
    ]
)}

## 移动平均（窗口=3）

{format_table(["起始", "结束", "移动平均"], ma_rows if ma_rows else [["（数据量不足）", "", ""]])}

## 数据点（前 {display_n} 个）

{format_table(["序号", "实际值", "预测值", "残差"], data_rows)}

## 解读

- **斜率**：每增加一个时间单位，数值平均变化 **{slope:.4f}**。
- **R²**：回归模型解释了数据变异的 **{r_squared * 100:.1f}%**。
- **显著性**：p 值为 **{p_value:.4f}**，趋势 {'显著' if p_value < 0.05 else '不显著'}（α=0.05）。
- **结论**：数据呈现 **{trend_direction}**。
"""
    return result


def outlier_detection(values: List[Union[int, float]], method: str = "iqr") -> str:
    """检测数值列表中的异常值。

    支持 IQR（四分位距）和 Z-score 两种检测方法。

    Args:
        values: 数值列表
        method: 检测方法，"iqr" 或 "zscore"，默认为 "iqr"

    Returns:
        Markdown 格式的异常值检测结果

    Examples:
        >>> result = outlier_detection([1, 2, 3, 4, 100])
        >>> "## 异常值检测" in result
        True
    """
    if not values:
        return "## 异常值检测\n\n[错误] 输入数据为空。"

    method = method.lower().strip()

    if method not in ("iqr", "zscore", "z-score", "z_score"):
        method = "iqr"

    arr = np.array(values, dtype=float)
    n = arr.size

    if n < 4:
        return f"""## 异常值检测

{format_table(
    ["属性", "值"],
    [
        ["数据量", str(n)],
        ["方法", method.upper()],
        ["说明", "数据量不足（至少需要4个数据点），无法可靠检测异常值"],
    ]
)}
"""

    outliers = []
    method_name = ""

    if method in ("iqr",):
        method_name = "IQR (四分位距法)"

        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # 检测异常值
        for i, val in enumerate(arr):
            val_f = float(val)
            if val_f < lower_bound or val_f > upper_bound:
                outliers.append({
                    "index": i,
                    "value": val_f,
                    "type": "低于下界" if val_f < lower_bound else "高于上界",
                    "detail": f"{'<' + f'{lower_bound:.4f}'}" if val_f < lower_bound else f">{upper_bound:.4f}",
                })

        # 参数表
        params_rows = [
            ["第一四分位 (Q1)", f"{q1:.4f}"],
            ["第三四分位 (Q3)", f"{q3:.4f}"],
            ["四分位距 (IQR)", f"{iqr:.4f}"],
            ["下界 (Q1 - 1.5*IQR)", f"{lower_bound:.4f}"],
            ["上界 (Q3 + 1.5*IQR)", f"{upper_bound:.4f}"],
        ]

    else:  # zscore
        method_name = "Z-Score (标准差法)"

        mean = float(arr.mean())
        std = float(arr.std(ddof=1))

        if std == 0:
            return f"""## 异常值检测

{format_table(
    ["属性", "值"],
    [
        ["数据量", str(n)],
        ["方法", method_name],
        ["均值", f"{mean:.4f}"],
        ["标准差", "0.0000"],
        ["说明", "所有值相同，标准差为0，无法检测异常值"],
    ]
)}
"""

        z_threshold = 3.0

        # 计算 Z-score
        z_scores = (arr - mean) / std

        for i, z in enumerate(z_scores):
            z_f = float(z)
            if abs(z_f) > z_threshold:
                outliers.append({
                    "index": i,
                    "value": float(arr[i]),
                    "type": "Z-score 超阈值",
                    "detail": f"z={z_f:.4f}",
                })

        params_rows = [
            ["均值 (mean)", f"{mean:.4f}"],
            ["标准差 (std)", f"{std:.4f}"],
            ["Z-score 阈值", f"±{z_threshold:.1f}"],
        ]

    # 构建异常值表格
    outlier_rows = []
    for outlier in outliers:
        outlier_rows.append([
            str(outlier["index"] + 1),
            f"{outlier['value']:.4f}",
            outlier["type"],
            outlier["detail"],
        ])

    # 正常值数量
    normal_count = n - len(outliers)
    outlier_percentage = (len(outliers) / n) * 100

    result = f"""## 异常值检测

{format_table(
    ["属性", "值"],
    [
        ["数据量 (n)", str(n)],
        ["检测方法", method_name],
        ["正常值数量", str(normal_count)],
        ["异常值数量", str(len(outliers))],
        ["异常值占比", f"{outlier_percentage:.2f}%"],
    ]
)}

## 方法参数

{format_table(["参数", "值"], params_rows)}

## 检测到的异常值

{format_table(["序号", "值", "类型", "详情"], outlier_rows if outlier_rows else [["（未检测到异常值）", "", "", ""]])}

## 解读

- 使用 **{method_name}** 检测到 **{len(outliers)}** 个异常值（共 {n} 个数据点）。
- 异常值占比 **{outlier_percentage:.2f}%**。
- {'建议检查异常值数据来源，确认是否为测量错误或真实极端值。' if outliers else '数据分布正常，未发现明显异常值。'}
"""
    return result
