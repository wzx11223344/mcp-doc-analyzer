"""
server.py - MCP Server 入口

文档分析MCP服务器（mcp-doc-analyzer）
基于 MCP 协议，使用 FastMCP 框架，为 AI 客户端提供
文档读取、文本分析、摘要生成和统计计算工具。

使用方法:
    python server.py

或通过 MCP 客户端配置:
    {
        "mcpServers": {
            "doc-analyzer": {
                "command": "python",
                "args": ["server.py"]
            }
        }
    }
"""

from mcp.server.fastmcp import FastMCP

from mcp_doc_analyzer import reader, analyzer, summarizer, statistics

# 创建 FastMCP 服务器实例
mcp = FastMCP("doc-analyzer")


# =========================================================================
# 文档读取工具（reader）
# =========================================================================

@mcp.tool()
def read_text_file(path: str) -> str:
    """读取纯文本文件内容（txt/md/log 等格式）。

    支持自动检测编码，返回文件内容及元信息。

    Args:
        path: 文件路径，支持 .txt, .md, .log 等文本格式

    Returns:
        Markdown 格式的文件内容摘要，包含文件信息和文件内容
    """
    return reader.read_text_file(path)


@mcp.tool()
def read_csv_file(path: str, max_rows: int = 100) -> str:
    """读取 CSV 文件，返回结构化摘要。

    解析 CSV 文件，返回列信息、数据预览和统计摘要。

    Args:
        path: CSV 文件路径
        max_rows: 最大返回行数，默认为100

    Returns:
        Markdown 格式的 CSV 文件摘要，包含列信息和数据预览
    """
    return reader.read_csv_file(path, max_rows)


@mcp.tool()
def read_json_file(path: str) -> str:
    """读取 JSON 文件，返回格式化摘要。

    解析 JSON 文件，返回数据类型、结构和格式化预览。

    Args:
        path: JSON 文件路径

    Returns:
        Markdown 格式的 JSON 文件摘要
    """
    return reader.read_json_file(path)


@mcp.tool()
def read_file_head(path: str, n_lines: int = 10) -> str:
    """读取文件头部 N 行。

    返回文件的前 N 行内容，适用于快速预览文件开头。

    Args:
        path: 文件路径
        n_lines: 要读取的行数，默认为10

    Returns:
        Markdown 格式的文件头部内容摘要
    """
    return reader.read_file_head(path, n_lines)


@mcp.tool()
def read_file_tail(path: str, n_lines: int = 10) -> str:
    """读取文件尾部 N 行。

    返回文件的最后 N 行内容，适用于查看日志末尾。

    Args:
        path: 文件路径
        n_lines: 要读取的行数，默认为10

    Returns:
        Markdown 格式的文件尾部内容摘要
    """
    return reader.read_file_tail(path, n_lines)


@mcp.tool()
def list_directory(path: str, pattern: str = "*") -> str:
    """列出目录中的文件，支持 glob 模式匹配。

    列出指定目录下匹配 glob 模式的文件和子目录。

    Args:
        path: 目录路径
        pattern: glob 匹配模式，默认为 "*"（所有文件）

    Returns:
        Markdown 格式的目录列表摘要
    """
    return reader.list_directory(path, pattern)


# =========================================================================
# 文本分析工具（analyzer）
# =========================================================================

@mcp.tool()
def word_frequency(text: str, top_n: int = 20) -> str:
    """统计文本中的词频（支持中英文分词）。

    使用 jieba 进行中文分词，对英文按空格分割，
    统计并返回出现频率最高的词。

    Args:
        text: 要分析的文本
        top_n: 返回前 N 个高频词，默认为20

    Returns:
        Markdown 格式的词频统计结果
    """
    return analyzer.word_frequency(text, top_n)


@mcp.tool()
def extract_keywords(text: str, top_n: int = 10) -> str:
    """从文本中提取关键词（基于 TF-IDF 算法）。

    使用 jieba 的 TF-IDF 关键词提取算法，结合 sklearn 的 TfidfVectorizer，
    提取文本中最具代表性的关键词。

    Args:
        text: 要分析的文本
        top_n: 返回前 N 个关键词，默认为10

    Returns:
        Markdown 格式的关键词提取结果
    """
    return analyzer.extract_keywords(text, top_n)


@mcp.tool()
def sentence_analysis(text: str) -> str:
    """分析文本的句子结构。

    统计句子数量、平均长度、最长句和最短句等信息。

    Args:
        text: 要分析的文本

    Returns:
        Markdown 格式的句子分析结果
    """
    return analyzer.sentence_analysis(text)


@mcp.tool()
def readability_score(text: str) -> str:
    """计算文本的可读性评分（基于 Flesch-Kincaid 的改编版本）。

    对中文文本使用改编的评分算法，对英文文本使用标准 Flesch-Kincaid。
    评分越高表示越容易阅读。

    Args:
        text: 要分析的文本

    Returns:
        Markdown 格式的可读性评分结果
    """
    return analyzer.readability_score(text)


@mcp.tool()
def sentiment_analysis(text: str) -> str:
    """简单的基于词典的情感分析。

    使用预定义的正负面情感词典，统计文本中的正负面词汇，
    返回情感倾向判断（正面/负面/中性）。

    Args:
        text: 要分析的文本

    Returns:
        Markdown 格式的情感分析结果
    """
    return analyzer.sentiment_analysis(text)


@mcp.tool()
def language_detect(text: str) -> str:
    """检测文本的语言（中/英/日/韩/混合）。

    基于字符 Unicode 范围检测，判断文本的主要语言。

    Args:
        text: 要检测的文本

    Returns:
        Markdown 格式的语言检测结果
    """
    return analyzer.language_detect(text)


# =========================================================================
# 摘要工具（summarizer）
# =========================================================================

@mcp.tool()
def extractive_summary(text: str, num_sentences: int = 3) -> str:
    """生成抽取式摘要（基于 TextRank 算法）。

    使用 TextRank 算法计算句子重要性，抽取文本中最重要的句子组成摘要。

    Args:
        text: 要摘要的文本
        num_sentences: 摘要包含的句子数，默认为3

    Returns:
        Markdown 格式的摘要结果
    """
    return summarizer.extractive_summary(text, num_sentences)


@mcp.tool()
def text_statistics(text: str) -> str:
    """文本统计信息（字数/词数/句数/段落数/阅读时间）。

    综合统计文本的基本指标，包括字符数、词数、句子数、段落数和预计阅读时间。

    Args:
        text: 要统计的文本

    Returns:
        Markdown 格式的文本统计结果
    """
    return summarizer.text_statistics(text)


@mcp.tool()
def find_similar_sentences(text: str, threshold: float = 0.5) -> str:
    """检测文本中的相似句子。

    计算所有句子对之间的相似度，找出相似度超过阈值的句子对。

    Args:
        text: 要分析的文本
        threshold: 相似度阈值（0-1），默认为0.5

    Returns:
        Markdown 格式的相似句子检测结果
    """
    return summarizer.find_similar_sentences(text, threshold)


@mcp.tool()
def text_outline(text: str) -> str:
    """从文本中自动提取大纲。

    基于标题标记（#）、关键词和句子重要性，自动生成文本大纲。

    Args:
        text: 要提取大纲的文本

    Returns:
        Markdown 格式的文本大纲
    """
    return summarizer.text_outline(text)


# =========================================================================
# 数据统计工具（statistics）
# =========================================================================

@mcp.tool()
def describe_numeric(values: list) -> str:
    """计算数值列表的描述性统计。

    包含均值、标准差、最小值、最大值、四分位数等统计指标。

    Args:
        values: 数值列表

    Returns:
        Markdown 格式的描述性统计结果
    """
    return statistics.describe_numeric(values)


@mcp.tool()
def frequency_distribution(values: list, bins: int = 10) -> str:
    """计算数值的频率分布。

    将数值数据分箱统计，返回频率分布表和直方图描述。

    Args:
        values: 数值列表
        bins: 分箱数量，默认为10

    Returns:
        Markdown 格式的频率分布结果
    """
    return statistics.frequency_distribution(values, bins)


@mcp.tool()
def correlation_analysis(x: list, y: list) -> str:
    """计算两组数据的相关性分析。

    计算 Pearson 和 Spearman 相关系数及其显著性检验。

    Args:
        x: 第一组数值数据
        y: 第二组数值数据

    Returns:
        Markdown 格式的相关性分析结果
    """
    return statistics.correlation_analysis(x, y)


@mcp.tool()
def trend_analysis(values: list) -> str:
    """分析数值序列的趋势。

    使用线性回归计算趋势斜率、R 平方值，判断数据是上升、下降还是平稳趋势。

    Args:
        values: 数值序列（按时间或顺序排列）

    Returns:
        Markdown 格式的趋势分析结果
    """
    return statistics.trend_analysis(values)


@mcp.tool()
def outlier_detection(values: list, method: str = "iqr") -> str:
    """检测数值列表中的异常值。

    支持 IQR（四分位距）和 Z-score 两种检测方法。

    Args:
        values: 数值列表
        method: 检测方法，"iqr" 或 "zscore"，默认为 "iqr"

    Returns:
        Markdown 格式的异常值检测结果
    """
    return statistics.outlier_detection(values, method)


# =========================================================================
# 主入口
# =========================================================================

if __name__ == "__main__":
    mcp.run()
