# 文档分析MCP服务器 (mcp-doc-analyzer)

[![CI](https://github.com/wzx11223344/mcp-doc-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/wzx11223344/mcp-doc-analyzer/actions/workflows/ci.yml)

基于 MCP (Model Context Protocol) 协议的文档分析服务器，使用 FastMCP 框架，为 AI 客户端提供文档读取、分析、摘要和统计工具。

## 功能概览

本服务器提供 **21 个工具**，分为四大类：

| 类别 | 工具数 | 说明 |
|------|--------|------|
| 文档读取 | 6 | 文本文件、CSV、JSON 读取，文件头尾读取，目录列表 |
| 文本分析 | 6 | 词频统计、关键词提取、句子分析、可读性评分、情感分析、语言检测 |
| 摘要工具 | 4 | 抽取式摘要、文本统计、相似句子检测、自动大纲 |
| 数据统计 | 5 | 描述性统计、频率分布、相关性分析、趋势分析、异常值检测 |

## 项目结构

```
mcp-doc-analyzer/
├── server.py                    # MCP Server 入口
├── mcp_doc_analyzer/
│   ├── __init__.py              # 包初始化
│   ├── reader.py                # 文档读取工具（6个）
│   ├── analyzer.py              # 文本分析工具（6个）
│   ├── summarizer.py            # 摘要工具（4个）
│   ├── statistics.py            # 统计工具（5个）
│   └── utils.py                 # 工具函数
├── README.md                    # 项目说明（本文件）
├── SKILL.md                     # 技能文档
└── requirements.txt             # Python 依赖
```

## 安装

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖列表：
- `mcp` - MCP 协议核心库（含 FastMCP）
- `numpy` - 数值计算
- `scipy` - 科学计算（统计模块）
- `jieba` - 中文分词
- `scikit-learn` - TF-IDF 关键词提取

### 2. 运行服务器

```bash
python server.py
```

### 3. 配置 MCP 客户端

在 MCP 客户端（如 Claude Desktop）配置文件中添加：

```json
{
    "mcpServers": {
        "doc-analyzer": {
            "command": "python",
            "args": ["/path/to/mcp-doc-analyzer/server.py"]
        }
    }
}
```

## 工具列表

### 文档读取工具

| 工具名 | 参数 | 说明 |
|--------|------|------|
| `read_text_file` | `path: str` | 读取纯文本文件（txt/md/log） |
| `read_csv_file` | `path: str, max_rows: int=100` | 读取CSV文件，返回结构化摘要 |
| `read_json_file` | `path: str` | 读取JSON文件，返回格式化摘要 |
| `read_file_head` | `path: str, n_lines: int=10` | 读取文件头部N行 |
| `read_file_tail` | `path: str, n_lines: int=10` | 读取文件尾部N行 |
| `list_directory` | `path: str, pattern: str="*"` | 列出目录文件（支持glob） |

### 文本分析工具

| 工具名 | 参数 | 说明 |
|--------|------|------|
| `word_frequency` | `text: str, top_n: int=20` | 词频统计（中英文分词） |
| `extract_keywords` | `text: str, top_n: int=10` | 关键词提取（TF-IDF） |
| `sentence_analysis` | `text: str` | 句子分析（数量/长度/最长/最短） |
| `readability_score` | `text: str` | 可读性评分（Flesch-Kincaid改编） |
| `sentiment_analysis` | `text: str` | 情感分析（正面/负面/中性） |
| `language_detect` | `text: str` | 语言检测（中/英/日/韩/混合） |

### 摘要工具

| 工具名 | 参数 | 说明 |
|--------|------|------|
| `extractive_summary` | `text: str, num_sentences: int=3` | 抽取式摘要（TextRank） |
| `text_statistics` | `text: str` | 文本统计（字数/词数/句数/段落/阅读时间） |
| `find_similar_sentences` | `text: str, threshold: float=0.5` | 相似句子检测 |
| `text_outline` | `text: str` | 自动大纲提取 |

### 数据统计工具

| 工具名 | 参数 | 说明 |
|--------|------|------|
| `describe_numeric` | `values: list` | 描述性统计（均值/标准差/四分位） |
| `frequency_distribution` | `values: list, bins: int=10` | 频率分布 |
| `correlation_analysis` | `x: list, y: list` | 相关性分析（Pearson/Spearman） |
| `trend_analysis` | `values: list` | 趋势分析（线性回归/R²） |
| `outlier_detection` | `values: list, method: str="iqr"` | 异常值检测（IQR/Z-score） |

## 技术栈

- **MCP 协议**：使用 FastMCP 框架创建标准 MCP 服务器
- **中文分词**：jieba（支持精确模式和搜索引擎模式）
- **关键词提取**：jieba.analyse + scikit-learn TfidfVectorizer
- **数值计算**：numpy 矩阵运算
- **统计分析**：scipy.stats（Pearson/Spearman/Kendall相关，线性回归）
- **摘要算法**：TextRank（基于 PageRank 的句子排序算法）

## 返回格式

所有工具均返回 **Markdown 格式字符串**，包含：
- 结构化表格（使用 Markdown 表格语法）
- 统计指标和数值
- 解读说明
- 数据预览

## 开发

### 代码规范

- 所有工具有类型提示和 docstring
- 使用 FastMCP 装饰器 `@mcp.tool()`
- 返回字符串（Markdown 格式）
- 模块化设计，各功能分离

### 验证

```bash
python -m py_compile server.py
python -m py_compile mcp_doc_analyzer/__init__.py
python -m py_compile mcp_doc_analyzer/utils.py
python -m py_compile mcp_doc_analyzer/reader.py
python -m py_compile mcp_doc_analyzer/analyzer.py
python -m py_compile mcp_doc_analyzer/summarizer.py
python -m py_compile mcp_doc_analyzer/statistics.py
```
## 测试

运行单元测试：

```bash
pip install pytest flake8
pytest tests/ -v --tb=short
```

代码质量检查：

```bash
flake8 . --count --max-line-length=120 --statistics
```

## 许可证

MIT License
