"""
mcp_doc_analyzer - 基于 MCP 协议的文档分析服务器

提供文档读取、文本分析、摘要生成和统计计算工具。
"""

__version__ = "1.0.0"
__author__ = "mcp-doc-analyzer"
__description__ = "文档分析MCP服务器"

from mcp_doc_analyzer import reader
from mcp_doc_analyzer import analyzer
from mcp_doc_analyzer import summarizer
from mcp_doc_analyzer import statistics
from mcp_doc_analyzer import utils

__all__ = [
    "reader",
    "analyzer",
    "summarizer",
    "statistics",
    "utils",
]
