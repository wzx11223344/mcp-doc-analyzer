"""
utils.py - 工具函数模块

提供文件读取、文本处理、表格格式化等通用工具函数。
"""

import os
from typing import List, Optional


def safe_read_file(path: str, encoding: str = "utf-8") -> str:
    """安全读取文件内容。

    支持自动尝试多种编码，处理读取异常。

    Args:
        path: 文件路径
        encoding: 首选编码，默认为 utf-8

    Returns:
        文件内容字符串，若读取失败则返回错误信息字符串

    Examples:
        >>> content = safe_read_file("/tmp/test.txt")
        >>> isinstance(content, str)
        True
    """
    if not os.path.exists(path):
        return f"[错误] 文件不存在: {path}"

    if not os.path.isfile(path):
        return f"[错误] 路径不是文件: {path}"

    # 尝试的编码列表
    encodings_to_try = [encoding, "utf-8", "gbk", "gb2312", "latin-1"]

    for enc in encodings_to_try:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            return f"[错误] 读取文件失败: {e}"

    return f"[错误] 无法用任何已知编码读取文件: {path}"


def truncate_text(text: str, max_chars: int = 5000) -> str:
    """截断文本到指定长度。

    在指定字符数处截断文本，并附加省略提示。

    Args:
        text: 原始文本
        max_chars: 最大字符数，默认为5000

    Returns:
        截断后的文本，若被截断则末尾追加省略提示

    Examples:
        >>> result = truncate_text("hello world", 5)
        >>> result.endswith("...")
        True
    """
    if max_chars <= 0:
        return text

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + f"\n\n... [文本已截断，共 {len(text)} 字符，仅显示前 {max_chars} 字符]"


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """将数据格式化为 Markdown 表格。

    Args:
        headers: 表头列表
        rows: 数据行列表，每行为字符串列表

    Returns:
        Markdown 格式的表格字符串

    Examples:
        >>> table = format_table(["Name", "Value"], [["a", "1"], ["b", "2"]])
        >>> "|" in table
        True
    """
    if not headers:
        return ""

    # 构建表头行
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"

    # 构建数据行
    data_lines = []
    for row in rows:
        # 确保每行列数与表头一致
        padded_row = list(row) + [""] * (len(headers) - len(row))
        padded_row = padded_row[:len(headers)]
        data_lines.append("| " + " | ".join(str(cell) for cell in padded_row) + " |")

    return "\n".join([header_line, separator_line] + data_lines)


def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """将文本分割为指定大小的块。

    优先在段落边界分割，若单段超过 chunk_size 则按句子分割。

    Args:
        text: 原始文本
        chunk_size: 每块最大字符数，默认为1000

    Returns:
        文本块列表

    Examples:
        >>> chunks = chunk_text("Hello. World.", 10)
        >>> len(chunks) >= 1
        True
    """
    if chunk_size <= 0:
        return [text]

    if not text:
        return []

    # 先按段落分割
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # 如果当前段落本身就超过 chunk_size，需要进一步分割
        if len(para) > chunk_size:
            # 先保存已有内容
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # 按句子分割长段落
            sentences = _split_sentences(para)
            for sentence in sentences:
                if len(sentence) <= chunk_size:
                    if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                        current_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                else:
                    # 句子仍然太长，强制按字符截断
                    for i in range(0, len(sentence), chunk_size):
                        chunks.append(sentence[i:i + chunk_size])
        else:
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk = (current_chunk + "\n\n" + para).strip() if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _split_sentences(text: str) -> List[str]:
    """将文本分割为句子（内部使用）。

    支持中英文标点符号。

    Args:
        text: 原始文本

    Returns:
        句子列表
    """
    import re
    # 匹配中英文句子结束标点
    sentence_endings = r'(?<=[.!?。！？；;])\s+'
    sentences = re.split(sentence_endings, text)
    # 过滤空句子
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences if sentences else [text]


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小为易读字符串。

    Args:
        size_bytes: 文件大小（字节数）

    Returns:
        格式化后的大小字符串，如 "1.5 MB"
    """
    if size_bytes < 0:
        return "0 B"
    size_float = float(size_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    while size_float >= 1024 and unit_index < len(units) - 1:
        size_float /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(size_float)} {units[unit_index]}"
    return f"{size_float:.1f} {units[unit_index]}"


def get_file_info(path: str) -> dict:
    """获取文件基本信息。

    Args:
        path: 文件路径

    Returns:
        包含文件信息的字典
    """
    if not os.path.exists(path):
        return {"exists": False}

    stat = os.stat(path)
    return {
        "exists": True,
        "size": stat.st_size,
        "size_readable": format_file_size(stat.st_size),
        "is_file": os.path.isfile(path),
        "is_dir": os.path.isdir(path),
        "modified": stat.st_mtime,
    }
