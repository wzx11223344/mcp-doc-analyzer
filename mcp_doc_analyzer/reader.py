"""
reader.py - 文档读取工具模块

提供多种文档读取工具，包括纯文本文件、CSV、JSON 读取，
文件头部/尾部读取，以及目录列表功能。
每个工具返回 Markdown 格式的摘要。
"""

import os
import glob
import json
import csv as csv_module
from typing import List, Optional

from mcp_doc_analyzer.utils import (
    safe_read_file,
    truncate_text,
    format_table,
    format_file_size,
    get_file_info,
)


def read_text_file(path: str) -> str:
    """读取纯文本文件内容（txt/md/log 等格式）。

    支持自动检测编码，返回文件内容及元信息。

    Args:
        path: 文件路径，支持 .txt, .md, .log 等文本格式

    Returns:
        Markdown 格式的文件内容摘要，包含文件信息和文件内容

    Examples:
        >>> result = read_text_file("/tmp/readme.md")
        >>> "# 文件信息" in result
        True
    """
    info = get_file_info(path)
    if not info.get("exists"):
        return f"[错误] 文件不存在: {path}"

    if not info.get("is_file"):
        return f"[错误] 路径不是文件: {path}"

    content = safe_read_file(path)
    if content.startswith("[错误]"):
        return content

    # 截断过长的文本
    max_chars = 50000
    truncated = truncate_text(content, max_chars)

    result = f"""## 文件信息

{format_table(
    ["属性", "值"],
    [
        ["文件路径", path],
        ["文件大小", info["size_readable"]],
        ["字符数", str(len(content))],
        ["行数", str(content.count(chr(10)) + 1)],
    ]
)}

## 文件内容

```
{truncated}
```
"""
    return result


def read_csv_file(path: str, max_rows: int = 100) -> str:
    """读取 CSV 文件，返回结构化摘要。

    解析 CSV 文件，返回列信息、数据预览和统计摘要。

    Args:
        path: CSV 文件路径
        max_rows: 最大返回行数，默认为100

    Returns:
        Markdown 格式的 CSV 文件摘要，包含列信息和数据预览

    Examples:
        >>> result = read_csv_file("/tmp/data.csv")
        >>> "## CSV 文件信息" in result
        True
    """
    info = get_file_info(path)
    if not info.get("exists"):
        return f"[错误] 文件不存在: {path}"

    if not info.get("is_file"):
        return f"[错误] 路径不是文件: {path}"

    if max_rows <= 0:
        max_rows = 100

    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv_module.reader(f)
            rows = list(reader)
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="gbk", newline="") as f:
                reader = csv_module.reader(f)
                rows = list(reader)
        except Exception as e:
            return f"[错误] 读取CSV文件失败: {e}"
    except Exception as e:
        return f"[错误] 读取CSV文件失败: {e}"

    if not rows:
        return "## CSV 文件信息\n\n[警告] 文件为空或无有效数据。"

    headers = rows[0]
    data_rows = rows[1:]
    total_rows = len(data_rows)
    display_rows = data_rows[:max_rows]

    # 尝试用 csv.Sniffer 检测是否有标题行
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            sample = f.read(1024)
            sniffer = csv_module.Sniffer()
            has_header = sniffer.has_header(sample)
    except Exception:
        has_header = True

    # 统计每列的非空值数量
    col_stats = []
    for col_idx in range(len(headers)):
        non_null = sum(1 for row in display_rows if col_idx < len(row) and row[col_idx].strip())
        col_stats.append([headers[col_idx], str(non_null)])

    result = f"""## CSV 文件信息

{format_table(
    ["属性", "值"],
    [
        ["文件路径", path],
        ["文件大小", info["size_readable"]],
        ["总行数（含表头）", str(len(rows))],
        ["数据行数", str(total_rows)],
        ["列数", str(len(headers))],
        ["检测到表头", "是" if has_header else "否"],
        ["显示行数", str(len(display_rows))],
    ]
)}

## 列信息

{format_table(["列名", "非空计数"], col_stats)}

## 数据预览（前 {len(display_rows)} 行）

{format_table([str(h) for h in headers], [[str(c) for c in row] for row in display_rows])}
"""
    return result


def read_json_file(path: str) -> str:
    """读取 JSON 文件，返回格式化摘要。

    解析 JSON 文件，返回数据类型、结构和格式化预览。

    Args:
        path: JSON 文件路径

    Returns:
        Markdown 格式的 JSON 文件摘要

    Examples:
        >>> result = read_json_file("/tmp/config.json")
        >>> "## JSON 文件信息" in result
        True
    """
    info = get_file_info(path)
    if not info.get("exists"):
        return f"[错误] 文件不存在: {path}"

    if not info.get("is_file"):
        return f"[错误] 路径不是文件: {path}"

    content = safe_read_file(path)
    if content.startswith("[错误]"):
        return content

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        return f"[错误] JSON 解析失败: {e}"

    # 分析数据结构
    data_type = type(data).__name__
    structure_info = _analyze_json_structure(data)

    # 格式化预览
    formatted = json.dumps(data, ensure_ascii=False, indent=2)
    truncated = truncate_text(formatted, 10000)

    result = f"""## JSON 文件信息

{format_table(
    ["属性", "值"],
    [
        ["文件路径", path],
        ["文件大小", info["size_readable"]],
        ["数据类型", data_type],
        ["字符数", str(len(content))],
    ]
)}

## 数据结构

{structure_info}

## 格式化预览

```json
{truncated}
```
"""
    return result


def read_file_head(path: str, n_lines: int = 10) -> str:
    """读取文件头部 N 行。

    返回文件的前 N 行内容，适用于快速预览文件开头。

    Args:
        path: 文件路径
        n_lines: 要读取的行数，默认为10

    Returns:
        Markdown 格式的文件头部内容摘要

    Examples:
        >>> result = read_file_head("/tmp/log.txt", 5)
        >>> "## 文件头部" in result
        True
    """
    info = get_file_info(path)
    if not info.get("exists"):
        return f"[错误] 文件不存在: {path}"

    if not info.get("is_file"):
        return f"[错误] 路径不是文件: {path}"

    if n_lines <= 0:
        n_lines = 10

    content = safe_read_file(path)
    if content.startswith("[错误]"):
        return content

    lines = content.split("\n")
    head_lines = lines[:n_lines]
    total_lines = len(lines)

    result = f"""## 文件头部（前 {len(head_lines)} 行）

{format_table(
    ["属性", "值"],
    [
        ["文件路径", path],
        ["文件大小", info["size_readable"]],
        ["总行数", str(total_lines)],
        ["读取行数", str(len(head_lines))],
    ]
)}

```
{chr(10).join(head_lines)}
```
"""
    return result


def read_file_tail(path: str, n_lines: int = 10) -> str:
    """读取文件尾部 N 行。

    返回文件的最后 N 行内容，适用于查看日志末尾。

    Args:
        path: 文件路径
        n_lines: 要读取的行数，默认为10

    Returns:
        Markdown 格式的文件尾部内容摘要

    Examples:
        >>> result = read_file_tail("/tmp/log.txt", 5)
        >>> "## 文件尾部" in result
        True
    """
    info = get_file_info(path)
    if not info.get("exists"):
        return f"[错误] 文件不存在: {path}"

    if not info.get("is_file"):
        return f"[错误] 路径不是文件: {path}"

    if n_lines <= 0:
        n_lines = 10

    content = safe_read_file(path)
    if content.startswith("[错误]"):
        return content

    lines = content.split("\n")
    total_lines = len(lines)
    tail_lines = lines[-n_lines:] if total_lines > n_lines else lines

    result = f"""## 文件尾部（后 {len(tail_lines)} 行）

{format_table(
    ["属性", "值"],
    [
        ["文件路径", path],
        ["文件大小", info["size_readable"]],
        ["总行数", str(total_lines)],
        ["读取行数", str(len(tail_lines))],
    ]
)}

```
{chr(10).join(tail_lines)}
```
"""
    return result


def list_directory(path: str, pattern: str = "*") -> str:
    """列出目录中的文件，支持 glob 模式匹配。

    列出指定目录下匹配 glob 模式的文件和子目录。

    Args:
        path: 目录路径
        pattern: glob 匹配模式，默认为 "*"（所有文件）

    Returns:
        Markdown 格式的目录列表摘要

    Examples:
        >>> result = list_directory("/tmp", "*.py")
        >>> "## 目录列表" in result
        True
    """
    if not os.path.exists(path):
        return f"[错误] 目录不存在: {path}"

    if not os.path.isdir(path):
        return f"[错误] 路径不是目录: {path}"

    # 构建 glob 搜索路径
    search_pattern = os.path.join(path, pattern)
    matched_items = sorted(glob.glob(search_pattern))

    if not matched_items:
        return f"""## 目录列表

{format_table(
    ["属性", "值"],
    [
        ["目录路径", path],
        ["匹配模式", pattern],
        ["匹配数量", "0"],
    ]
)}

[提示] 没有找到匹配 `{pattern}` 的文件。
"""

    # 收集文件信息
    file_rows = []
    dir_rows = []
    total_size = 0

    for item in matched_items:
        item_name = os.path.basename(item)
        if os.path.isdir(item):
            dir_rows.append([item_name, "<DIR>", "-"])
        else:
            item_info = get_file_info(item)
            size = item_info.get("size", 0)
            size_readable = item_info.get("size_readable", "0 B")
            total_size += size
            file_rows.append([item_name, "文件", size_readable])

    all_rows = dir_rows + file_rows

    result = f"""## 目录列表

{format_table(
    ["属性", "值"],
    [
        ["目录路径", path],
        ["匹配模式", pattern],
        ["匹配数量", str(len(matched_items))],
        ["文件数", str(len(file_rows))],
        ["子目录数", str(len(dir_rows))],
        ["文件总大小", format_file_size(total_size)],
    ]
)}

## 文件列表

{format_table(["名称", "类型", "大小"], all_rows if all_rows else [["（空）", "", ""]])}
"""
    return result


def _analyze_json_structure(data, indent: int = 0) -> str:
    """分析 JSON 数据结构（内部使用）。

    递归分析 JSON 数据的结构，返回 Markdown 格式的描述。

    Args:
        data: JSON 数据
        indent: 缩进层级

    Returns:
        结构描述字符串
    """
    lines = []
    prefix = "  " * indent

    if isinstance(data, dict):
        lines.append(f"{prefix}- 对象，包含 {len(data)} 个键:")
        for key, value in data.items():
            value_type = type(value).__name__
            if isinstance(value, (dict, list)) and indent < 2:
                lines.append(f"{prefix}  - `{key}` ({value_type}):")
                lines.append(_analyze_json_structure(value, indent + 2))
            else:
                if isinstance(value, str) and len(value) > 50:
                    preview = value[:50] + "..."
                elif isinstance(value, (int, float, bool, type(None))):
                    preview = str(value)
                else:
                    preview = str(value)[:50]
                lines.append(f"{prefix}  - `{key}` ({value_type}): {preview}")
    elif isinstance(data, list):
        lines.append(f"{prefix}- 数组，包含 {len(data)} 个元素:")
        if data:
            first_item = data[0]
            if isinstance(first_item, (dict, list)) and indent < 2:
                lines.append(f"{prefix}  - 第一个元素的结构:")
                lines.append(_analyze_json_structure(first_item, indent + 2))
            else:
                lines.append(f"{prefix}  - 元素类型: {type(first_item).__name__}")
    else:
        lines.append(f"{prefix}- 值: {type(data).__name__}")

    return "\n".join(lines)
