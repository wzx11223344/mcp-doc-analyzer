"""
summarizer.py - 摘要工具模块

提供抽取式摘要、文本统计、相似句子检测和自动大纲提取工具。
使用 numpy 做矩阵运算，jieba 做中文分词。
"""

import re
import math
from typing import List, Dict, Tuple

import jieba
import numpy as np

from mcp_doc_analyzer.utils import format_table


def extractive_summary(text: str, num_sentences: int = 3) -> str:
    """生成抽取式摘要（基于 TextRank 算法）。

    使用 TextRank 算法计算句子重要性，抽取文本中最重要的句子组成摘要。

    Args:
        text: 要摘要的文本
        num_sentences: 摘要包含的句子数，默认为3

    Returns:
        Markdown 格式的摘要结果

    Examples:
        >>> result = extractive_summary("这是第一句话。这是第二句话。这是第三句话。")
        >>> "## 抽取式摘要" in result
        True
    """
    if not text or not text.strip():
        return "## 抽取式摘要\n\n[错误] 输入文本为空。"

    if num_sentences <= 0:
        num_sentences = 3

    # 分句
    sentences = _split_sentences(text)

    if len(sentences) <= num_sentences:
        # 句子不够多，直接返回全部
        summary = "\n\n".join(sentences)
        result = f"""## 抽取式摘要

{format_table(
    ["属性", "值"],
    [
        ["算法", "TextRank"],
        ["原文句子数", str(len(sentences))],
        ["摘要句子数", str(len(sentences))],
        ["摘要比例", "100%"],
    ]
)}

## 摘要内容

{summary}
"""
        return result

    # 为每个句子分词
    sentence_words = []
    for sent in sentences:
        words = [w.strip().lower() for w in jieba.lcut(sent) if w.strip() and len(w.strip()) > 1]
        sentence_words.append(words)

    # 构建句子相似度矩阵
    n = len(sentences)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            sim = _calculate_sentence_similarity(sentence_words[i], sentence_words[j])
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim

    # TextRank 迭代计算
    scores = _textrank_scores(similarity_matrix, damping=0.85, max_iter=100, tol=1e-5)

    # 按分数排序并选择 top_n 句子
    ranked_indices = np.argsort(scores)[::-1]
    selected_indices = sorted(ranked_indices[:num_sentences])
    selected_sentences = [sentences[i] for i in selected_indices]

    summary = "\n\n".join(selected_sentences)
    summary_ratio = (len(num_sentences) / len(sentences)) * 100 if len(sentences) > 0 else 0

    # 句子得分表
    score_rows = []
    for idx in ranked_indices[:min(10, n)]:
        score_rows.append([str(idx + 1), sentences[idx][:80] + ("..." if len(sentences[idx]) > 80 else ""), f"{scores[idx]:.4f}"])

    result = f"""## 抽取式摘要

{format_table(
    ["属性", "值"],
    [
        ["算法", "TextRank"],
        ["原文句子数", str(len(sentences))],
        ["摘要句子数", str(num_sentences)],
        ["摘要比例", f"{summary_ratio:.1f}%"],
        ["迭代次数", "100"],
        ["阻尼系数", "0.85"],
    ]
)}

## 摘要内容

{summary}

## 句子重要性得分（Top {min(10, n)}）

{format_table(["序号", "句子预览", "TextRank得分"], score_rows)}
"""
    return result


def text_statistics(text: str) -> str:
    """文本统计信息（字数/词数/句数/段落数/阅读时间）。

    综合统计文本的基本指标，包括字符数、词数、句子数、段落数和预计阅读时间。

    Args:
        text: 要统计的文本

    Returns:
        Markdown 格式的文本统计结果

    Examples:
        >>> result = text_statistics("Hello world. This is a test.")
        >>> "## 文本统计" in result
        True
    """
    if not text or not text.strip():
        return "## 文本统计\n\n[错误] 输入文本为空。"

    # 字符统计
    total_chars = len(text)
    total_chars_no_space = len(text.replace(" ", "").replace("\t", "").replace("\n", ""))
    chinese_chars = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)
    english_chars = sum(1 for c in text if (0x0041 <= ord(c) <= 0x005A) or (0x0061 <= ord(c) <= 0x007A))
    digit_chars = sum(1 for c in text if c.isdigit())

    # 词数统计
    words = jieba.lcut(text)
    valid_words = [w for w in words if w.strip() and not re.match(r'^[\W_]+$', w, re.UNICODE)]
    word_count = len(valid_words)

    # 句子统计
    sentences = _split_sentences(text)
    sentence_count = len(sentences)

    # 段落统计
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraph_count = len(paragraphs)

    # 阅读时间估算
    # 中文：约 300-400 字/分钟，英文：约 200-250 词/分钟
    if chinese_chars > english_chars:
        reading_time = max(1, math.ceil(chinese_chars / 350))
        reading_unit = "字/分钟"
        reading_rate = 350
    else:
        reading_time = max(1, math.ceil(word_count / 225))
        reading_unit = "词/分钟"
        reading_rate = 225

    # 平均值
    avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
    avg_chars_per_sentence = total_chars / sentence_count if sentence_count > 0 else 0
    avg_sentence_per_paragraph = sentence_count / paragraph_count if paragraph_count > 0 else 0

    result = f"""## 文本统计

{format_table(
    ["属性", "值"],
    [
        ["总字符数", str(total_chars)],
        ["字符数（不含空格）", str(total_chars_no_space)],
        ["中文字符数", str(chinese_chars)],
        ["英文字符数", str(english_chars)],
        ["数字字符数", str(digit_chars)],
        ["总词数", str(word_count)],
        ["句子数", str(sentence_count)],
        ["段落数", str(paragraph_count)],
        ["预计阅读时间", f"约 {reading_time} 分钟"],
        ["阅读速度基准", f"{reading_rate} {reading_unit}"],
    ]
)}

## 平均值统计

{format_table(
    ["统计项", "值"],
    [
        ["每句平均词数", f"{avg_words_per_sentence:.1f}"],
        ["每句平均字符数", f"{avg_chars_per_sentence:.1f}"],
        ["每段平均句子数", f"{avg_sentence_per_paragraph:.1f}"],
    ]
)}

## 字符类型分布

{format_table(
    ["字符类型", "数量", "占比"],
    [
        ["中文字符", str(chinese_chars), f"{(chinese_chars / max(1, total_chars)) * 100:.1f}%"],
        ["英文字符", str(english_chars), f"{(english_chars / max(1, total_chars)) * 100:.1f}%"],
        ["数字字符", str(digit_chars), f"{(digit_chars / max(1, total_chars)) * 100:.1f}%"],
        ["其他字符", str(total_chars - chinese_chars - english_chars - digit_chars), f"{((total_chars - chinese_chars - english_chars - digit_chars) / max(1, total_chars)) * 100:.1f}%"],
    ]
)}
"""
    return result


def find_similar_sentences(text: str, threshold: float = 0.5) -> str:
    """检测文本中的相似句子。

    计算所有句子对之间的相似度，找出相似度超过阈值的句子对。

    Args:
        text: 要分析的文本
        threshold: 相似度阈值（0-1），默认为0.5

    Returns:
        Markdown 格式的相似句子检测结果

    Examples:
        >>> result = find_similar_sentences("今天是好天气。今天是好天气。明天也不错。")
        >>> "## 相似句子检测" in result
        True
    """
    if not text or not text.strip():
        return "## 相似句子检测\n\n[错误] 输入文本为空。"

    if threshold < 0:
        threshold = 0.0
    if threshold > 1:
        threshold = 1.0

    # 分句
    sentences = _split_sentences(text)

    if len(sentences) < 2:
        return "## 相似句子检测\n\n[提示] 文本中句子不足，至少需要2个句子进行比较。"

    # 为每个句子分词
    sentence_words = []
    for sent in sentences:
        words = [w.strip().lower() for w in jieba.lcut(sent) if w.strip() and len(w.strip()) > 1]
        sentence_words.append(words)

    # 计算所有句子对的相似度
    similar_pairs = []
    n = len(sentences)
    all_scores = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = _calculate_sentence_similarity(sentence_words[i], sentence_words[j])
            all_scores.append((i, j, sim))
            if sim >= threshold:
                similar_pairs.append((i, j, sim))

    # 按相似度降序排列
    similar_pairs.sort(key=lambda x: x[2], reverse=True)

    # 构建相似句子对表格
    pair_rows = []
    for i, j, sim in similar_pairs[:20]:
        sent_i_preview = sentences[i][:100] + ("..." if len(sentences[i]) > 100 else "")
        sent_j_preview = sentences[j][:100] + ("..." if len(sentences[j]) > 100 else "")
        pair_rows.append([
            f"句子 {i + 1} & {j + 1}",
            f"{sim:.4f}",
            sent_i_preview,
            sent_j_preview,
        ])

    # 统计所有句子对的相似度分布
    if all_scores:
        all_sim_values = [s[2] for s in all_scores]
        avg_sim = sum(all_sim_values) / len(all_sim_values)
        max_sim = max(all_sim_values)
        min_sim = min(all_sim_values)

        # 分布统计
        high_count = sum(1 for s in all_sim_values if s >= 0.7)
        medium_count = sum(1 for s in all_sim_values if 0.4 <= s < 0.7)
        low_count = sum(1 for s in all_sim_values if s < 0.4)
    else:
        avg_sim = max_sim = min_sim = 0.0
        high_count = medium_count = low_count = 0

    result = f"""## 相似句子检测

{format_table(
    ["属性", "值"],
    [
        ["句子总数", str(n)],
        ["比较对数", str(len(all_scores))],
        ["相似度阈值", f"{threshold:.2f}"],
        ["相似句子对数", str(len(similar_pairs))],
        ["平均相似度", f"{avg_sim:.4f}"],
        ["最高相似度", f"{max_sim:.4f}"],
        ["最低相似度", f"{min_sim:.4f}"],
    ]
)}

## 相似度分布

{format_table(
    ["相似度范围", "数量", "占比"],
    [
        ["高（≥0.7）", str(high_count), f"{(high_count / max(1, len(all_scores))) * 100:.1f}%"],
        ["中（0.4-0.7）", str(medium_count), f"{(medium_count / max(1, len(all_scores))) * 100:.1f}%"],
        ["低（<0.4）", str(low_count), f"{(low_count / max(1, len(all_scores))) * 100:.1f}%"],
    ]
)}

## 相似句子对（Top {min(20, len(similar_pairs))}）

{format_table(["句子对", "相似度", "句子A", "句子B"], pair_rows if pair_rows else [["（无超过阈值的相似句子对）", "", "", ""]])}
"""
    return result


def text_outline(text: str) -> str:
    """从文本中自动提取大纲。

    基于标题标记（#）、关键词和句子重要性，自动生成文本大纲。

    Args:
        text: 要提取大纲的文本

    Returns:
        Markdown 格式的文本大纲

    Examples:
        >>> result = text_outline("# 标题\\n\\n正文内容。\\n\\n## 子标题")
        >>> "## 文本大纲" in result
        True
    """
    if not text or not text.strip():
        return "## 文本大纲\n\n[错误] 输入文本为空。"

    outline_items = []
    paragraphs = text.split("\n")
    outline_level = 0

    # 检测 Markdown 标题
    has_markdown_headers = False
    for line in paragraphs:
        if re.match(r'^#{1,6}\s+', line.strip()):
            has_markdown_headers = True
            break

    if has_markdown_headers:
        # 从 Markdown 标题提取大纲
        for line in paragraphs:
            stripped = line.strip()
            header_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                outline_items.append((level, title, "标题"))
            elif stripped and not stripped.startswith("#"):
                # 非标题行也可能包含关键信息
                words = jieba.lcut(stripped)
                key_words = [w for w in words if len(w) > 2 and not re.match(r'^[\W_]+$', w, re.UNICODE)][:3]
                if key_words:
                    outline_items.append((3, "、".join(key_words[:2]), "关键词"))
    else:
        # 没有标题标记，从内容自动提取
        # 按段落分割
        text_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if len(text_paragraphs) == 1:
            # 单段落：按句子提取要点
            sentences = _split_sentences(text)
            if len(sentences) <= 1:
                # 如果只有一个句子，提取关键词
                import jieba.analyse
                keywords = jieba.analyse.extract_tags(text, topK=5)
                for i, kw in enumerate(keywords):
                    outline_items.append((1, kw, "关键词"))
            else:
                # 多句子：用 TextRank 选出重要句子作为要点
                sentence_words = []
                for sent in sentences:
                    words = [w.strip().lower() for w in jieba.lcut(sent) if w.strip() and len(w.strip()) > 1]
                    sentence_words.append(words)

                n = len(sentences)
                if n > 1:
                    similarity_matrix = np.zeros((n, n))
                    for i in range(n):
                        for j in range(i + 1, n):
                            sim = _calculate_sentence_similarity(sentence_words[i], sentence_words[j])
                            similarity_matrix[i][j] = sim
                            similarity_matrix[j][i] = sim

                    scores = _textrank_scores(similarity_matrix, damping=0.85, max_iter=50, tol=1e-4)
                    ranked = np.argsort(scores)[::-1]

                    top_count = min(5, n)
                    selected = sorted(ranked[:top_count])
                    for idx in selected:
                        sent_preview = sentences[idx][:100] + ("..." if len(sentences[idx]) > 100 else "")
                        outline_items.append((2, sent_preview, "要点"))
        else:
            # 多段落：每段提取首句或关键词作为要点
            for para in text_paragraphs:
                # 取首句
                first_sentence_match = re.match(r'^(.+?[.!?。！？])', para)
                if first_sentence_match:
                    first_sentence = first_sentence_match.group(1).strip()
                    if len(first_sentence) > 100:
                        first_sentence = first_sentence[:100] + "..."
                    outline_items.append((2, first_sentence, "段落首句"))
                else:
                    # 提取关键词
                    import jieba.analyse
                    keywords = jieba.analyse.extract_tags(para, topK=2)
                    if keywords:
                        outline_items.append((2, "、".join(keywords), "关键词"))

    # 构建大纲 Markdown
    outline_md = []
    for level, title, source in outline_items:
        indent = "  " * (level - 1)
        outline_md.append(f"{indent}- {title} _（{source}）_")

    outline_text = "\n".join(outline_md) if outline_md else "（无法提取大纲）"

    # 统计大纲信息
    level_counts = {}
    for level, _, _ in outline_items:
        level_counts[f"层级 {level}"] = level_counts.get(f"层级 {level}", 0) + 1

    source_counts = {}
    for _, _, source in outline_items:
        source_counts[source] = source_counts.get(source, 0) + 1

    level_rows = [[k, str(v)] for k, v in sorted(level_counts.items())]
    source_rows = [[k, str(v)] for k, v in sorted(source_counts.items())]

    result = f"""## 文本大纲

{format_table(
    ["属性", "值"],
    [
        ["大纲条目数", str(len(outline_items))],
        ["检测到标题", "是" if has_markdown_headers else "否"],
        ["提取方式", "标题解析" if has_markdown_headers else "自动提取（TextRank + 关键词）"],
    ]
)}

## 层级分布

{format_table(["层级", "数量"], level_rows if level_rows else [["（无）", ""]])}

## 来源分布

{format_table(["来源类型", "数量"], source_rows if source_rows else [["（无）", ""]])}

## 大纲内容

{outline_text}
"""
    return result


def _split_sentences(text: str) -> List[str]:
    """将文本分割为句子（内部使用）。

    Args:
        text: 原始文本

    Returns:
        句子列表
    """
    # 匹配中英文句子结束标点
    pattern = r'(?<=[.!?。！？；;])\s+'
    sentences = re.split(pattern, text)
    # 过滤空句子和过短句子
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]
    return sentences


def _calculate_sentence_similarity(words1: List[str], words2: List[str]) -> float:
    """计算两个句子之间的相似度（基于词重叠，内部使用）。

    使用改进的 Jaccard 相似度，考虑词频。

    Args:
        words1: 句子1的词列表
        words2: 句子2的词列表

    Returns:
        相似度值（0-1）
    """
    if not words1 or not words2:
        return 0.0

    set1 = set(words1)
    set2 = set(words2)

    # Jaccard 相似度
    intersection = set1 & set2
    union = set1 | set2

    if not union:
        return 0.0

    jaccard_sim = len(intersection) / len(union)

    # 考虑重叠比例（双向）
    overlap_1 = len(intersection) / len(set1) if set1 else 0
    overlap_2 = len(intersection) / len(set2) if set2 else 0

    # 综合相似度
    similarity = jaccard_sim * 0.5 + (overlap_1 + overlap_2) / 2 * 0.5

    return similarity


def _textrank_scores(similarity_matrix: np.ndarray, damping: float = 0.85,
                     max_iter: int = 100, tol: float = 1e-5) -> np.ndarray:
    """计算 TextRank 分数（内部使用）。

    使用 PageRank 算法迭代计算每个句子的 TextRank 分数。

    Args:
        similarity_matrix: 句子相似度矩阵
        damping: 阻尼系数，默认0.85
        max_iter: 最大迭代次数
        tol: 收敛容差

    Returns:
        每个句子的 TextRank 分数数组
    """
    n = similarity_matrix.shape[0]

    if n == 0:
        return np.array([])

    if n == 1:
        return np.array([1.0])

    # 归一化相似度矩阵（按行归一化）
    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    # 避免除以0
    row_sums[row_sums == 0] = 1.0
    normalized_matrix = similarity_matrix / row_sums

    # 初始化分数
    scores = np.ones(n) / n

    # 迭代计算
    for _ in range(max_iter):
        new_scores = (1 - damping) / n + damping * normalized_matrix.T.dot(scores)
        if np.abs(new_scores - scores).sum() < tol:
            scores = new_scores
            break
        scores = new_scores

    # 归一化分数
    total = scores.sum()
    if total > 0:
        scores = scores / total

    return scores
