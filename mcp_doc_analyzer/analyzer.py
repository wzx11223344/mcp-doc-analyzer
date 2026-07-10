"""
analyzer.py - 文本分析工具模块

提供词频统计、关键词提取、句子分析、可读性评分、
情感分析和语言检测等文本分析工具。
使用 jieba 做中文分词，sklearn 做 TF-IDF。
"""

import re
import math
from collections import Counter
from typing import List, Tuple, Dict

import jieba
import jieba.analyse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from mcp_doc_analyzer.utils import format_table


def word_frequency(text: str, top_n: int = 20) -> str:
    """统计文本中的词频（支持中英文分词）。

    使用 jieba 进行中文分词，对英文按空格分割，
    统计并返回出现频率最高的词。

    Args:
        text: 要分析的文本
        top_n: 返回前 N 个高频词，默认为20

    Returns:
        Markdown 格式的词频统计结果

    Examples:
        >>> result = word_frequency("自然语言处理 自然语言 人工智能")
        >>> "## 词频统计" in result
        True
    """
    if not text or not text.strip():
        return "## 词频统计\n\n[错误] 输入文本为空。"

    if top_n <= 0:
        top_n = 20

    # 使用 jieba 分词
    words = jieba.lcut(text)

    # 过滤：去除标点符号、空白和单字符（中文保留双字符以上）
    filtered_words = []
    for word in words:
        word = word.strip()
        if not word:
            continue
        # 跳过纯标点
        if re.match(r'^[\W_]+$', word, re.UNICODE):
            continue
        # 中文词至少2个字符，英文词至少2个字符
        if len(word) < 2:
            continue
        filtered_words.append(word.lower())

    if not filtered_words:
        return "## 词频统计\n\n[提示] 未找到有效词汇。"

    # 统计词频
    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(top_n)
    total_words = len(filtered_words)

    # 构建表格数据
    rows = []
    for rank, (word, count) in enumerate(top_words, 1):
        percentage = (count / total_words) * 100
        rows.append([str(rank), word, str(count), f"{percentage:.2f}%"])

    result = f"""## 词频统计

{format_table(
    ["属性", "值"],
    [
        ["总词数", str(total_words)],
        ["独立词数", str(len(word_counts))],
        ["返回数量", str(len(top_words))],
    ]
)}

## 高频词排行（前 {len(top_words)} 名）

{format_table(["排名", "词语", "出现次数", "占比"], rows)}
"""
    return result


def extract_keywords(text: str, top_n: int = 10) -> str:
    """从文本中提取关键词（基于 TF-IDF 算法）。

    使用 jieba 的 TF-IDF 关键词提取算法，结合 sklearn 的 TfidfVectorizer，
    提取文本中最具代表性的关键词。

    Args:
        text: 要分析的文本
        top_n: 返回前 N 个关键词，默认为10

    Returns:
        Markdown 格式的关键词提取结果

    Examples:
        >>> result = extract_keywords("人工智能 机器学习 深度学习")
        >>> "## 关键词提取" in result
        True
    """
    if not text or not text.strip():
        return "## 关键词提取\n\n[错误] 输入文本为空。"

    if top_n <= 0:
        top_n = 10

    # 方法1：使用 jieba.analyse 的 TF-IDF 提取
    jieba_keywords = jieba.analyse.extract_tags(
        text, topK=top_n, withWeight=True
    )

    # 方法2：使用 sklearn TfidfVectorizer（按句子分词后计算）
    tfidf_keywords = []
    try:
        sentences = _split_text_to_sentences(text)
        if len(sentences) >= 1:
            # 对每个句子用 jieba 分词后重新拼接
            tokenized_sentences = [" ".join(jieba.lcut(s)) for s in sentences]
            vectorizer = TfidfVectorizer(max_features=top_n * 2)
            tfidf_matrix = vectorizer.fit_transform(tokenized_sentences)
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray().sum(axis=0)
            # 取 top_n
            sorted_indices = np.argsort(scores)[::-1][:top_n]
            tfidf_keywords = [
                (feature_names[i], float(scores[i]))
                for i in sorted_indices
                if scores[i] > 0
            ]
    except Exception:
        pass

    # 合并结果，优先用 jieba 的结果
    rows = []
    for rank, (word, weight) in enumerate(jieba_keywords, 1):
        rows.append([str(rank), word, f"{weight:.4f}"])

    # 构建 sklearn 补充结果
    sklearn_rows = []
    for word, score in tfidf_keywords[:top_n]:
        sklearn_rows.append([word, f"{score:.4f}"])

    result = f"""## 关键词提取

{format_table(
    ["属性", "值"],
    [
        ["提取算法", "TF-IDF (jieba + sklearn)"],
        ["返回数量", str(len(jieba_keywords))],
    ]
)}

## 关键词列表（jieba TF-IDF）

{format_table(["排名", "关键词", "权重"], rows if rows else [["（无）", "", ""]])}

## TF-IDF 得分（sklearn 补充）

{format_table(["关键词", "TF-IDF 得分"], sklearn_rows if sklearn_rows else [["（无）", ""]])}
"""
    return result


def sentence_analysis(text: str) -> str:
    """分析文本的句子结构。

    统计句子数量、平均长度、最长句和最短句等信息。

    Args:
        text: 要分析的文本

    Returns:
        Markdown 格式的句子分析结果

    Examples:
        >>> result = sentence_analysis("Hello world. This is a test.")
        >>> "## 句子分析" in result
        True
    """
    if not text or not text.strip():
        return "## 句子分析\n\n[错误] 输入文本为空。"

    # 分句
    sentences = _split_text_to_sentences(text)

    if not sentences:
        return "## 句子分析\n\n[提示] 未检测到有效句子。"

    # 计算每个句子的长度
    sentence_lengths = [len(s.strip()) for s in sentences]
    avg_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

    max_idx = sentence_lengths.index(max(sentence_lengths))
    min_idx = sentence_lengths.index(min(sentence_lengths))

    longest_sentence = sentences[max_idx].strip()
    shortest_sentence = sentences[min_idx].strip()

    # 截断过长的句子用于展示
    display_longest = longest_sentence[:200] + ("..." if len(longest_sentence) > 200 else "")
    display_shortest = shortest_sentence[:200] + ("..." if len(shortest_sentence) > 200 else "")

    result = f"""## 句子分析

{format_table(
    ["属性", "值"],
    [
        ["句子总数", str(len(sentences))],
        ["平均句长", f"{avg_length:.1f} 字符"],
        ["最长句长度", f"{max(sentence_lengths)} 字符"],
        ["最短句长度", f"{min(sentence_lengths)} 字符"],
        ["总字符数", str(len(text))],
    ]
)}

## 最长句子

> {display_longest}

## 最短句子

> {display_shortest}

## 句子长度分布

{format_table(
    ["序号", "长度（字符）", "预览"],
    [
        [str(i + 1), str(length), s.strip()[:80] + ("..." if len(s.strip()) > 80 else "")]
        for i, (s, length) in enumerate(zip(sentences, sentence_lengths))
    ]
)}
"""
    return result


def readability_score(text: str) -> str:
    """计算文本的可读性评分（基于 Flesch-Kincaid 的改编版本）。

    对中文文本使用改编的评分算法，对英文文本使用标准 Flesch-Kincaid。
    评分越高表示越容易阅读。

    Args:
        text: 要分析的文本

    Returns:
        Markdown 格式的可读性评分结果

    Examples:
        >>> result = readability_score("This is a simple sentence.")
        >>> "## 可读性评分" in result
        True
    """
    if not text or not text.strip():
        return "## 可读性评分\n\n[错误] 输入文本为空。"

    # 分句
    sentences = _split_text_to_sentences(text)
    num_sentences = len(sentences)

    if num_sentences == 0:
        return "## 可读性评分\n\n[错误] 无法分析句子结构。"

    # 分词
    words = jieba.lcut(text)
    # 过滤有效词
    valid_words = [w for w in words if w.strip() and not re.match(r'^[\W_]+$', w, re.UNICODE)]
    num_words = len(valid_words)

    if num_words == 0:
        return "## 可读性评分\n\n[错误] 无法提取有效词汇。"

    # 计算平均词长（字符）
    total_chars = sum(len(w) for w in valid_words)
    avg_word_length = total_chars / num_words

    # 计算平均句长（词数）
    avg_sentence_length = num_words / num_sentences

    # 检测语言
    lang = _detect_language_code(text)

    if lang == "english":
        # 英文：使用标准 Flesch Reading Ease
        # 统计音节（简化：按元音数）
        total_syllables = sum(_count_syllables(w) for w in valid_words)
        avg_syllables_per_word = total_syllables / num_words

        flesch_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        flesch_score = max(0, min(100, flesch_score))

        # Flesch-Kincaid Grade Level
        grade_level = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59

        # 确定难度等级
        if flesch_score >= 90:
            level = "非常容易（5年级）"
        elif flesch_score >= 80:
            level = "容易（6年级）"
        elif flesch_score >= 70:
            level = "较容易（7年级）"
        elif flesch_score >= 60:
            level = "标准难度（8-9年级）"
        elif flesch_score >= 50:
            level = "较难（10-12年级）"
        elif flesch_score >= 30:
            level = "困难（大学水平）"
        else:
            level = "非常困难（研究生水平）"

        result = f"""## 可读性评分

{format_table(
    ["属性", "值"],
    [
        ["语言", "英语"],
        ["算法", "Flesch Reading Ease"],
        ["总词数", str(num_words)],
        ["句子数", str(num_sentences)],
        ["平均句长（词）", f"{avg_sentence_length:.1f}"],
        ["平均词长（字符）", f"{avg_word_length:.1f}"],
        ["平均音节/词", f"{avg_syllables_per_word:.2f}"],
        ["Flesch 得分", f"{flesch_score:.1f} / 100"],
        ["年级水平", f"{grade_level:.1f}"],
        ["难度等级", level],
    ]
)}
"""
    else:
        # 中文：使用改编的可读性评分
        # 基于句长和词频的简化评分
        avg_sentence_char = len(text) / num_sentences

        # 中文可读性公式（改编）
        # 得分范围 0-100，越高越容易阅读
        chinese_score = 100 - (avg_sentence_length - 10) * 2 - (avg_word_length - 2) * 5
        chinese_score = max(0, min(100, chinese_score))

        if chinese_score >= 80:
            level = "非常容易"
        elif chinese_score >= 60:
            level = "较容易"
        elif chinese_score >= 40:
            level = "中等难度"
        elif chinese_score >= 20:
            level = "较难"
        else:
            level = "非常困难"

        result = f"""## 可读性评分

{format_table(
    ["属性", "值"],
    [
        ["语言", "中文"],
        ["算法", "改编 Flesch-Kincaid"],
        ["总词数", str(num_words)],
        ["句子数", str(num_sentences)],
        ["平均句长（词）", f"{avg_sentence_length:.1f}"],
        ["平均句长（字符）", f"{avg_sentence_char:.1f}"],
        ["平均词长（字符）", f"{avg_word_length:.1f}"],
        ["可读性得分", f"{chinese_score:.1f} / 100"],
        ["难度等级", level],
    ]
)}
"""
    return result


def sentiment_analysis(text: str) -> str:
    """简单的基于词典的情感分析。

    使用预定义的正负面情感词典，统计文本中的正负面词汇，
    返回情感倾向判断（正面/负面/中性）。

    Args:
        text: 要分析的文本

    Returns:
        Markdown 格式的情感分析结果

    Examples:
        >>> result = sentiment_analysis("今天天气很好，非常开心！")
        >>> "## 情感分析" in result
        True
    """
    if not text or not text.strip():
        return "## 情感分析\n\n[错误] 输入文本为空。"

    # 正面情感词典
    positive_words = {
        "好", "棒", "优秀", "出色", "完美", "精彩", "喜欢", "爱", "开心", "快乐",
        "幸福", "满意", "成功", "胜利", "赞", "美", "佳", "优", "强", "厉害",
        "感谢", "谢谢", "支持", "推荐", "值得", "受益", "有效", "高效", "便捷",
        "创新", "突破", "进步", "提升", "增长", "繁荣", "兴旺", "顺利", "满意",
        "good", "great", "excellent", "amazing", "wonderful", "perfect", "love",
        "like", "happy", "best", "awesome", "fantastic", "brilliant", "success",
        "positive", "benefit", "win", "recommend", "enjoy", "beautiful", "nice",
    }

    # 负面情感词典
    negative_words = {
        "坏", "差", "糟", "差劲", "糟糕", "失败", "错误", "问题", "困难", "难",
        "讨厌", "恨", "生气", "愤怒", "悲伤", "伤心", "痛苦", "失望", "担心",
        "害怕", "恐惧", "焦虑", "压力", "累", "疲惫", "无聊", "烦", "怒", "忧",
        "跌", "降", "少", "缺", "损", "伤", "病", "死", "危", "险",
        "bad", "worst", "terrible", "awful", "hate", "dislike", "sad", "angry",
        "fail", "failure", "error", "problem", "difficult", "hard", "wrong",
        "negative", "poor", "boring", "annoying", "frustrated", "worried", "scared",
        "disappointed", "broken", "crash", "bug", "issue", "pain", "sick",
    }

    # 分词
    words = jieba.lcut(text)
    words_lower = [w.lower().strip() for w in words if w.strip()]

    # 统计正面词
    found_positive = []
    for word in words_lower:
        if word in positive_words:
            found_positive.append(word)

    # 统计负面词
    found_negative = []
    for word in words_lower:
        if word in negative_words:
            found_negative.append(word)

    pos_count = len(found_positive)
    neg_count = len(found_negative)
    total_sentiment_words = pos_count + neg_count

    # 判断情感倾向
    if total_sentiment_words == 0:
        sentiment = "中性"
        sentiment_score = 0.0
    else:
        sentiment_score = (pos_count - neg_count) / total_sentiment_words
        if sentiment_score > 0.1:
            sentiment = "正面"
        elif sentiment_score < -0.1:
            sentiment = "负面"
        else:
            sentiment = "中性"

    # 情感强度
    if total_sentiment_words == 0:
        intensity = "无"
    elif abs(sentiment_score) > 0.5:
        intensity = "强烈"
    elif abs(sentiment_score) > 0.2:
        intensity = "中等"
    else:
        intensity = "轻微"

    # 正面/负面词频统计
    pos_counter = Counter(found_positive)
    neg_counter = Counter(found_negative)

    pos_rows = [[word, str(count)] for word, count in pos_counter.most_common(10)]
    neg_rows = [[word, str(count)] for word, count in neg_counter.most_common(10)]

    result = f"""## 情感分析

{format_table(
    ["属性", "值"],
    [
        ["总词数", str(len(words_lower))],
        ["正面情感词数", str(pos_count)],
        ["负面情感词数", str(neg_count)],
        ["情感词总数", str(total_sentiment_words)],
        ["情感得分", f"{sentiment_score:.4f}（范围 -1 到 1）"],
        ["情感倾向", sentiment],
        ["情感强度", intensity],
    ]
)}

## 正面情感词

{format_table(["词语", "出现次数"], pos_rows if pos_rows else [["（无）", ""]])}

## 负面情感词

{format_table(["词语", "出现次数"], neg_rows if neg_rows else [["（无）", ""]])}
"""
    return result


def language_detect(text: str) -> str:
    """检测文本的语言（中/英/日/韩/混合）。

    基于字符 Unicode 范围检测，判断文本的主要语言。

    Args:
        text: 要检测的文本

    Returns:
        Markdown 格式的语言检测结果

    Examples:
        >>> result = language_detect("Hello 世界")
        >>> "## 语言检测" in result
        True
    """
    if not text or not text.strip():
        return "## 语言检测\n\n[错误] 输入文本为空。"

    # 统计各语言字符数
    chinese_count = 0
    english_count = 0
    japanese_count = 0
    korean_count = 0
    digit_count = 0
    space_count = 0
    punctuation_count = 0
    other_count = 0

    for char in text:
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF:
            # CJK 统一汉字
            chinese_count += 1
        elif 0x3040 <= code <= 0x309F:
            # 平假名
            japanese_count += 1
        elif 0x30A0 <= code <= 0x30FF:
            # 片假名
            japanese_count += 1
        elif 0xAC00 <= code <= 0xD7AF:
            # 韩文音节
            korean_count += 1
        elif 0x1100 <= code <= 0x11FF:
            # 韩文字母
            korean_count += 1
        elif 0x0041 <= code <= 0x005A or 0x0061 <= code <= 0x007A:
            # 英文字母
            english_count += 1
        elif 0x0030 <= code <= 0x0039:
            # 数字
            digit_count += 1
        elif char.isspace():
            space_count += 1
        elif 0x2000 <= code <= 0x206F or 0x3000 <= code <= 0x303F or 0xFF00 <= code <= 0xFFEF:
            # 标点符号
            punctuation_count += 1
        else:
            other_count += 1

    total_chars = len(text)
    total_meaningful = chinese_count + english_count + japanese_count + korean_count

    # 判断主语言
    if total_meaningful == 0:
        primary_language = "未知"
    else:
        lang_counts = {
            "中文": chinese_count,
            "英语": english_count,
            "日语": japanese_count,
            "韩语": korean_count,
        }
        primary = max(lang_counts, key=lang_counts.get)

        # 判断是否为混合
        sorted_counts = sorted(lang_counts.values(), reverse=True)
        if sorted_counts[0] > 0 and sorted_counts[1] > 0:
            second_ratio = sorted_counts[1] / total_meaningful
            if second_ratio > 0.2:
                primary_language = f"混合（主要: {primary}）"
            else:
                primary_language = primary
        else:
            primary_language = primary

    # 计算各语言占比
    def calc_ratio(count):
        if total_meaningful == 0:
            return 0.0
        return (count / total_meaningful) * 100

    result = f"""## 语言检测

{format_table(
    ["属性", "值"],
    [
        ["总字符数", str(total_chars)],
        ["主要语言", primary_language],
        ["有意义字符数", str(total_meaningful)],
    ]
)}

## 字符分布

{format_table(
    ["字符类型", "数量", "占比"],
    [
        ["中文字符", str(chinese_count), f"{calc_ratio(chinese_count):.1f}%"],
        ["英文字符", str(english_count), f"{calc_ratio(english_count):.1f}%"],
        ["日文字符", str(japanese_count), f"{calc_ratio(japanese_count):.1f}%"],
        ["韩文字符", str(korean_count), f"{calc_ratio(korean_count):.1f}%"],
        ["数字", str(digit_count), "-"],
        ["空格", str(space_count), "-"],
        ["标点符号", str(punctuation_count), "-"],
        ["其他", str(other_count), "-"],
    ]
)}
"""
    return result


def _split_text_to_sentences(text: str) -> List[str]:
    """将文本分割为句子（内部使用）。

    支持中英文句子结束标点。

    Args:
        text: 原始文本

    Returns:
        句子列表
    """
    # 匹配中英文句子结束标点
    pattern = r'(?<=[.!?。！？；;])\s+'
    sentences = re.split(pattern, text)
    # 过滤空句子
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def _count_syllables(word: str) -> int:
    """计算英文单词的音节数（简化版本）。

    Args:
        word: 英文单词

    Returns:
        音节数
    """
    word = word.lower().strip()
    if not word:
        return 0

    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel

    # 至少返回1个音节
    if count == 0:
        count = 1

    # 处理结尾的 'e'
    if word.endswith("e") and count > 1:
        count -= 1

    return count


def _detect_language_code(text: str) -> str:
    """检测文本语言代码（内部使用）。

    Args:
        text: 要检测的文本

    Returns:
        语言代码字符串："chinese", "english", "japanese", "korean", "mixed"
    """
    chinese_count = 0
    english_count = 0
    japanese_count = 0
    korean_count = 0

    for char in text:
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF:
            chinese_count += 1
        elif 0x3040 <= code <= 0x309F or 0x30A0 <= code <= 0x30FF:
            japanese_count += 1
        elif 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
            korean_count += 1
        elif 0x0041 <= code <= 0x005A or 0x0061 <= code <= 0x007A:
            english_count += 1

    counts = {
        "chinese": chinese_count,
        "english": english_count,
        "japanese": japanese_count,
        "korean": korean_count,
    }
    primary = max(counts, key=counts.get)

    if counts[primary] == 0:
        return "unknown"

    # 检查是否混合
    sorted_values = sorted(counts.values(), reverse=True)
    if sorted_values[1] > 0 and sorted_values[1] / max(1, sorted_values[0]) > 0.2:
        return "mixed"

    return primary
