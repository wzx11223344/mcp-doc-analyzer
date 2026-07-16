"""Tests for mcp-doc-analyzer functions."""

import pytest
from mcp_doc_analyzer.analyzer import language_detect, sentence_analysis
from mcp_doc_analyzer.summarizer import text_statistics


class TestLanguageDetect:
    """Tests for language_detect function."""

    def test_chinese_text(self):
        """Test detecting Chinese text."""
        result = language_detect("这是一段中文文本用于测试语言检测功能")
        assert "中文" in result or "zh" in result.lower()

    def test_english_text(self):
        """Test detecting English text."""
        result = language_detect("This is an English text for language detection testing")
        assert "English" in result or "英文" in result or "en" in result.lower()

    def test_mixed_text(self):
        """Test detecting mixed language text."""
        result = language_detect("This is English 这是中文")
        assert "混合" in result or "Mixed" in result or "both" in result.lower()

    def test_empty_text(self):
        """Test empty text handling."""
        result = language_detect("")
        assert isinstance(result, str)

    def test_numbers_only(self):
        """Test text with only numbers."""
        result = language_detect("12345 67890")
        assert isinstance(result, str)


class TestSentenceAnalysis:
    """Tests for sentence_analysis function."""

    def test_simple_text(self):
        """Test sentence analysis on simple text."""
        result = sentence_analysis("这是第一句。这是第二句。这是第三句。")
        assert "句子分析" in result or "句子" in result
        assert isinstance(result, str)

    def test_single_sentence(self):
        """Test analysis on a single sentence."""
        result = sentence_analysis("Hello world.")
        assert isinstance(result, str)

    def test_empty_text(self):
        """Test analysis on empty text."""
        result = sentence_analysis("")
        assert isinstance(result, str)

    def test_paragraph_text(self):
        """Test analysis on a paragraph."""
        result = sentence_analysis(
            "First sentence about data. Second sentence with more details. "
            "Third one is also here. And a fourth one."
        )
        assert isinstance(result, str)


class TestTextStatistics:
    """Tests for text_statistics function."""

    def test_basic_statistics(self):
        """Test basic text statistics."""
        result = text_statistics("这是一个测试文本，用于验证统计功能。")
        assert isinstance(result, str)
        assert "字符" in result or "字" in result or "字符数" in result

    def test_long_text(self):
        """Test statistics on longer text."""
        result = text_statistics(
            "This is a longer piece of text that should have multiple sentences. "
            "It is used to test the text statistics functionality. "
            "The statistics should include character count, word count, "
            "sentence count, and estimated reading time."
        )
        assert isinstance(result, str)

    def test_empty_text(self):
        """Test statistics on empty text."""
        result = text_statistics("")
        assert isinstance(result, str)


class TestWordFrequency:
    """Tests for word_frequency function."""

    def test_word_frequency_output(self):
        """Test that word_frequency returns markdown output."""
        from mcp_doc_analyzer.analyzer import word_frequency
        result = word_frequency(
            "自然语言处理是人工智能的一个重要分支。"
            "自然语言处理技术正在快速发展。"
        )
        assert isinstance(result, str)
        assert "词频" in result

    def test_empty_text(self):
        """Test word_frequency with empty text."""
        from mcp_doc_analyzer.analyzer import word_frequency
        result = word_frequency("")
        assert isinstance(result, str)
        assert "错误" in result or "空" in result

    def test_top_n_parameter(self):
        """Test word_frequency with custom top_n."""
        from mcp_doc_analyzer.analyzer import word_frequency
        result = word_frequency("数据 分析 统计 数据 挖掘 分析 学习 机器 数据", top_n=3)
        assert isinstance(result, str)
