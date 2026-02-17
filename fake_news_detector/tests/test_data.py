"""
Unit tests for TextPreprocessor and DataLoader.

Tests use synthetic data — no file I/O or network access required.
"""
import pytest
import pandas as pd
import numpy as np

from fake_news_detector.data.preprocessor import TextPreprocessor


# ── TextPreprocessor Tests ───────────────────────────────────────────

class TestTextPreprocessor:

    @pytest.fixture
    def preprocessor(self):
        pp = TextPreprocessor()
        pp.fit()
        return pp

    # --- URL Detection ---

    def test_is_url_true(self):
        assert TextPreprocessor.is_url("https://www.example.com/article") is True
        assert TextPreprocessor.is_url("http://example.org") is True

    def test_is_url_false(self):
        assert TextPreprocessor.is_url("this is just some text") is False
        assert TextPreprocessor.is_url("not a url at all") is False
        assert TextPreprocessor.is_url("") is False

    # --- Text Cleaning ---

    def test_removes_urls(self, preprocessor):
        text = "Check out https://example.com for more info"
        result = preprocessor.transform([text])
        assert "https" not in result.iloc[0]
        assert "example.com" not in result.iloc[0]

    def test_removes_html_tags(self, preprocessor):
        text = "<p>Hello <b>world</b></p>"
        result = preprocessor.transform([text])
        assert "<p>" not in result.iloc[0]
        assert "<b>" not in result.iloc[0]

    def test_removes_special_characters(self, preprocessor):
        text = "hello!! world?? @#$% test"
        result = preprocessor.transform([text])
        cleaned = result.iloc[0]
        assert "@" not in cleaned
        assert "#" not in cleaned
        assert "$" not in cleaned

    def test_lowercases_text(self, preprocessor):
        text = "THIS IS UPPERCASE TEXT"
        result = preprocessor.transform([text])
        assert result.iloc[0] == result.iloc[0].lower()

    def test_handles_empty_string(self, preprocessor):
        result = preprocessor.transform([""])
        assert isinstance(result, pd.Series)
        assert len(result) == 1

    # --- Transform Input Types ---

    def test_transform_list_input(self, preprocessor):
        result = preprocessor.transform(["hello world", "test text"])
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_transform_series_input(self, preprocessor):
        series = pd.Series(["hello world", "test text"])
        result = preprocessor.transform(series)
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_transform_dataframe_input(self, preprocessor):
        df = pd.DataFrame({"text": ["hello world", "test text"]})
        result = preprocessor.transform(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    # --- Preprocess DataFrame ---

    def test_preprocess_dataframe_adds_cleaned_column(self, preprocessor):
        df = pd.DataFrame({
            "text": ["Hello World!", "Test article."],
            "label": [0, 1]
        })
        result = preprocessor.preprocess_dataframe(df)
        assert "cleaned_text" in result.columns
        assert len(result) == 2

    def test_preprocess_dataframe_combines_title_and_text(self, preprocessor):
        df = pd.DataFrame({
            "title": ["Breaking News"],
            "text": ["Some article body"],
            "label": [0]
        })
        result = preprocessor.preprocess_dataframe(df)
        cleaned = result["cleaned_text"].iloc[0]
        assert "breaking" in cleaned or "news" in cleaned


# ── DataLoader Tests ─────────────────────────────────────────────────

class TestDataLoader:
    """Test the DataLoader split logic using synthetic DataFrames."""

    def test_train_test_split_sizes(self):
        from fake_news_detector.data.loader import DataLoader
        loader = DataLoader()

        df = pd.DataFrame({
            "text": [f"sample text {i}" for i in range(100)],
            "label": [0] * 50 + [1] * 50
        })

        X_train, X_test, y_train, y_test = loader.get_train_test_split(
            df, test_size=0.2, target_column="label"
        )
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_train_test_split_stratification(self):
        from fake_news_detector.data.loader import DataLoader
        loader = DataLoader()

        df = pd.DataFrame({
            "text": [f"text {i}" for i in range(100)],
            "label": [0] * 50 + [1] * 50
        })

        _, _, _, y_test = loader.get_train_test_split(
            df, test_size=0.2, target_column="label"
        )
        # With 50/50 distribution and stratification, test set should be ~50/50
        counts = y_test.value_counts()
        assert counts[0] == 10
        assert counts[1] == 10

    def test_missing_target_column_raises(self):
        from fake_news_detector.data.loader import DataLoader
        loader = DataLoader()

        df = pd.DataFrame({"text": ["hello"], "other": [1]})
        with pytest.raises(ValueError, match="Target column"):
            loader.get_train_test_split(df, target_column="label")
