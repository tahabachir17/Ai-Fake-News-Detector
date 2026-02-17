"""
Unit tests for NaiveBayesModel and SVMModel.

Uses small synthetic data to keep tests fast and CI-friendly
(no large datasets or network access required).
"""
import os
import pytest
import numpy as np

from fake_news_detector.models.naive_bayes import NaiveBayesModel
from fake_news_detector.models.svm_model import SVMModel


# ── Fixtures ─────────────────────────────────────────────────────────

FAKE_TEXTS = [
    "government conspiracy exposed shocking secret",
    "unbelievable scandal politicians hiding truth",
    "breaking shocking fraud exposed corrupt",
    "secret plot anonymous sources shocking leak",
    "fake hoax misleading propaganda lies",
    "conspiracy theory deep state cover up",
    "media lies propaganda brainwashing exposed",
    "anonymous tip shocking government scandal",
]

REAL_TEXTS = [
    "the president held a press conference today",
    "stock market rose three percent this quarter",
    "scientists published new research on climate",
    "congress passed the annual budget bill today",
    "the federal reserve announced interest rates",
    "local election results announced by officials",
    "new trade agreement signed between countries",
    "the unemployment rate dropped last month",
]

TRAIN_TEXTS = FAKE_TEXTS + REAL_TEXTS
TRAIN_LABELS = [1] * len(FAKE_TEXTS) + [0] * len(REAL_TEXTS)


@pytest.fixture
def trained_nb():
    """Return a NaiveBayesModel trained on synthetic data."""
    model = NaiveBayesModel(ngram_range=(1, 2))
    model.train(TRAIN_TEXTS, TRAIN_LABELS)
    return model


@pytest.fixture
def trained_svm():
    """Return an SVMModel trained on synthetic data."""
    model = SVMModel(ngram_range=(1, 2), max_iter=500)
    model.train(TRAIN_TEXTS, TRAIN_LABELS)
    return model


# ── NaiveBayesModel Tests ────────────────────────────────────────────

class TestNaiveBayesModel:

    def test_init_creates_pipeline(self):
        model = NaiveBayesModel(ngram_range=(1, 2))
        assert model.model is not None
        assert hasattr(model.model, 'fit')
        assert hasattr(model.model, 'predict')

    def test_train_and_predict(self, trained_nb):
        preds = trained_nb.predict(TRAIN_TEXTS)
        assert len(preds) == len(TRAIN_TEXTS)
        # Should fit the training data well
        accuracy = np.mean(np.array(preds) == np.array(TRAIN_LABELS))
        assert accuracy >= 0.7

    def test_predict_proba_shape(self, trained_nb):
        probs = trained_nb.predict_proba(TRAIN_TEXTS)
        assert probs.shape == (len(TRAIN_TEXTS), 2)
        # Each row should sum to ~1
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_evaluate_returns_metrics(self, trained_nb):
        result = trained_nb.evaluate(TRAIN_TEXTS, TRAIN_LABELS)
        assert "accuracy" in result
        assert "classification_report" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_predict_from_input_text(self, trained_nb):
        result = trained_nb.predict_from_input("government scandal conspiracy exposed")
        assert result['input_type'] == 'text'
        assert result['label'] in ['0', '1', 0, 1]
        assert 0.0 <= result['score'] <= 1.0
        assert isinstance(result['probabilities'], dict)

    def test_save_and_load(self, trained_nb, tmp_path):
        filepath = str(tmp_path / "nb_model.pkl")
        trained_nb.save_model(filepath)
        assert os.path.exists(filepath)

        new_model = NaiveBayesModel()
        new_model.load_model(filepath)

        original_preds = trained_nb.predict(TRAIN_TEXTS[:3])
        loaded_preds = new_model.predict(TRAIN_TEXTS[:3])
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_load_nonexistent_raises(self):
        model = NaiveBayesModel()
        with pytest.raises(FileNotFoundError):
            model.load_model("/nonexistent/path/model.pkl")


# ── SVMModel Tests ───────────────────────────────────────────────────

class TestSVMModel:

    def test_init_creates_pipeline(self):
        model = SVMModel(ngram_range=(1, 2))
        assert model.model is not None
        assert hasattr(model.model, 'fit')
        assert hasattr(model.model, 'predict')

    def test_train_and_predict(self, trained_svm):
        preds = trained_svm.predict(TRAIN_TEXTS)
        assert len(preds) == len(TRAIN_TEXTS)
        accuracy = np.mean(np.array(preds) == np.array(TRAIN_LABELS))
        assert accuracy >= 0.7

    def test_predict_proba_shape(self, trained_svm):
        probs = trained_svm.predict_proba(TRAIN_TEXTS)
        assert probs.shape == (len(TRAIN_TEXTS), 2)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_evaluate_returns_metrics(self, trained_svm):
        result = trained_svm.evaluate(TRAIN_TEXTS, TRAIN_LABELS)
        assert "accuracy" in result
        assert "classification_report" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_predict_from_input_text(self, trained_svm):
        result = trained_svm.predict_from_input("government scandal conspiracy exposed")
        assert result['input_type'] == 'text'
        assert result['label'] in ['0', '1', 0, 1]
        assert 0.0 <= result['score'] <= 1.0
        assert isinstance(result['probabilities'], dict)

    def test_save_and_load(self, trained_svm, tmp_path):
        filepath = str(tmp_path / "svm_model.pkl")
        trained_svm.save_model(filepath)
        assert os.path.exists(filepath)

        new_model = SVMModel()
        new_model.load_model(filepath)

        original_preds = trained_svm.predict(TRAIN_TEXTS[:3])
        loaded_preds = new_model.predict(TRAIN_TEXTS[:3])
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_load_nonexistent_raises(self):
        model = SVMModel()
        with pytest.raises(FileNotFoundError):
            model.load_model("/nonexistent/path/model.pkl")
