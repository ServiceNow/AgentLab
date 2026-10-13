import builtins
import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentlab.llm import llm_utils


@pytest.fixture(autouse=True)
def clear_tokenizer_cache():
    llm_utils.get_tokenizer.cache_clear()
    yield
    llm_utils.get_tokenizer.cache_clear()


@pytest.fixture
def tokenizer_loaders(monkeypatch):
    fallback = Mock()
    fallback.encode.return_value = [1, 2, 3]

    def encoding_for_model(model_name):
        if model_name == "gpt-4":
            return fallback
        raise KeyError(model_name)

    monkeypatch.setattr(llm_utils.tiktoken, "encoding_for_model", encoding_for_model)
    hf_loader = Mock(side_effect=OSError("tokenizer unavailable"))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=hf_loader)),
    )
    return fallback, hf_loader


def test_count_tokens_warns_when_using_fallback(tokenizer_loaders, caplog):
    fallback, hf_loader = tokenizer_loaders
    model = "anthropic/unavailable-model"
    with caplog.at_level(logging.WARNING):
        assert llm_utils.count_tokens("Hello", model=model) == 3

    fallback.encode.assert_called_once_with("Hello")
    hf_loader.assert_called_once_with(model)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert model in caplog.text
    assert "Defaulting to gpt-4" in caplog.text


def test_missing_transformers_warns_and_keeps_fallback(tokenizer_loaders, monkeypatch, caplog):
    fallback, _ = tokenizer_loaders
    original_import = builtins.__import__

    def import_without_transformers(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("transformers is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_transformers)
    with caplog.at_level(logging.WARNING):
        assert llm_utils.get_tokenizer("missing-model") is fallback

    assert "missing-model" in caplog.text
    assert "Defaulting to gpt-4" in caplog.text


def test_cached_fallback_warns_once(tokenizer_loaders, caplog):
    fallback, hf_loader = tokenizer_loaders
    with caplog.at_level(logging.WARNING):
        assert llm_utils.get_tokenizer("missing-model") is fallback
        assert llm_utils.get_tokenizer("missing-model") is fallback

    hf_loader.assert_called_once_with("missing-model")
    assert len(caplog.records) == 1


def test_known_tiktoken_model_does_not_warn(tokenizer_loaders, caplog):
    fallback, hf_loader = tokenizer_loaders
    with caplog.at_level(logging.WARNING):
        assert llm_utils.get_tokenizer("gpt-4") is fallback

    hf_loader.assert_not_called()
    assert not caplog.records


def test_successful_huggingface_tokenizer_does_not_warn(tokenizer_loaders, caplog):
    _, hf_loader = tokenizer_loaders
    tokenizer = Mock()
    hf_loader.side_effect = None
    hf_loader.return_value = tokenizer
    with caplog.at_level(logging.WARNING):
        assert llm_utils.get_tokenizer("organization/model") is tokenizer

    hf_loader.assert_called_once_with("organization/model")
    assert not caplog.records
