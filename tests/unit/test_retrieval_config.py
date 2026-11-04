import pytest

from retrieval.retrieval_config import RetrievalConfig


def test_default_configuration():

    config = RetrievalConfig()

    assert config.default_k == 5
    assert config.max_k == 20


def test_resolve_default_k():

    config = RetrievalConfig()

    result = config.resolve_k()

    assert result == 5


def test_resolve_custom_k():

    config = RetrievalConfig()

    result = config.resolve_k(10)

    assert result == 10


def test_requested_k_exceeding_maximum():

    config = RetrievalConfig()

    with pytest.raises(ValueError, match="cannot exceed maximum K"):

        config.resolve_k(25)


def test_zero_k_raises_error():

    config = RetrievalConfig()

    with pytest.raises(ValueError):

        config.resolve_k(0)


def test_negative_k_raises_error():

    config = RetrievalConfig()

    with pytest.raises(ValueError):

        config.resolve_k(-1)


def test_invalid_k_type_raises_error():

    config = RetrievalConfig()

    with pytest.raises(TypeError):

        config.resolve_k("five")


def test_default_k_cannot_exceed_maximum():

    with pytest.raises(ValueError):

        RetrievalConfig(
            default_k=25,
            max_k=20
        )


def test_configuration_rejects_zero_default():

    with pytest.raises(ValueError):

        RetrievalConfig(default_k=0)


def test_configuration_rejects_zero_maximum():

    with pytest.raises(ValueError):

        RetrievalConfig(max_k=0)