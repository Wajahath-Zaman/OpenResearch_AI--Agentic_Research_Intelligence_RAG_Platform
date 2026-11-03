from config.settings import get_settings


def test_settings_load_successfully():

    settings = get_settings()

    assert settings.app_name == (
        "Open Research Intelligence Assistant"
    )

    assert settings.api_prefix == "/api/v1"

    assert settings.embedding_model


def test_llm_api_keys_are_optional():

    settings = get_settings()

    assert (
        settings.groq_api_key is None
        or isinstance(settings.groq_api_key, str)
    )

    assert (
        settings.gemini_api_key is None
        or isinstance(settings.gemini_api_key, str)
    )