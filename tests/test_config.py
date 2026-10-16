"""Settings: the model choice and secrets from the environment (.env), the rest from agent.toml."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from ticket_agent.config import ModelChoice, Settings

REPO_CONFIG = Path(__file__).resolve().parents[1] / "agent.toml"


def use_config(monkeypatch, tmp_path, text: str) -> None:
    path = tmp_path / "agent.toml"
    path.write_text(text)
    monkeypatch.setenv("AGENT_CONFIG", str(path))


def test_the_committed_agent_toml_is_valid(monkeypatch):
    monkeypatch.setenv("AGENT_CONFIG", str(REPO_CONFIG))
    settings = Settings()
    assert settings.llm.timeout_s == 60
    assert settings.agent.max_steps == 6 and settings.ticketing.url == "http://localhost:8000"


def test_the_file_sets_values_and_the_environment_overrides_one_key(monkeypatch, tmp_path):
    use_config(monkeypatch, tmp_path, "[llm]\ntimeout_s = 30\n[agent]\nmax_steps = 4\n")
    monkeypatch.setenv("TICKETING__URL", "http://ticketing:8000")  # what compose does
    settings = Settings()
    assert (settings.llm.timeout_s, settings.agent.max_steps) == (30, 4)
    assert settings.ticketing.url == "http://ticketing:8000"


def test_a_missing_config_file_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CONFIG", str(tmp_path / "nope.toml"))
    with pytest.raises(FileNotFoundError, match="nope.toml"):
        Settings()


def test_secrets_are_refused_in_the_config_file(monkeypatch, tmp_path):
    use_config(monkeypatch, tmp_path, 'gemini_api_key = "AIza-leaked"\n[llm]\ntimeout_s = 30\n')
    with pytest.raises(ValueError, match=r"\.env"):
        Settings()


def test_a_typo_in_the_config_file_fails_at_startup(monkeypatch, tmp_path):
    use_config(monkeypatch, tmp_path, "[llm]\ntimout_s = 30\n")
    with pytest.raises(ValidationError, match="timout_s"):
        Settings()


def test_the_model_choice_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-6-luna")
    monkeypatch.setenv("LLM_REASONING", "medium")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", " openai/gpt-4o-mini , gemini/gemini-3.5-flash@low ")
    settings = Settings()
    assert settings.main_model == ModelChoice(
        provider="openai", model="gpt-6-luna", reasoning="medium"
    )
    assert [m.label for m in settings.llm_fallback_models] == [
        "openai/gpt-4o-mini", "gemini/gemini-3.5-flash",
    ]  # fmt: skip
    assert settings.llm_fallback_models[0].reasoning is None  # nothing sent for gpt-4o-mini


def test_reasoning_is_optional_and_a_blank_line_means_unset(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_REASONING", "")
    assert Settings().main_model.reasoning is None  # no reasoning_effort is sent


def test_no_model_means_no_main_model(monkeypatch):
    assert Settings().main_model is None  # the service then fails at startup, naming the variables


def test_no_fallback_models_is_an_empty_chain(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-6-luna")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "")
    assert Settings().llm_fallback_models == []


@pytest.mark.parametrize(
    ("env", "problem"),
    [
        ({"LLM_MODEL": "gpt-6-luna"}, "LLM_PROVIDER is not set"),
        ({"LLM_PROVIDER": "mistral", "LLM_MODEL": "large"}, "llm_provider"),
        ({"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-6-luna", "LLM_REASONING": "extreme"},
         "llm_reasoning"),
        ({"LLM_PROVIDER": "openai", "LLM_MODEL": "openai/gpt-6-luna@medium"},
         "LLM_MODEL is the model id only"),  # the old one-string format
        ({"LLM_FALLBACK_MODELS": "gpt-4o-mini"}, "provider/model"),
    ],
)  # fmt: skip
def test_a_bad_model_setting_fails_at_startup_and_says_what_to_set(monkeypatch, env, problem):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError, match=problem):
        Settings()


def test_keys_come_from_the_environment_and_are_never_shown(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "s3cret-value")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-s3cret")
    monkeypatch.setenv("TICKETING_API_KEY", "t1cket-key")
    settings = Settings()
    assert settings.gemini_api_key.get_secret_value() == "s3cret-value"
    assert settings.openai_api_key.get_secret_value() == "sk-s3cret"
    for secret in ("s3cret-value", "sk-s3cret", "t1cket-key"):
        assert secret not in repr(settings)


def test_the_ticketing_key_defaults_to_the_mock_and_may_not_be_blank(monkeypatch):
    assert Settings().ticketing_api_key.get_secret_value() == "dev-key-123"
    monkeypatch.setenv("TICKETING_API_KEY", "  ")
    with pytest.raises(ValidationError, match="TICKETING_API_KEY"):
        Settings()
