"""Settings. The environment (.env) holds the model choice and the secrets, as the assignment asks
("read the provider and model from environment variables"). agent.toml holds the rest.

Precedence: constructor arguments > environment variables > agent.toml > defaults. Any value in
agent.toml can be overridden from the environment with a double underscore between section and
key, e.g. TICKETING__URL (docker-compose.yml does this).
"""

import os
import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    InitSettingsSource,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

Provider = Literal["openai", "gemini"]
Reasoning = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
SECTIONS = {"llm", "agent", "ticketing"}
MODEL_FORMAT = "provider/model or provider/model@reasoning, e.g. openai/gpt-4o-mini"


class ModelChoice(BaseModel):
    """One model: the main one (LLM_PROVIDER, LLM_MODEL, LLM_REASONING), or one entry of
    LLM_FALLBACK_MODELS."""

    provider: Provider
    model: str = Field(min_length=1)
    reasoning: Reasoning | None = None  # None: no reasoning_effort is sent (e.g. gpt-4o-mini)

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"

    @classmethod
    def parse(cls, text: str) -> "ModelChoice":
        spec, _, reasoning = text.strip().partition("@")
        provider, slash, model = spec.partition("/")
        if not slash or not model:
            raise ValueError(f"{text!r} is not {MODEL_FORMAT}")
        return cls(provider=provider, model=model, reasoning=reasoning or None)


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo in agent.toml fails at startup

    timeout_s: float = Field(default=60.0, gt=0)


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(default=6, ge=1)  # LLM calls per turn before the agent gives up
    log_level: str = "INFO"


class TicketingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = "http://localhost:8000"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")

    # The model, from the environment. Required when the service starts (OpenAIChatClient).
    llm_provider: Provider | None = None
    llm_model: str | None = None  # the model id only, e.g. gpt-6-luna
    llm_reasoning: Reasoning | None = None  # None: no reasoning_effort is sent (e.g. gpt-4o-mini)
    # Optional, in order: "provider/model[@reasoning]" entries, separated by commas.
    llm_fallback_models: Annotated[list[ModelChoice], NoDecode] = Field(default_factory=list)

    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    ticketing: TicketingConfig = Field(default_factory=TicketingConfig)

    # Secrets: from the environment (.env) only. agent.toml may not hold them.
    ticketing_api_key: SecretStr = SecretStr("dev-key-123")  # the mock's documented default
    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    @field_validator("ticketing_api_key")
    @classmethod
    def _not_blank(cls, value: SecretStr) -> SecretStr:
        # A blank TICKETING_API_KEY= line would mean a 401 on every ticket call.
        if not value.get_secret_value().strip():
            raise ValueError("TICKETING_API_KEY is blank; remove the line to use the mock's key")
        return value

    @field_validator("llm_provider", "llm_model", "llm_reasoning", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        # LLM_REASONING= in .env means "not set", not an invalid reasoning level.
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("llm_model")
    @classmethod
    def _model_id_only(cls, value: str | None) -> str | None:
        if value and "@" in value:  # the old provider/model@reasoning format
            raise ValueError(
                "LLM_MODEL is the model id only (e.g. gpt-6-luna): set the provider in "
                "LLM_PROVIDER and the reasoning level in LLM_REASONING"
            )
        return value

    @model_validator(mode="after")
    def _a_model_needs_its_provider(self) -> "Settings":
        if self.llm_model and self.llm_provider is None:
            raise ValueError("LLM_PROVIDER is not set: openai or gemini (see .env.example)")
        return self

    @property
    def main_model(self) -> ModelChoice | None:
        """The main model, or None when LLM_MODEL is not set."""
        if not self.llm_model or self.llm_provider is None:
            return None
        return ModelChoice(
            provider=self.llm_provider, model=self.llm_model, reasoning=self.llm_reasoning
        )

    @field_validator("llm_fallback_models", mode="before")
    @classmethod
    def _model_list(cls, value: object) -> object:
        if isinstance(value, str):  # comma-separated, in order
            return [ModelChoice.parse(item) for item in value.split(",") if item.strip()]
        return value

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # .env reaches us as environment variables (make run, docker compose), not as a file.
        return init_settings, env_settings, _config_file(settings_cls)


def _config_file(settings_cls: type[BaseSettings]) -> PydanticBaseSettingsSource:
    """agent.toml, or the file AGENT_CONFIG names. AGENT_CONFIG="" means no file (tests)."""
    path = os.environ.get("AGENT_CONFIG", "agent.toml")
    if not path:
        return InitSettingsSource(settings_cls, init_kwargs={})
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"config file {file.resolve()} not found (set AGENT_CONFIG)")
    data = tomllib.loads(file.read_text())
    unknown = set(data) - SECTIONS
    if unknown:
        raise ValueError(
            f"{path}: unexpected {sorted(unknown)}. It holds only [llm], [agent] and "
            "[ticketing]; keys and tokens go in .env."
        )
    return InitSettingsSource(settings_cls, init_kwargs=data)
