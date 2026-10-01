"""Environment-only settings. Domain objects never read this module."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_backend: str = "dummy"
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    adapter_path: str = ""
    adapter_revision: str = "base"
    schema_version: str = "1.0.0"
    prompt_version: str = "1.0.0"
    max_upload_mb: int = 10
    max_pages: int = 5
    long_edge: int = 1008
    max_pixels: int = 1008 * 28 * 28
    queue_depth: int = 8
    queue_timeout_s: float = 60.0
    inference_timeout_s: float = 120.0
    ttl_s: int = 86_400
    redis_url: str = ""
    cors_origins: str = "http://localhost:5173,http://localhost:8080,http://127.0.0.1:5173"
    logfire_token: str = ""
    logfire_send_to_logfire: str = ""
    hf_home: str = ""
    hf_token: str = ""
    hf_adapter_repo: str = "your-username/vlm-receipt-extraction-lora"
    wandb_api_key: str = ""
    wandb_project: str = "vlm-receipt-extraction"
    wandb_mode: str = ""
    wandb_log_samples: bool = False

    @property
    def max_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def send_to_logfire(self) -> bool:
        if self.logfire_send_to_logfire.lower() in {"false", "0", "no"}:
            return False
        return bool(self.logfire_token)
