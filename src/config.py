from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite:///bookkeeping.db"
    household_id: str = "default"
    secret_key: str = "change-me-to-a-random-string"
    boc_valet_base_url: str = "https://www.bankofcanada.ca/valet"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
