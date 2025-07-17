from pydantic_settings import BaseSettings

class DatabaseSettings(BaseSettings):
    url: str

class EmbeddingSettings(BaseSettings):
    azure_endpoint: str
    api_key: str
    model_name: str
    api_version: str

    class Config:
        env_prefix = "AZURE_OPENAI_"

class Settings(BaseSettings):
    database_url: str
    azure_openai_endpoint: str
    azure_openai_api_key: str

    embedding_model_name: str
    embedding_api_version: str

    chat_model_name: str
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = 'ignore'

settings = Settings()