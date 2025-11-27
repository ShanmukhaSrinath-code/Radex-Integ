from pydantic_settings import BaseSettings
from typing import Optional
from pydantic import computed_field


class Settings(BaseSettings):
    # ================= DATABASE =================
    database_url: Optional[str] = None
    db_host: Optional[str] = "localhost"
    db_port: Optional[int] = 5432
    db_name: Optional[str] = "ragdb"
    db_user: Optional[str] = "raguser"
    db_password: Optional[str] = "changeme"

    # ================= REDIS =================
    redis_url: Optional[str] = None
    redis_host: Optional[str] = "localhost"
    redis_port: Optional[int] = 6379
    redis_password: Optional[str] = None

    # ================= MINIO =================
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str = "documents"
    minio_secure: bool = False

    # ================= JWT =================
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 30

    # ================= OPENAI =================
    openai_api_key: str
    openai_chat_model: str = "gpt-3.5-turbo"
    openai_reformulation_model: str = "gpt-3.5-turbo"

    # ================= FIREBASE =================
    firebase_admin_sdk_json: Optional[str] = None

    # ================= OKTA =================
    okta_client_id: Optional[str] = None
    okta_client_secret: Optional[str] = None
    okta_domain: Optional[str] = None
    okta_redirect_uri: Optional[str] = None
    okta_api_token: Optional[str] = None
    okta_api_audience: Optional[str] = None

    # ================= SHAREPOINT =================
    enable_sharepoint_provider: bool = False
    sp_client_id: Optional[str] = None
    sp_client_secret: Optional[str] = None
    sp_redirect_uri: Optional[str] = None

    # ================= ENCRYPTION =================
    encryption_key: Optional[str] = None

    # ================= APP =================
    app_name: str = "RAG RBAC System"
    debug: bool = True


    # -------------------------------------------------
    # ✅ DATABASE URL (FOR psycopg v3)
    # -------------------------------------------------
    @computed_field
    @property
    def effective_database_url(self) -> str:
        if self.database_url:
            return self.database_url.replace(
                "postgresql://",
                "postgresql+psycopg://"
            )

        return (
            f"postgresql+psycopg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


    # -------------------------------------------------
    # ✅ REDIS URL
    # -------------------------------------------------
    @computed_field
    @property
    def effective_redis_url(self) -> str:
        if self.redis_url:
            return self.redis_url

        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"

        return f"redis://{self.redis_host}:{self.redis_port}/0"


    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


settings = Settings()
