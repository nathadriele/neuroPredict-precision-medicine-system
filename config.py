"""
Módulo de configuração centralizada do NeuroPredict.
Utiliza Pydantic para validação e gerenciamento de configurações.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Configurações de banco de dados."""
    
    model_config = SettingsConfigDict(env_prefix="DB_")
    
    host: str = Field(default="localhost", description="Host do PostgreSQL")
    port: int = Field(default=5432, description="Porta do PostgreSQL")
    name: str = Field(default="neuropredict", description="Nome do banco")
    user: str = Field(default="postgres", description="Usuário")
    password: str = Field(default="", description="Senha")
    
    @property
    def url(self) -> str:
        """Retorna URL de conexão SQLAlchemy."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class Neo4jConfig(BaseSettings):
    """Configurações do Neo4j para grafo de conhecimento."""
    
    model_config = SettingsConfigDict(env_prefix="NEO4J_")
    
    uri: str = Field(default="bolt://localhost:7687", description="URI do Neo4j")
    user: str = Field(default="neo4j", description="Usuário")
    password: str = Field(default="", description="Senha")
    database: str = Field(default="neo4j", description="Nome do banco")


class RedisConfig(BaseSettings):
    """Configurações do Redis para cache."""
    
    model_config = SettingsConfigDict(env_prefix="REDIS_")
    
    host: str = Field(default="localhost", description="Host do Redis")
    port: int = Field(default=6379, description="Porta do Redis")
    db: int = Field(default=0, description="Database index")
    password: Optional[str] = Field(default=None, description="Senha")
    ttl: int = Field(default=3600, description="TTL padrão em segundos")


class MLFlowConfig(BaseSettings):
    """Configurações do MLFlow."""
    
    model_config = SettingsConfigDict(env_prefix="MLFLOW_")
    
    tracking_uri: str = Field(
        default="http://localhost:5000",
        description="URI do servidor MLFlow"
    )
    experiment_name: str = Field(
        default="neuropredict_experiments",
        description="Nome do experimento"
    )
    artifact_location: Optional[str] = Field(
        default=None,
        description="Localização dos artifacts"
    )


class ModelConfig(BaseSettings):
    """Configurações de modelos ML/DL."""
    
    model_config = SettingsConfigDict(env_prefix="MODEL_")
    
    # Ensemble
    ensemble_models: List[str] = Field(
        default=["xgboost", "lightgbm", "catboost", "neural_net"],
        description="Modelos no ensemble"
    )
    ensemble_method: str = Field(
        default="voting",
        description="Método de ensemble (voting, stacking, blending)"
    )
    
    # Hiperparâmetros
    cv_folds: int = Field(default=5, description="Número de folds para CV")
    random_state: int = Field(default=42, description="Seed para reprodutibilidade")
    n_jobs: int = Field(default=-1, description="Número de jobs paralelos")
    
    # Otimização
    hpo_trials: int = Field(default=100, description="Número de trials para HPO")
    hpo_timeout: int = Field(default=3600, description="Timeout para HPO em segundos")
    
    # Neural Networks
    batch_size: int = Field(default=32, description="Tamanho do batch")
    epochs: int = Field(default=100, description="Número de épocas")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    early_stopping_patience: int = Field(
        default=10,
        description="Paciência para early stopping"
    )
    
    # GNN
    gnn_hidden_channels: int = Field(default=128, description="Canais escondidos GNN")
    gnn_num_layers: int = Field(default=3, description="Número de camadas GNN")
    gnn_dropout: float = Field(default=0.3, description="Dropout GNN")


class LLMConfig(BaseSettings):
    """Configurações de LLMs e RAG."""
    
    model_config = SettingsConfigDict(env_prefix="LLM_")
    
    provider: str = Field(default="openai", description="Provider (openai, anthropic)")
    model_name: str = Field(
        default="gpt-4-turbo-preview",
        description="Nome do modelo"
    )
    api_key: str = Field(default="", description="API key")
    temperature: float = Field(default=0.1, description="Temperatura")
    max_tokens: int = Field(default=2000, description="Tokens máximos")
    
    # RAG
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Modelo de embeddings"
    )
    chunk_size: int = Field(default=1000, description="Tamanho dos chunks")
    chunk_overlap: int = Field(default=200, description="Overlap entre chunks")
    top_k: int = Field(default=5, description="Top K documentos recuperados")
    
    # ChromaDB
    chroma_persist_directory: str = Field(
        default="./data/chroma_db",
        description="Diretório do ChromaDB"
    )


class DataConfig(BaseSettings):
    """Configurações de dados."""
    
    model_config = SettingsConfigDict(env_prefix="DATA_")
    
    raw_data_path: Path = Field(
        default=Path("./data/raw"),
        description="Caminho dos dados brutos"
    )
    processed_data_path: Path = Field(
        default=Path("./data/processed"),
        description="Caminho dos dados processados"
    )
    external_data_path: Path = Field(
        default=Path("./data/external"),
        description="Caminho dos dados externos"
    )
    
    # Validação
    min_samples: int = Field(default=100, description="Mínimo de amostras")
    test_size: float = Field(default=0.2, description="Proporção de teste")
    val_size: float = Field(default=0.1, description="Proporção de validação")
    
    @field_validator("test_size", "val_size")
    @classmethod
    def validate_proportion(cls, v: float) -> float:
        """Valida proporções entre 0 e 1."""
        if not 0 < v < 1:
            raise ValueError("Proporção deve estar entre 0 e 1")
        return v


class APIConfig(BaseSettings):
    """Configurações da API."""
    
    model_config = SettingsConfigDict(env_prefix="API_")
    
    host: str = Field(default="0.0.0.0", description="Host")
    port: int = Field(default=8000, description="Porta")
    reload: bool = Field(default=False, description="Auto-reload")
    workers: int = Field(default=4, description="Número de workers")
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501"],
        description="Origens permitidas para CORS"
    )
    
    # Rate limiting
    rate_limit: int = Field(default=100, description="Requests por minuto")
    
    # JWT
    secret_key: str = Field(default="", description="Secret key para JWT")
    algorithm: str = Field(default="HS256", description="Algoritmo JWT")
    access_token_expire_minutes: int = Field(
        default=30,
        description="Expiração do token"
    )


class LogConfig(BaseSettings):
    """Configurações de logging."""
    
    model_config = SettingsConfigDict(env_prefix="LOG_")
    
    level: str = Field(default="INFO", description="Nível de log")
    format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
                "<level>{message}</level>",
        description="Formato do log"
    )
    file_path: Optional[Path] = Field(
        default=Path("./logs/neuropredict.log"),
        description="Caminho do arquivo de log"
    )
    rotation: str = Field(default="100 MB", description="Rotação do log")
    retention: str = Field(default="30 days", description="Retenção do log")


class Settings(BaseSettings):
    """Configurações principais do sistema."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Informações do projeto
    project_name: str = Field(
        default="NeuroPredict",
        description="Nome do projeto"
    )
    version: str = Field(default="0.1.0", description="Versão")
    debug: bool = Field(default=False, description="Modo debug")
    
    # Configurações de sub-módulos
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    mlflow: MLFlowConfig = Field(default_factory=MLFlowConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    
    def __init__(self, **kwargs: Any) -> None:
        """Inicializa configurações."""
        super().__init__(**kwargs)
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Cria diretórios necessários se não existirem."""
        directories = [
            self.data.raw_data_path,
            self.data.processed_data_path,
            self.data.external_data_path,
            Path("./models"),
            Path("./logs"),
            Path(self.llm.chroma_persist_directory),
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte configurações para dicionário."""
        return self.model_dump()


@lru_cache()
def get_settings() -> Settings:
    """
    Retorna instância singleton das configurações.
    Utiliza cache para evitar múltiplas leituras.
    """
    return Settings()


# Instância global para importação direta
settings = get_settings()