"""
Funções utilitárias para o NeuroPredict.
"""

import hashlib
import json
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import numpy as np
import pandas as pd
from loguru import logger

T = TypeVar("T")


def timing_decorator(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator para medir tempo de execução.
    
    Args:
        func: Função a ser decorada
        
    Returns:
        Função decorada
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} executado em {elapsed:.2f}s")
        return result
    return wrapper


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator para retry em caso de falha.
    
    Args:
        max_attempts: Número máximo de tentativas
        delay: Delay entre tentativas (segundos)
        exceptions: Tupla de exceções para retry
        
    Returns:
        Decorator
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(
                        f"{func.__name__} falhou (tentativa {attempt + 1}/{max_attempts}): {e}"
                    )
                    time.sleep(delay * (attempt + 1))
            raise RuntimeError("Não deveria chegar aqui")
        return wrapper
    return decorator


def calculate_hash(data: Union[str, bytes, Dict]) -> str:
    """
    Calcula hash MD5 de dados.
    
    Args:
        data: Dados para hash
        
    Returns:
        Hash MD5
    """
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)
    if isinstance(data, str):
        data = data.encode()
    return hashlib.md5(data).hexdigest()


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Garante que diretório existe.
    
    Args:
        path: Caminho do diretório
        
    Returns:
        Path do diretório
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict, path: Union[str, Path]) -> None:
    """
    Salva dados em JSON.
    
    Args:
        data: Dados a salvar
        path: Caminho do arquivo
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"JSON salvo em {path}")


def load_json(path: Union[str, Path]) -> Dict:
    """
    Carrega dados de JSON.
    
    Args:
        path: Caminho do arquivo
        
    Returns:
        Dados carregados
    """
    with open(path, "r") as f:
        data = json.load(f)
    logger.info(f"JSON carregado de {path}")
    return data


def split_train_val_test(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify_col: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide dados em train/val/test.
    
    Args:
        df: DataFrame
        train_size: Proporção de treino
        val_size: Proporção de validação
        test_size: Proporção de teste
        random_state: Seed
        stratify_col: Coluna para estratificação
        
    Returns:
        Tupla (train, val, test)
    """
    from sklearn.model_selection import train_test_split
    
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6
    
    stratify = df[stratify_col] if stratify_col else None
    
    # Split train e temp
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify,
    )
    
    # Split val e test
    val_proportion = val_size / (val_size + test_size)
    stratify_temp = temp_df[stratify_col] if stratify_col else None
    
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_proportion,
        random_state=random_state,
        stratify=stratify_temp,
    )
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calcula pesos de classes para balanceamento.
    
    Args:
        y: Array de labels
        
    Returns:
        Dicionário {classe: peso}
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y,
    )
    
    return dict(zip(classes.tolist(), weights.tolist()))


def format_duration(seconds: float) -> str:
    """
    Formata duração em formato legível.
    
    Args:
        seconds: Duração em segundos
        
    Returns:
        String formatada
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def get_memory_usage(df: pd.DataFrame) -> str:
    """
    Retorna uso de memória do DataFrame.
    
    Args:
        df: DataFrame
        
    Returns:
        String com uso de memória
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / 1024 / 1024
    return f"{memory_mb:.2f} MB"


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduz uso de memória otimizando tipos de dados.
    
    Args:
        df: DataFrame
        
    Returns:
        DataFrame otimizado
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
    reduction = 100 * (start_mem - end_mem) / start_mem
    
    logger.info(
        f"Memória reduzida de {start_mem:.2f}MB para {end_mem:.2f}MB "
        f"({reduction:.1f}% de redução)"
    )
    
    return df


def get_timestamp() -> str:
    """
    Retorna timestamp atual formatado.
    
    Returns:
        String timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def convert_to_serializable(obj: Any) -> Any:
    """
    Converte objeto para formato serializável em JSON.
    
    Args:
        obj: Objeto a converter
        
    Returns:
        Objeto serializável
    """
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Path):
        return str(obj)
    return obj


def validate_patient_data(data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Valida dados do paciente.
    
    Args:
        data: Dicionário com dados do paciente
        
    Returns:
        Tupla (válido, lista de erros)
    """
    errors = []
    
    # Campos obrigatórios
    required_fields = [
        "patient_id",
        "age",
        "sex",
        "seizure_type",
        "seizure_frequency_per_month",
        "age_at_onset",
        "epilepsy_duration_years",
    ]
    
    for field in required_fields:
        if field not in data:
            errors.append(f"Campo obrigatório ausente: {field}")
    
    # Validações específicas
    if "age" in data:
        if not isinstance(data["age"], (int, float)) or data["age"] < 0 or data["age"] > 120:
            errors.append("Idade inválida (deve estar entre 0 e 120)")
    
    if "sex" in data:
        if data["sex"] not in ["M", "F", "Other"]:
            errors.append("Sexo inválido (deve ser M, F ou Other)")
    
    if "seizure_frequency_per_month" in data:
        if not isinstance(data["seizure_frequency_per_month"], (int, float)) or data["seizure_frequency_per_month"] < 0:
            errors.append("Frequência de crises inválida (deve ser >= 0)")
    
    if "age_at_onset" in data and "age" in data:
        if data["age_at_onset"] > data["age"]:
            errors.append("Idade de início não pode ser maior que idade atual")
    
    return len(errors) == 0, errors


def create_experiment_id() -> str:
    """
    Cria ID único para experimento.
    
    Returns:
        ID do experimento
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
    return f"exp_{timestamp}_{random_suffix}"


def log_experiment_config(config: Dict[str, Any], output_dir: Path) -> None:
    """
    Loga configuração do experimento.
    
    Args:
        config: Configuração
        output_dir: Diretório de saída
    """
    ensure_dir(output_dir)
    config_path = output_dir / "experiment_config.json"
    save_json(config, config_path)
    logger.info(f"Configuração salva em {config_path}")


def compare_distributions(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """
    Compara distribuições entre dois DataFrames.
    
    Args:
        df1: Primeiro DataFrame
        df2: Segundo DataFrame
        columns: Colunas para comparar
        
    Returns:
        DataFrame com estatísticas comparativas
    """
    from scipy import stats
    
    comparisons = []
    
    for col in columns:
        if col not in df1.columns or col not in df2.columns:
            continue
        
        # KS test para comparar distribuições
        ks_stat, p_value = stats.ks_2samp(df1[col].dropna(), df2[col].dropna())
        
        comparisons.append({
            "column": col,
            "df1_mean": df1[col].mean(),
            "df2_mean": df2[col].mean(),
            "df1_std": df1[col].std(),
            "df2_std": df2[col].std(),
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "distributions_differ": p_value < 0.05,
        })
    
    return pd.DataFrame(comparisons)