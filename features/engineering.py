"""
Feature engineering para dados de epilepsia.
Extrai e transforma features de múltiplas modalidades.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    PolynomialFeatures,
    StandardScaler,
)


class FeatureEngineer:
    """Engine de feature engineering."""
    
    def __init__(self) -> None:
        """Inicializa feature engineer."""
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.pca: Optional[PCA] = None
        self.logger = logger.bind(component="FeatureEngineer")
        self.feature_names: List[str] = []
    
    def create_temporal_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Cria features temporais relacionadas à epilepsia.
        
        Args:
            df: DataFrame com dados clínicos
            
        Returns:
            DataFrame com features temporais
        """
        df = df.copy()
        
        # Duração desde o início
        if "age" in df.columns and "age_at_onset" in df.columns:
            df["years_since_onset"] = df["age"] - df["age_at_onset"]
            df["years_since_onset_squared"] = df["years_since_onset"] ** 2
        
        # Frequência normalizada por duração
        if "seizure_frequency_per_month" in df.columns and "epilepsy_duration_years" in df.columns:
            df["seizure_burden"] = (
                df["seizure_frequency_per_month"] * df["epilepsy_duration_years"] * 12
            )
            df["avg_yearly_seizures"] = df["seizure_frequency_per_month"] * 12
        
        # Categorias de idade
        if "age" in df.columns:
            df["age_category"] = pd.cut(
                df["age"],
                bins=[0, 12, 18, 30, 50, 120],
                labels=["child", "adolescent", "young_adult", "adult", "senior"],
            )
            
            # One-hot encoding
            age_dummies = pd.get_dummies(df["age_category"], prefix="age")
            df = pd.concat([df, age_dummies], axis=1)
            df.drop("age_category", axis=1, inplace=True)
        
        # Idade relativa ao onset
        if "age_at_onset" in df.columns:
            df["onset_category"] = pd.cut(
                df["age_at_onset"],
                bins=[0, 1, 5, 12, 18, 120],
                labels=["neonatal", "infant", "childhood", "adolescent", "adult_onset"],
            )
            
            onset_dummies = pd.get_dummies(df["onset_category"], prefix="onset")
            df = pd.concat([df, onset_dummies], axis=1)
            df.drop("onset_category", axis=1, inplace=True)
        
        self.logger.info(f"Features temporais criadas: {df.shape}")
        return df
    
    def create_treatment_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Cria features relacionadas a tratamentos.
        
        Args:
            df: DataFrame com histórico de tratamentos
            
        Returns:
            DataFrame com features de tratamento
        """
        df = df.copy()
        
        if "previous_treatments" in df.columns:
            # Número de tratamentos
            df["n_previous_treatments"] = df["previous_treatments"].apply(
                lambda x: len(x.split(";")) if isinstance(x, str) and x else 0
            )
            
            # Teve tratamento anterior
            df["has_previous_treatment"] = (df["n_previous_treatments"] > 0).astype(int)
            
            # Tratamentos específicos (one-hot)
            common_treatments = [
                "levetiracetam",
                "lamotrigine",
                "oxcarbazepine",
                "topiramate",
                "valproato",
                "carbamazepina",
            ]
            
            for treatment in common_treatments:
                df[f"prev_{treatment}"] = df["previous_treatments"].apply(
                    lambda x: 1 if isinstance(x, str) and treatment in x.lower() else 0
                )
            
            # Resistência a tratamento (proxy)
            df["treatment_resistance_score"] = df["n_previous_treatments"].apply(
                lambda x: min(x / 5.0, 1.0)  # Normaliza entre 0-1
            )
        
        self.logger.info(f"Features de tratamento criadas: {df.shape}")
        return df
    
    def create_genetic_features(
        self,
        clinical_df: pd.DataFrame,
        genetic_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Cria features genéticas agregadas por paciente.
        
        Args:
            clinical_df: DataFrame clínico
            genetic_df: DataFrame genético
            
        Returns:
            DataFrame com features genéticas
        """
        if genetic_df.empty:
            self.logger.warning("Sem dados genéticos")
            clinical_df["n_variants"] = 0
            clinical_df["has_pathogenic_variant"] = 0
            clinical_df["has_epilepsy_gene"] = 0
            return clinical_df
        
        # Genes conhecidos de epilepsia
        epilepsy_genes = {
            "SCN1A", "SCN2A", "SCN8A", "KCNQ2", "KCNQ3",
            "GABRA1", "GABRG2", "STXBP1", "PCDH19", "CDKL5",
        }
        
        # Agregar por paciente
        genetic_agg = genetic_df.groupby("patient_id").agg({
            "gene": [
                ("n_variants", "count"),
                ("unique_genes", lambda x: len(set(x))),
            ],
            "variant_type": lambda x: list(x),
            "clinvar_significance": lambda x: list(x),
        }).reset_index()
        
        genetic_agg.columns = ["_".join(col).strip("_") for col in genetic_agg.columns]
        
        # Features derivadas
        genetic_agg["has_pathogenic_variant"] = genetic_df.groupby("patient_id")[
            "clinvar_significance"
        ].apply(
            lambda x: int(any("pathogenic" in str(v).lower() for v in x))
        ).values
        
        genetic_agg["has_epilepsy_gene"] = genetic_df.groupby("patient_id")[
            "gene"
        ].apply(
            lambda x: int(any(g in epilepsy_genes for g in x))
        ).values
        
        # Contagem de tipos de variante
        variant_types = ["missense", "nonsense", "frameshift", "splice_site"]
        for vtype in variant_types:
            genetic_agg[f"n_{vtype}"] = genetic_df.groupby("patient_id")[
                "variant_type"
            ].apply(
                lambda x: sum(1 for v in x if v == vtype)
            ).values
        
        # Merge com dados clínicos
        result = clinical_df.merge(
            genetic_agg,
            left_on="patient_id",
            right_on="patient_id",
            how="left",
        )
        
        # Preenche NaN
        genetic_cols = [col for col in result.columns if col.startswith("n_") or col.startswith("has_")]
        result[genetic_cols] = result[genetic_cols].fillna(0)
        
        self.logger.info(f"Features genéticas criadas: {result.shape}")
        return result
    
    def create_neuroimaging_features(
        self,
        clinical_df: pd.DataFrame,
        neuro_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Cria features de neuroimagem.
        
        Args:
            clinical_df: DataFrame clínico
            neuro_df: DataFrame de neuroimagem
            
        Returns:
            DataFrame com features de neuroimagem
        """
        if neuro_df.empty:
            self.logger.warning("Sem dados de neuroimagem")
            return clinical_df
        
        # Normaliza intensidades
        if "mean_intensity" in neuro_df.columns:
            neuro_df["mean_intensity_norm"] = (
                neuro_df["mean_intensity"] - neuro_df["mean_intensity"].mean()
            ) / neuro_df["mean_intensity"].std()
        
        # Ratios
        if "hippocampal_volume" in neuro_df.columns and "white_matter_volume" in neuro_df.columns:
            neuro_df["hippocampal_to_wm_ratio"] = (
                neuro_df["hippocampal_volume"] / (neuro_df["white_matter_volume"] + 1e-6)
            )
        
        # Atrofia hippocampal (comparado com média normal)
        if "hippocampal_volume" in neuro_df.columns:
            normal_hippocampal_vol = 3500  # mm³
            neuro_df["hippocampal_atrophy"] = (
                (normal_hippocampal_vol - neuro_df["hippocampal_volume"]) 
                / normal_hippocampal_vol
            ).clip(lower=0)
        
        # Merge
        result = clinical_df.merge(
            neuro_df,
            on="patient_id",
            how="left",
        )
        
        self.logger.info(f"Features de neuroimagem criadas: {result.shape}")
        return result
    
    def create_interaction_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Cria features de interação entre variáveis.
        
        Args:
            df: DataFrame com features
            
        Returns:
            DataFrame com features de interação
        """
        df = df.copy()
        
        # Interações importantes
        interactions = [
            ("age", "seizure_frequency_per_month"),
            ("years_since_onset", "seizure_frequency_per_month"),
            ("n_previous_treatments", "seizure_frequency_per_month"),
            ("age", "n_previous_treatments"),
        ]
        
        for col1, col2 in interactions:
            if col1 in df.columns and col2 in df.columns:
                # Multiplicação
                df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
                
                # Divisão (com proteção)
                df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-6)
        
        self.logger.info(f"Features de interação criadas: {df.shape}")
        return df
    
    def create_polynomial_features(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        degree: int = 2,
    ) -> pd.DataFrame:
        """
        Cria features polinomiais.
        
        Args:
            df: DataFrame
            numeric_cols: Colunas numéricas
            degree: Grau do polinômio
            
        Returns:
            DataFrame com features polinomiais
        """
        df = df.copy()
        
        # Seleciona apenas colunas existentes
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if not available_cols:
            return df
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(df[available_cols])
        
        # Nomes das features
        feature_names = poly.get_feature_names_out(available_cols)
        
        # Adiciona apenas novas features (não as originais)
        poly_df = pd.DataFrame(
            poly_features[:, len(available_cols):],
            columns=feature_names[len(available_cols):],
            index=df.index,
        )
        
        result = pd.concat([df, poly_df], axis=1)
        
        self.logger.info(f"Features polinomiais criadas: {result.shape}")
        return result
    
    def encode_categorical(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
    ) -> pd.DataFrame:
        """
        Codifica variáveis categóricas.
        
        Args:
            df: DataFrame
            categorical_cols: Colunas categóricas
            
        Returns:
            DataFrame com variáveis codificadas
        """
        df = df.copy()
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f"{col}_encoded"] = self.label_encoders[col].fit_transform(
                    df[col].astype(str)
                )
            else:
                df[f"{col}_encoded"] = self.label_encoders[col].transform(
                    df[col].astype(str)
                )
        
        return df
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        method: str = "standard",
    ) -> pd.DataFrame:
        """
        Normaliza features numéricas.
        
        Args:
            df: DataFrame
            numeric_cols: Colunas numéricas
            method: Método (standard ou minmax)
            
        Returns:
            DataFrame normalizado
        """
        df = df.copy()
        
        # Seleciona apenas colunas existentes
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if not available_cols:
            return df
        
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Método {method} não suportado")
        
        df[[f"{col}_norm" for col in available_cols]] = scaler.fit_transform(
            df[available_cols]
        )
        
        self.scaler = scaler
        
        return df
    
    def apply_pca(
        self,
        df: pd.DataFrame,
        n_components: int = 10,
    ) -> pd.DataFrame:
        """
        Aplica PCA para redução de dimensionalidade.
        
        Args:
            df: DataFrame
            n_components: Número de componentes
            
        Returns:
            DataFrame com componentes principais
        """
        # Remove colunas não numéricas
        numeric_df = df.select_dtypes(include=[np.number])
        
        self.pca = PCA(n_components=n_components)
        pca_features = self.pca.fit_transform(numeric_df)
        
        pca_df = pd.DataFrame(
            pca_features,
            columns=[f"pca_{i}" for i in range(n_components)],
            index=df.index,
        )
        
        result = pd.concat([df, pca_df], axis=1)
        
        explained_var = self.pca.explained_variance_ratio_.sum()
        self.logger.info(
            f"PCA aplicado: {n_components} componentes explicam "
            f"{explained_var:.2%} da variância"
        )
        
        return result
    
    def transform(
        self,
        clinical_df: pd.DataFrame,
        genetic_df: Optional[pd.DataFrame] = None,
        neuro_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Pipeline completo de feature engineering.
        
        Args:
            clinical_df: DataFrame clínico
            genetic_df: DataFrame genético
            neuro_df: DataFrame de neuroimagem
            
        Returns:
            DataFrame com todas as features
        """
        self.logger.info("Iniciando feature engineering...")
        
        # Features temporais
        df = self.create_temporal_features(clinical_df)
        
        # Features de tratamento
        df = self.create_treatment_features(df)
        
        # Features genéticas
        if genetic_df is not None and not genetic_df.empty:
            df = self.create_genetic_features(df, genetic_df)
        
        # Features de neuroimagem
        if neuro_df is not None and not neuro_df.empty:
            df = self.create_neuroimaging_features(df, neuro_df)
        
        # Features de interação
        df = self.create_interaction_features(df)
        
        # Codifica categóricas
        categorical_cols = ["sex", "seizure_type"]
        df = self.encode_categorical(df, categorical_cols)
        
        # Normaliza numéricas
        numeric_cols = [
            "age",
            "seizure_frequency_per_month",
            "epilepsy_duration_years",
            "years_since_onset",
        ]
        df = self.normalize_features(df, numeric_cols)
        
        # Armazena nomes de features
        self.feature_names = df.columns.tolist()
        
        self.logger.info(f"Feature engineering concluído: {df.shape}")
        
        return df