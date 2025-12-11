"""
Pipeline de ETL para dados clínicos, genômicos e de neuroimagem.
Implementa validação, limpeza e transformação de dados multimodais.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import pandera as pa
import pydicom
from loguru import logger
from pandera.typing import DataFrame, Series
from sklearn.preprocessing import LabelEncoder, StandardScaler

============================================================================

class ClinicalDataSchema(pa.DataFrameModel):
    """Schema para dados clínicos."""
    
    patient_id: Series[str] = pa.Field(unique=True, nullable=False)
    age: Series[int] = pa.Field(ge=0, le=120)
    sex: Series[str] = pa.Field(isin=["M", "F", "Other"])
    seizure_type: Series[str] = pa.Field(
        isin=[
            "focal_aware",
            "focal_impaired_awareness",
            "focal_to_bilateral_tonic_clonic",
            "generalized_tonic_clonic",
            "absence",
            "myoclonic",
            "atonic",
        ]
    )
    seizure_frequency_per_month: Series[float] = pa.Field(ge=0)
    age_at_onset: Series[int] = pa.Field(ge=0, le=120)
    epilepsy_duration_years: Series[float] = pa.Field(ge=0)
    previous_treatments: Series[str]
    treatment_response: Series[str] = pa.Field(
        isin=["responder", "non_responder", "partial_responder"]
    )
    
    class Config:
        strict = True
        coerce = True


class GeneticDataSchema(pa.DataFrameModel):
    """Schema para dados genômicos."""
    
    patient_id: Series[str] = pa.Field(nullable=False)
    gene: Series[str] = pa.Field(nullable=False)
    variant: Series[str] = pa.Field(nullable=False)
    variant_type: Series[str] = pa.Field(
        isin=["missense", "nonsense", "frameshift", "splice_site", "synonymous"]
    )
    allele_frequency: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    clinvar_significance: Series[str] = pa.Field(nullable=True)
    
    class Config:
        strict = False


@dataclass
class ProcessingResult:
    """Resultado do processamento de dados."""
    
    data: pd.DataFrame
    metadata: Dict[str, Any]
    validation_errors: List[str]
    processing_time: float
    n_records_input: int
    n_records_output: int


class DataProcessor(ABC):
    """Classe base abstrata para processadores de dados."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.logger = logger.bind(processor=self.__class__.__name__)
    
    @abstractmethod
    def load(self, source: Any) -> pd.DataFrame:
        """Carrega dados da fonte."""
        pass
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Valida dados."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforma dados."""
        pass
    
    def process(self, source: Any) -> ProcessingResult:
        import time
        
        start_time = time.time()
        
        self.logger.info("Carregando dados...")
        data = self.load(source)
        n_input = len(data)
        
        self.logger.info("Validando dados...")
        data, errors = self.validate(data)
        
        self.logger.info("Transformando dados...")
        data = self.transform(data)
        
        processing_time = time.time() - start_time
        
        result = ProcessingResult(
            data=data,
            metadata={
                "processor": self.__class__.__name__,
                "config": self.config,
            },
            validation_errors=errors,
            processing_time=processing_time,
            n_records_input=n_input,
            n_records_output=len(data),
        )
        
        self.logger.info(
            f"Processamento concluído: {n_input} -> {len(data)} registros "
            f"em {processing_time:.2f}s"
        )
        
        return result


class ClinicalDataProcessor(DataProcessor):
    """Processador para dados clínicos."""
    
    def load(self, source: Path) -> pd.DataFrame:
        """Carrega dados clínicos de CSV ou banco."""
        if isinstance(source, Path):
            return pd.read_csv(source)
        elif isinstance(source, str) and source.startswith("postgresql://"):
            from sqlalchemy import create_engine
            engine = create_engine(source)
            return pd.read_sql("SELECT * FROM clinical_data", engine)
        else:
            raise ValueError(f"Fonte não suportada: {type(source)}")
    
    def validate(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Valida dados clínicos usando Pandera."""
        errors = []
        
        try:
            validated_data = ClinicalDataSchema.validate(data, lazy=True)
            return validated_data, errors
        except pa.errors.SchemaErrors as e:
            for failure in e.failure_cases.itertuples():
                errors.append(
                    f"Linha {failure.index}: {failure.column} - {failure.check}"
                )
            
            invalid_indices = e.failure_cases["index"].unique()
            cleaned_data = data.drop(invalid_indices)
            
            self.logger.warning(
                f"Removidas {len(invalid_indices)} linhas inválidas"
            )
            
            return cleaned_data, errors
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforma dados clínicos."""
        df = data.copy()
        
        df["years_since_onset"] = df["age"] - df["age_at_onset"]
        df["seizure_burden"] = (
            df["seizure_frequency_per_month"] * df["epilepsy_duration_years"] * 12
        )
        
        df["n_previous_treatments"] = df["previous_treatments"].str.split(";").str.len()
        df["has_previous_treatment"] = df["n_previous_treatments"] > 0
        
        label_encoders = {}
        cat_columns = ["sex", "seizure_type", "treatment_response"]
        
        for col in cat_columns:
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        numeric_cols = [
            "age",
            "seizure_frequency_per_month",
            "epilepsy_duration_years",
            "seizure_burden",
        ]
        
        scaler = StandardScaler()
        df[[f"{col}_scaled" for col in numeric_cols]] = scaler.fit_transform(
            df[numeric_cols]
        )
        
        df.attrs["encoders"] = label_encoders
        df.attrs["scaler"] = scaler
        
        return df


class GeneticDataProcessor(DataProcessor):
    """Processador para dados genômicos (VCF, anotações)."""
    
    def load(self, source: Path) -> pd.DataFrame:
        """Carrega dados genômicos."""
        if source.suffix == ".vcf":
            return self._load_vcf(source)
        elif source.suffix == ".csv":
            return pd.read_csv(source)
        else:
            raise ValueError(f"Formato não suportado: {source.suffix}")
    
    def _load_vcf(self, vcf_path: Path) -> pd.DataFrame:
        """Carrega arquivo VCF."""
        from Bio import SeqIO
        
        records = []
        
        with open(vcf_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                
                fields = line.strip().split("\t")
                records.append({
                    "chrom": fields[0],
                    "pos": int(fields[1]),
                    "ref": fields[3],
                    "alt": fields[4],
                    "qual": float(fields[5]) if fields[5] != "." else None,
                })
        
        return pd.DataFrame(records)
    
    def validate(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Valida dados genômicos."""
        errors = []
        
        try:
            validated_data = GeneticDataSchema.validate(data, lazy=True)
            return validated_data, errors
        except pa.errors.SchemaErrors as e:
            for failure in e.failure_cases.itertuples():
                errors.append(
                    f"Linha {failure.index}: {failure.column} - {failure.check}"
                )
            
            invalid_indices = e.failure_cases["index"].unique()
            cleaned_data = data.drop(invalid_indices)
            
            return cleaned_data, errors
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforma dados genômicos."""
        df = data.copy()
        
        patient_variants = (
            df.groupby("patient_id")
            .agg({
                "gene": lambda x: list(x.unique()),
                "variant": lambda x: list(x.unique()),
                "variant_type": lambda x: list(x.unique()),
            })
            .reset_index()
        )
        
        patient_variants["n_genes_affected"] = patient_variants["gene"].str.len()
        patient_variants["n_variants"] = patient_variants["variant"].str.len()
        
        epilepsy_genes = {
            "SCN1A", "SCN2A", "SCN8A", "KCNQ2", "KCNQ3",
            "GABRA1", "GABRG2", "STXBP1", "PCDH19", "CDKL5",
        }
        
        patient_variants["has_epilepsy_gene"] = patient_variants["gene"].apply(
            lambda genes: any(g in epilepsy_genes for g in genes)
        )
        
        patient_variants["epilepsy_genes"] = patient_variants["gene"].apply(
            lambda genes: [g for g in genes if g in epilepsy_genes]
        )
        
        return patient_variants


class NeuroimagingProcessor(DataProcessor):
    """Processador para neuroimagem (MRI, fMRI)."""
    
    def load(self, source: Path) -> pd.DataFrame:
        """Carrega dados de neuroimagem."""
        if source.suffix in [".nii", ".nii.gz"]:
            return self._load_nifti(source)
        elif source.suffix == ".dcm":
            return self._load_dicom(source)
        else:
            raise ValueError(f"Formato não suportado: {source.suffix}")
    
    def _load_nifti(self, nifti_path: Path) -> pd.DataFrame:
        """Carrega arquivo NIfTI."""
        img = nib.load(str(nifti_path))
        data = img.get_fdata()
        
        features = {
            "patient_id": nifti_path.stem,
            "shape": data.shape,
            "mean_intensity": np.mean(data),
            "std_intensity": np.std(data),
            "min_intensity": np.min(data),
            "max_intensity": np.max(data),
            "volume_cm3": np.prod(img.header.get_zooms()[:3]) * np.prod(data.shape),
        }
        
        return pd.DataFrame([features])
    
    def _load_dicom(self, dicom_path: Path) -> pd.DataFrame:
        """Carrega arquivo DICOM."""
        dcm = pydicom.dcmread(str(dicom_path))
        
        features = {
            "patient_id": dcm.PatientID,
            "modality": dcm.Modality,
            "acquisition_date": dcm.AcquisitionDate,
            "slice_thickness": dcm.SliceThickness,
        }
        
        return pd.DataFrame([features])
    
    def validate(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Valida dados de neuroimagem."""
        errors = []
        
        if "patient_id" not in data.columns:
            errors.append("Coluna patient_id ausente")
        
        if data["patient_id"].duplicated().any():
            errors.append("IDs de paciente duplicados")
            data = data.drop_duplicates(subset=["patient_id"])
        
        return data, errors
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforma dados de neuroimagem."""
        df = data.copy()
        
        if "mean_intensity" in df.columns:
            scaler = StandardScaler()
            df["mean_intensity_scaled"] = scaler.fit_transform(
                df[["mean_intensity"]]
            )
        
        return df


class IntegratedDataPipeline:
    """Pipeline integrado para todos os tipos de dados."""
    
    def __init__(self) -> None:
        """Inicializa pipeline."""
        self.processors = {
            "clinical": ClinicalDataProcessor(),
            "genetic": GeneticDataProcessor(),
            "neuroimaging": NeuroimagingProcessor(),
        }
        self.logger = logger.bind(pipeline="IntegratedDataPipeline")
    
    def process_all(
        self,
        sources: Dict[str, Any],
    ) -> Dict[str, ProcessingResult]:
        results = {}
        
        for data_type, source in sources.items():
            if data_type not in self.processors:
                self.logger.warning(f"Processador não encontrado para {data_type}")
                continue
            
            self.logger.info(f"Processando {data_type}...")
            results[data_type] = self.processors[data_type].process(source)
        
        return results
    
    def merge_data(
        self,
        results: Dict[str, ProcessingResult],
    ) -> pd.DataFrame:
        merged = results["clinical"].data.copy()
        
        if "genetic" in results:
            genetic_df = results["genetic"].data
            merged = merged.merge(
                genetic_df,
                on="patient_id",
                how="left",
                suffixes=("", "_genetic"),
            )
        
        if "neuroimaging" in results:
            neuro_df = results["neuroimaging"].data
            merged = merged.merge(
                neuro_df,
                on="patient_id",
                how="left",
                suffixes=("", "_neuro"),
            )
        
        self.logger.info(f"Dados integrados: {merged.shape}")
        
        return merged