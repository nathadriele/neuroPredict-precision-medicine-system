"""
Router para endpoints de predição.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from neuropredict.models.predictor import TreatmentPredictor

router = APIRouter(prefix="/predict", tags=["prediction"])

# Estado global (simplificado, em produção use dependency injection)
_predictor: Optional[TreatmentPredictor] = None


def get_predictor() -> TreatmentPredictor:
    """Retorna predictor singleton."""
    global _predictor
    if _predictor is None:
        model_path = Path("models/ensemble_model_v1.pkl")
        if model_path.exists():
            _predictor = TreatmentPredictor.load(model_path)
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modelo não disponível",
            )
    return _predictor


# Schemas
class GeneticVariant(BaseModel):
    """Variante genética."""
    gene: str
    variant: str
    variant_type: str


class PatientData(BaseModel):
    """Dados do paciente."""
    patient_id: str
    age: int = Field(..., ge=0, le=120)
    sex: str
    seizure_type: str
    seizure_frequency_per_month: float = Field(..., ge=0)
    age_at_onset: int = Field(..., ge=0)
    epilepsy_duration_years: float = Field(..., ge=0)
    previous_treatments: List[str] = Field(default_factory=list)
    genetic_variants: List[GeneticVariant] = Field(default_factory=list)
    eeg_features: Optional[Dict[str, float]] = None
    mri_features: Optional[Dict[str, float]] = None


class PredictionRequest(BaseModel):
    """Request de predição."""
    patient: PatientData
    explain: bool = Field(default=True, description="Incluir explicações")


class AlternativeTreatment(BaseModel):
    """Tratamento alternativo."""
    treatment: str
    probability: float


class PredictionResponse(BaseModel):
    """Response de predição."""
    patient_id: str
    predicted_treatment: str
    response_probability: float
    confidence: float
    alternative_treatments: List[AlternativeTreatment]
    all_probabilities: Dict[str, float]
    explanation: Optional[Dict[str, Any]] = None
    timestamp: datetime


class BatchPredictionRequest(BaseModel):
    """Request de predição em batch."""
    patients: List[PatientData]
    explain: bool = False


class BatchPredictionResponse(BaseModel):
    """Response de predição em batch."""
    predictions: List[PredictionResponse]
    total: int
    timestamp: datetime


# Endpoints
@router.post("/", response_model=PredictionResponse)
async def predict_treatment(request: PredictionRequest) -> PredictionResponse:
    """
    Prediz tratamento recomendado para paciente.
    
    Args:
        request: Dados do paciente
        
    Returns:
        Predição com probabilidades e explicações
    """
    try:
        predictor = get_predictor()
        
        # Prepara dados
        patient_dict = request.patient.model_dump()
        
        # Predição
        if request.explain:
            result = predictor.predict_with_explanation(patient_dict)
        else:
            result = predictor.predict(patient_dict)
        
        # Formata alternativas
        alternatives = [
            AlternativeTreatment(
                treatment=alt["treatment"],
                probability=alt["probability"],
            )
            for alt in result["alternative_treatments"]
        ]
        
        return PredictionResponse(
            patient_id=result["patient_id"],
            predicted_treatment=result["recommended_treatment"],
            response_probability=result["response_probability"],
            confidence=result["confidence"],
            alternative_treatments=alternatives,
            all_probabilities=result["all_probabilities"],
            explanation=result.get("explanation"),
            timestamp=datetime.now(),
        )
        
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro na predição: {str(e)}",
        )


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Predição em batch para múltiplos pacientes.
    
    Args:
        request: Lista de pacientes
        
    Returns:
        Lista de predições
    """
    try:
        predictor = get_predictor()
        
        predictions = []
        for patient in request.patients:
            patient_dict = patient.model_dump()
            
            result = predictor.predict(patient_dict)
            
            alternatives = [
                AlternativeTreatment(
                    treatment=alt["treatment"],
                    probability=alt["probability"],
                )
                for alt in result["alternative_treatments"]
            ]
            
            pred = PredictionResponse(
                patient_id=result["patient_id"],
                predicted_treatment=result["recommended_treatment"],
                response_probability=result["response_probability"],
                confidence=result["confidence"],
                alternative_treatments=alternatives,
                all_probabilities=result["all_probabilities"],
                timestamp=datetime.now(),
            )
            
            predictions.append(pred)
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            timestamp=datetime.now(),
        )
        
    except Exception as e:
        logger.error(f"Erro na predição batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro: {str(e)}",
        )


@router.get("/info")
async def model_info() -> Dict[str, Any]:
    """
    Retorna informações sobre o modelo carregado.
    
    Returns:
        Informações do modelo
    """
    try:
        predictor = get_predictor()
        info = predictor.get_model_info()
        
        return {
            "model_info": info,
            "status": "loaded",
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        return {
            "status": "not_loaded",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }