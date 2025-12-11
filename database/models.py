"""
SQLAlchemy models para o banco de dados PostgreSQL.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Patient(Base):
    """Modelo de paciente."""
    
    __tablename__ = "patients"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.uuid_generate_v4())
    patient_id = Column(String(50), unique=True, nullable=False, index=True)
    age = Column(Integer, nullable=False)
    sex = Column(String(10), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    clinical_data = relationship("ClinicalData", back_populates="patient", uselist=False)
    treatments = relationship("Treatment", back_populates="patient")
    genetic_variants = relationship("GeneticVariant", back_populates="patient")
    predictions = relationship("Prediction", back_populates="patient")
    
    def __repr__(self) -> str:
        return f"<Patient(id={self.patient_id}, age={self.age})>"


class ClinicalData(Base):
    """Dados clínicos do paciente."""
    
    __tablename__ = "clinical_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.uuid_generate_v4())
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    seizure_type = Column(String(100), nullable=False)
    seizure_frequency_per_month = Column(Float, nullable=False)
    age_at_onset = Column(Integer, nullable=False)
    epilepsy_duration_years = Column(Float, nullable=False)
    treatment_response = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    patient = relationship("Patient", back_populates="clinical_data")
    
    def __repr__(self) -> str:
        return f"<ClinicalData(patient_id={self.patient_id}, seizure_type={self.seizure_type})>"


class Treatment(Base):
    """Histórico de tratamentos."""
    
    __tablename__ = "treatments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.uuid_generate_v4())
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    treatment_name = Column(String(100), nullable=False)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    dosage = Column(String(50))
    response = Column(String(50))
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    patient = relationship("Patient", back_populates="treatments")
    
    def __repr__(self) -> str:
        return f"<Treatment(treatment={self.treatment_name}, response={self.response})>"


class GeneticVariant(Base):
    """Variantes genéticas."""
    
    __tablename__ = "genetic_variants"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.uuid_generate_v4())
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    gene = Column(String(50), nullable=False, index=True)
    variant = Column(String(200), nullable=False)
    variant_type = Column(String(50))
    allele_frequency = Column(Float)
    clinvar_significance = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    patient = relationship("Patient", back_populates="genetic_variants")
    
    def __repr__(self) -> str:
        return f"<GeneticVariant(gene={self.gene}, variant={self.variant})>"


class Prediction(Base):
    """Predições do modelo."""
    
    __tablename__ = "predictions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.uuid_generate_v4())
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    model_version = Column(String(50), nullable=False)
    predicted_treatment = Column(String(100), nullable=False)
    response_probability = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    explanation = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationship
    patient = relationship("Patient", back_populates="predictions")
    
    def __repr__(self) -> str:
        return f"<Prediction(treatment={self.predicted_treatment}, prob={self.response_probability:.2f})>"


class APILog(Base):
    """Logs de requisições da API."""
    
    __tablename__ = "api_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.uuid_generate_v4())
    endpoint = Column(String(200), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)
    response_time_ms = Column(Integer)
    user_id = Column(String(100))
    ip_address = Column(String(45))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self) -> str:
        return f"<APILog(endpoint={self.endpoint}, status={self.status_code})>"


class ModelMetadata(Base):
    """Metadados de modelos treinados."""
    
    __tablename__ = "model_metadata"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.uuid_generate_v4())
    version = Column(String(50), unique=True, nullable=False)
    model_type = Column(String(50), nullable=False)
    metrics = Column(JSON, nullable=False)
    hyperparameters = Column(JSON)
    training_date = Column(DateTime, default=datetime.utcnow)
    dataset_size = Column(Integer)
    feature_names = Column(JSON)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self) -> str:
        return f"<ModelMetadata(version={self.version}, active={self.is_active})>"