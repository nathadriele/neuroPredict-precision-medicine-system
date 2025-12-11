"""
Carrega dados de exemplo no banco de dados PostgreSQL.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from neuropredict.config import get_settings
from neuropredict.database.models import (
    Base,
    ClinicalData,
    GeneticVariant,
    Patient,
    Treatment,
)


def create_tables(engine):
    """Cria todas as tabelas."""
    logger.info("Criando tabelas...")
    Base.metadata.create_all(engine)
    logger.info("✓ Tabelas criadas")


def load_patients(session, clinical_df: pd.DataFrame):
    """Carrega pacientes."""
    logger.info("Carregando pacientes...")
    
    for _, row in clinical_df.iterrows():
        patient = Patient(
            patient_id=row["patient_id"],
            age=int(row["age"]),
            sex=row["sex"],
        )
        session.add(patient)
    
    session.commit()
    logger.info(f"✓ {len(clinical_df)} pacientes carregados")


def load_clinical_data(session, clinical_df: pd.DataFrame):
    """Carrega dados clínicos."""
    logger.info("Carregando dados clínicos...")
    
    # Busca pacientes
    patients = {p.patient_id: p.id for p in session.query(Patient).all()}
    
    for _, row in clinical_df.iterrows():
        if row["patient_id"] not in patients:
            continue
        
        clinical = ClinicalData(
            patient_id=patients[row["patient_id"]],
            seizure_type=row["seizure_type"],
            seizure_frequency_per_month=float(row["seizure_frequency_per_month"]),
            age_at_onset=int(row["age_at_onset"]),
            epilepsy_duration_years=float(row["epilepsy_duration_years"]),
            treatment_response=row["treatment_response"],
        )
        session.add(clinical)
    
    session.commit()
    logger.info(f"✓ {len(clinical_df)} registros clínicos carregados")


def load_treatments(session, clinical_df: pd.DataFrame):
    """Carrega tratamentos."""
    logger.info("Carregando tratamentos...")
    
    patients = {p.patient_id: p.id for p in session.query(Patient).all()}
    
    count = 0
    for _, row in clinical_df.iterrows():
        if row["patient_id"] not in patients:
            continue
        
        if pd.notna(row["previous_treatments"]) and row["previous_treatments"]:
            treatments = row["previous_treatments"].split(";")
            
            for treatment_name in treatments:
                treatment = Treatment(
                    patient_id=patients[row["patient_id"]],
                    treatment_name=treatment_name.strip(),
                    response=row["treatment_response"],
                )
                session.add(treatment)
                count += 1
    
    session.commit()
    logger.info(f"✓ {count} tratamentos carregados")


def load_genetic_data(session, genetic_df: pd.DataFrame):
    """Carrega dados genéticos."""
    if genetic_df.empty:
        logger.warning("Sem dados genéticos para carregar")
        return
    
    logger.info("Carregando dados genéticos...")
    
    patients = {p.patient_id: p.id for p in session.query(Patient).all()}
    
    count = 0
    for _, row in genetic_df.iterrows():
        if row["patient_id"] not in patients:
            continue
        
        variant = GeneticVariant(
            patient_id=patients[row["patient_id"]],
            gene=row["gene"],
            variant=row["variant"],
            variant_type=row["variant_type"],
            allele_frequency=float(row["allele_frequency"]) if pd.notna(row["allele_frequency"]) else None,
            clinvar_significance=row["clinvar_significance"] if pd.notna(row["clinvar_significance"]) else None,
        )
        session.add(variant)
        count += 1
    
    session.commit()
    logger.info(f"✓ {count} variantes genéticas carregadas")


def main():
    """Função principal."""
    logger.info("=" * 80)
    logger.info("CARREGANDO DADOS DE EXEMPLO NO BANCO DE DADOS")
    logger.info("=" * 80)
    
    # Configurações
    settings = get_settings()
    
    # Cria engine
    engine = create_engine(settings.database.url)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Cria tabelas
        create_tables(engine)
        
        # Carrega dados
        data_path = Path("data/raw")
        
        # Clinical data
        clinical_df = pd.read_csv(data_path / "clinical_data.csv")
        logger.info(f"Carregados {len(clinical_df)} registros clínicos do CSV")
        
        # Genetic data
        genetic_path = data_path / "genetic_data.csv"
        if genetic_path.exists():
            genetic_df = pd.read_csv(genetic_path)
            logger.info(f"Carregados {len(genetic_df)} registros genéticos do CSV")
        else:
            genetic_df = pd.DataFrame()
        
        # Carrega no banco
        load_patients(session, clinical_df)
        load_clinical_data(session, clinical_df)
        load_treatments(session, clinical_df)
        
        if not genetic_df.empty:
            load_genetic_data(session, genetic_df)
        
        logger.info("=" * 80)
        logger.info("DADOS CARREGADOS COM SUCESSO!")
        logger.info("=" * 80)
        
        # Estatísticas
        logger.info(f"Total de pacientes: {session.query(Patient).count()}")
        logger.info(f"Total de dados clínicos: {session.query(ClinicalData).count()}")
        logger.info(f"Total de tratamentos: {session.query(Treatment).count()}")
        logger.info(f"Total de variantes: {session.query(GeneticVariant).count()}")
        
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        session.rollback()
        raise
    
    finally:
        session.close()


if __name__ == "__main__":
    main()