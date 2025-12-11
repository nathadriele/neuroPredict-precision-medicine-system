-- ============================================================================
-- Script de inicialização do banco de dados PostgreSQL para NeuroPredict
-- ============================================================================

-- Cria extensões necessárias
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ============================================================================
-- Tabelas Principais
-- ============================================================================

-- Tabela de Pacientes
CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id VARCHAR(50) UNIQUE NOT NULL,
    age INTEGER NOT NULL CHECK (age >= 0 AND age <= 120),
    sex VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabela de Dados Clínicos
CREATE TABLE IF NOT EXISTS clinical_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    seizure_type VARCHAR(100) NOT NULL,
    seizure_frequency_per_month DECIMAL(10, 2) NOT NULL CHECK (seizure_frequency_per_month >= 0),
    age_at_onset INTEGER NOT NULL CHECK (age_at_onset >= 0),
    epilepsy_duration_years DECIMAL(10, 2) NOT NULL CHECK (epilepsy_duration_years >= 0),
    treatment_response VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabela de Tratamentos
CREATE TABLE IF NOT EXISTS treatments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    treatment_name VARCHAR(100) NOT NULL,
    start_date DATE,
    end_date DATE,
    dosage VARCHAR(50),
    response VARCHAR(50),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabela de Variantes Genéticas
CREATE TABLE IF NOT EXISTS genetic_variants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    gene VARCHAR(50) NOT NULL,
    variant VARCHAR(200) NOT NULL,
    variant_type VARCHAR(50),
    allele_frequency DECIMAL(10, 6),
    clinvar_significance VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabela de Predições
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    model_version VARCHAR(50) NOT NULL,
    predicted_treatment VARCHAR(100) NOT NULL,
    response_probability DECIMAL(5, 4) NOT NULL CHECK (response_probability >= 0 AND response_probability <= 1),
    confidence DECIMAL(5, 4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    explanation JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabela de Logs de API
CREATE TABLE IF NOT EXISTS api_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint VARCHAR(200) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER,
    user_id VARCHAR(100),
    ip_address INET,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Índices para Performance
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_patients_patient_id ON patients(patient_id);
CREATE INDEX IF NOT EXISTS idx_clinical_patient_id ON clinical_data(patient_id);
CREATE INDEX IF NOT EXISTS idx_treatments_patient_id ON treatments(patient_id);
CREATE INDEX IF NOT EXISTS idx_genetic_patient_id ON genetic_variants(patient_id);
CREATE INDEX IF NOT EXISTS idx_genetic_gene ON genetic_variants(gene);
CREATE INDEX IF NOT EXISTS idx_predictions_patient_id ON predictions(patient_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_api_logs_created_at ON api_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_api_logs_endpoint ON api_logs(endpoint);

-- Índice para busca textual
CREATE INDEX IF NOT EXISTS idx_genetic_gene_trgm ON genetic_variants USING gin(gene gin_trgm_ops);

-- ============================================================================
-- Triggers para Updated_at
-- ============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_patients_updated_at
    BEFORE UPDATE ON patients
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_clinical_updated_at
    BEFORE UPDATE ON clinical_data
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Views Úteis
-- ============================================================================

-- View com dados completos do paciente
CREATE OR REPLACE VIEW patient_complete_view AS
SELECT 
    p.id,
    p.patient_id,
    p.age,
    p.sex,
    cd.seizure_type,
    cd.seizure_frequency_per_month,
    cd.age_at_onset,
    cd.epilepsy_duration_years,
    cd.treatment_response,
    COUNT(DISTINCT t.id) as total_treatments,
    COUNT(DISTINCT gv.id) as total_genetic_variants,
    p.created_at,
    p.updated_at
FROM patients p
LEFT JOIN clinical_data cd ON p.id = cd.patient_id
LEFT JOIN treatments t ON p.id = t.patient_id
LEFT JOIN genetic_variants gv ON p.id = gv.patient_id
GROUP BY p.id, p.patient_id, p.age, p.sex, cd.seizure_type, 
         cd.seizure_frequency_per_month, cd.age_at_onset, 
         cd.epilepsy_duration_years, cd.treatment_response,
         p.created_at, p.updated_at;

-- View de estatísticas
CREATE OR REPLACE VIEW prediction_stats AS
SELECT 
    predicted_treatment,
    COUNT(*) as total_predictions,
    AVG(response_probability) as avg_probability,
    AVG(confidence) as avg_confidence
FROM predictions
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY predicted_treatment
ORDER BY total_predictions DESC;

-- ============================================================================
-- Funções Úteis
-- ============================================================================

-- Função para buscar pacientes similares
CREATE OR REPLACE FUNCTION find_similar_patients(
    p_patient_id UUID,
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    similar_patient_id UUID,
    similarity_score DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p2.id as similar_patient_id,
        (
            CASE WHEN cd1.seizure_type = cd2.seizure_type THEN 1.0 ELSE 0.0 END +
            CASE WHEN ABS(cd1.age_at_onset - cd2.age_at_onset) <= 5 THEN 0.5 ELSE 0.0 END +
            CASE WHEN ABS(cd1.seizure_frequency_per_month - cd2.seizure_frequency_per_month) <= 2 THEN 0.5 ELSE 0.0 END
        ) as similarity_score
    FROM patients p1
    JOIN clinical_data cd1 ON p1.id = cd1.patient_id
    JOIN clinical_data cd2 ON cd1.id != cd2.id
    JOIN patients p2 ON cd2.patient_id = p2.id
    WHERE p1.id = p_patient_id
    AND p2.id != p_patient_id
    ORDER BY similarity_score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Database para MLFlow
-- ============================================================================

CREATE DATABASE mlflow;

-- ============================================================================
-- Permissões
-- ============================================================================

-- Grant permissions para aplicação
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO postgres;

-- ============================================================================
-- Dados de Exemplo (Opcional)
-- ============================================================================

-- Inserir alguns dados de exemplo para testes
-- Comentar em produção

/*
INSERT INTO patients (patient_id, age, sex) VALUES
    ('PAT0001', 35, 'M'),
    ('PAT0002', 28, 'F'),
    ('PAT0003', 42, 'M');

INSERT INTO clinical_data (patient_id, seizure_type, seizure_frequency_per_month, 
                           age_at_onset, epilepsy_duration_years, treatment_response)
SELECT id, 'focal_aware', 4.5, 12, 23.0, 'partial_responder'
FROM patients WHERE patient_id = 'PAT0001';
*/

-- ============================================================================
-- Fim do Script
-- ============================================================================

-- Mensagem de sucesso
DO $$
BEGIN
    RAISE NOTICE 'Database initialization completed successfully!';
END $$;