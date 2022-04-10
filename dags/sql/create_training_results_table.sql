CREATE TABLE IF NOT EXISTS training_results (
    id SERIAL PRIMARY KEY,
    timestamp_utc NUMERIC NOT NULL,
    train_size NUMERIC NOT NULL,
    max_pca_components NUMERIC NOT NULL,
    cv_folds NUMERIC NOT NULL,
    max_iter NUMERIC NOT NULL,
    best_penalty NUMERIC NOT NULL,
    best_pca_components NUMERIC NOT NULL,
    auroc NUMERIC NOT NULL
);