import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

from dataset_processing import DATASET_PICKLE_PATH, get_intermediate_output_path

# optional import
from xgboost import XGBClassifier

from dataset_processing.hmm_data_processing import build_name_to_hmm_set_dict, compute_features


def build_model(model_type):
    if model_type == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))
        ])

    elif model_type == "xgboost":
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss"
        )

    else:
        raise ValueError(f"Unknown MODEL_TYPE: {model_type}")


def filter_pairs(pairs_df, allowed_genomes):
    return pairs_df[
        pairs_df['genome1'].isin(allowed_genomes) &
        pairs_df['genome2'].isin(allowed_genomes)
        ]


def describe_probs(name, probs):
    print(f"\n{name} distribution:")
    print("  min:", np.min(probs))
    print("  max:", np.max(probs))
    print("  mean:", np.mean(probs))
    print("  std:", np.std(probs))

    # percentiles give a better picture than just mean/std
    percentiles = np.percentile(probs, [1, 5, 25, 50, 75, 95, 99])
    print("  percentiles (1,5,25,50,75,95,99):", percentiles)

    # how confident the model is
    print("  % < 0.01:", np.mean(probs < 0.01))
    print("  % > 0.99:", np.mean(probs > 0.99))
    print("  % in [0.4, 0.6]:", np.mean((probs >= 0.4) & (probs <= 0.6)))


def main():
    TAXONOMICAL_RANK = 'Family'
    MODEL_TYPE = "xgboost"

    input_path = get_intermediate_output_path(TAXONOMICAL_RANK, 'hmm', 'pairs')
    model_output_path = get_intermediate_output_path(
        TAXONOMICAL_RANK,
        'hmm',
        MODEL_TYPE
    )

    print("Loading data...")
    pairs_df = pd.read_csv(input_path)
    df = pd.read_pickle(DATASET_PICKLE_PATH)

    print("Building genome dictionary...")
    genome_dict = build_name_to_hmm_set_dict(df)

    # -----------------------------
    # Split by genomes (NO LEAKAGE)
    # -----------------------------

    print("Splitting genomes...")
    all_genomes = list(genome_dict.keys())

    train_genomes, temp_genomes = train_test_split(
        all_genomes, test_size=0.3, random_state=42
    )

    val_genomes, test_genomes = train_test_split(
        temp_genomes, test_size=0.5, random_state=42
    )

    print("Filtering pairs...")
    train_pairs = filter_pairs(pairs_df, train_genomes)
    val_pairs = filter_pairs(pairs_df, val_genomes)
    test_pairs = filter_pairs(pairs_df, test_genomes)

    print(f"Pairs count -> Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

    # -----------------------------
    # Compute features per split
    # -----------------------------
    print("Computing features...")
    X_train, y_train, _, _ = compute_features(train_pairs, genome_dict)
    X_val, y_val, _, _ = compute_features(val_pairs, genome_dict)
    X_test, y_test, _, _ = compute_features(test_pairs, genome_dict)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # -----------------------------
    # Train model
    # -----------------------------
    print("Building model...")
    model = build_model(MODEL_TYPE)

    print("Training model only on train set...")
    model.fit(X_train, y_train)

    # -----------------------------
    # Evaluation
    # -----------------------------
    if hasattr(model, "predict_proba"):
        from sklearn.metrics import log_loss

        val_probs = model.predict_proba(X_val)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]
        print("Val LogLoss:", log_loss(y_val, val_probs))
        print("Test LogLoss:", log_loss(y_test, test_probs))
        describe_probs("TEST", test_probs)

    print("Training model on the entire set...")
    X, y = compute_features(pairs_df, genome_dict)
    model.fit(X, y)

    print("Saving model...")
    joblib.dump(model, model_output_path)

    print("Done.")


if __name__ == "__main__":
    main()
