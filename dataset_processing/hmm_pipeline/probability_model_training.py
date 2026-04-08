import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, classification_report

from xgboost import XGBClassifier

from dataset_processing.hmm_pipeline.hmm_data_processing import compute_features


def _build_model(model_type, model_config, random_seed=42):
    if model_type == "logreg":
        cfg = model_config["logreg"]
        return LogisticRegression(
            max_iter=cfg["max_iter"],
            C=cfg["C"],
            solver=cfg["solver"],
            random_state=random_seed
        )

    elif model_type == "xgboost":
        cfg = model_config["xgboost"]
        return XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            eval_metric=cfg["eval_metric"],
            random_state=random_seed,
            use_label_encoder=False
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _filter_pairs(pairs_df, allowed_genomes):
    allowed_genomes = set(allowed_genomes)
    return pairs_df[
        pairs_df["genome1"].isin(allowed_genomes) &
        pairs_df["genome2"].isin(allowed_genomes)
    ].copy()


def _describe_probs(name, probs):
    lines = []
    lines.append(f"{name} probability distribution:")
    lines.append(f"  min: {np.min(probs):.6f}")
    lines.append(f"  max: {np.max(probs):.6f}")
    lines.append(f"  mean: {np.mean(probs):.6f}")
    lines.append(f"  std: {np.std(probs):.6f}")

    percentiles = np.percentile(probs, [1, 5, 25, 50, 75, 95, 99])
    lines.append(f"  percentiles (1,5,25,50,75,95,99): {percentiles.tolist()}")
    lines.append(f"  % < 0.01: {np.mean(probs < 0.01):.6f}")
    lines.append(f"  % > 0.99: {np.mean(probs > 0.99):.6f}")
    lines.append(f"  % in [0.4, 0.6]: {np.mean((probs >= 0.4) & (probs <= 0.6)):.6f}")

    return "\n".join(lines)


def train_probability_model(
    df,
    pairs_df,
    taxon_rank,
    model_type,
    model_config,
    training_config,
    feature_config,
    random_seed=42
):
    genome_dict = {
        row["Accession"]: set(row["hmms_hits"])
        for _, row in df.iterrows()
    }

    split_cfg = training_config["genome_split"]
    train_frac = split_cfg["train_fraction"]
    val_frac = split_cfg["val_fraction"]
    test_frac = split_cfg["test_fraction"]

    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "Train/val/test fractions must sum to 1."

    all_genomes = list(genome_dict.keys())

    train_genomes, temp_genomes = train_test_split(
        all_genomes,
        test_size=(1 - train_frac),
        random_state=random_seed
    )

    relative_test_size = test_frac / (val_frac + test_frac)

    val_genomes, test_genomes = train_test_split(
        temp_genomes,
        test_size=relative_test_size,
        random_state=random_seed
    )

    train_pairs = _filter_pairs(pairs_df, train_genomes)
    val_pairs = _filter_pairs(pairs_df, val_genomes)
    test_pairs = _filter_pairs(pairs_df, test_genomes)

    X_train, y_train, _, _ = compute_features(train_pairs, genome_dict, feature_config)
    X_val, y_val, _, _ = compute_features(val_pairs, genome_dict, feature_config)
    X_test, y_test, _, _ = compute_features(test_pairs, genome_dict, feature_config)

    model = _build_model(model_type, model_config, random_seed=random_seed)
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    val_preds = (val_probs >= 0.5).astype(int)
    test_preds = (test_probs >= 0.5).astype(int)

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PAIRWISE MODEL EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Model: {model_type}")
    report_lines.append(f"Taxonomic rank: {taxon_rank}")
    report_lines.append("")
    report_lines.append(f"Pairs count -> Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
    report_lines.append(f"Feature matrix shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    report_lines.append("")
    report_lines.append(f"Validation LogLoss: {log_loss(y_val, val_probs):.6f}")
    report_lines.append(f"Validation Accuracy: {accuracy_score(y_val, val_preds):.6f}")
    report_lines.append(f"Test LogLoss: {log_loss(y_test, test_probs):.6f}")
    report_lines.append(f"Test Accuracy: {accuracy_score(y_test, test_preds):.6f}")
    report_lines.append("")
    report_lines.append("Validation classification report:")
    report_lines.append(classification_report(y_val, val_preds, digits=4))
    report_lines.append("")
    report_lines.append("Test classification report:")
    report_lines.append(classification_report(y_test, test_preds, digits=4))
    report_lines.append("")
    report_lines.append(_describe_probs("VALIDATION", val_probs))
    report_lines.append("")
    report_lines.append(_describe_probs("TEST", test_probs))

    report_str = "\n".join(report_lines)

    # Retrain on full pair dataset
    X_all, y_all, _, _ = compute_features(pairs_df, genome_dict, feature_config)
    final_model = _build_model(model_type, model_config, random_seed=random_seed)
    final_model.fit(X_all, y_all)

    return report_str, final_model