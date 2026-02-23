import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.shared.data_loader import DataLoader


def main():
    X, y, feature_names = DataLoader.prepare_match_features(strict_no_leakage=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)

    model_dir = Path("models/4_match_outcome_predictor/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "match_outcome_model_strict.pkl")
    joblib.dump(scaler, model_dir / "scaler_strict.pkl")

    metadata = {
        "feature_names": feature_names,
        "scaler_path": str((model_dir / "scaler_strict.pkl").resolve()),
        "model_type": type(model).__name__,
        "target": "team_won",
        "n_features": len(feature_names),
        "strict_no_leakage": True,
        "metrics": {
            "accuracy": float(acc),
            "roc_auc": float(auc),
        },
    }

    with open(model_dir / "metadata_strict.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSTRICT MATCH OUTCOME RETRAIN COMPLETE")
    print(f"features: {feature_names}")
    print(f"accuracy: {acc:.4f}")
    print(f"roc_auc : {auc:.4f}")
    print(f"saved: {model_dir / 'match_outcome_model_strict.pkl'}")
    print(f"saved: {model_dir / 'metadata_strict.json'}")


if __name__ == "__main__":
    main()
