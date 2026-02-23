import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def evaluate(df: pd.DataFrame, features: list):
    X = df[features].fillna(0)
    y = df["team_won"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=300)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    return {
        "features": features,
        "n_features": len(features),
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def main():
    leaky_df = pd.read_csv("data/processed/match_features.csv")
    strict_df = pd.read_csv("data/processed/match_features_noleak.csv")

    all_feature_cols = [
        c for c in leaky_df.columns if c not in ["team_won", "match_id", "team_id"]
    ]
    strict_cols = [c for c in ["rank_diff"] if c in strict_df.columns]

    leaked = evaluate(leaky_df, all_feature_cols)
    strict = evaluate(strict_df, strict_cols)

    print("\n" + "=" * 80)
    print("MATCH OUTCOME LEAKAGE CHECK")
    print("=" * 80)

    print("\nLeaky setting (uses in-game final differentials):")
    print(f"- features ({leaked['n_features']}): {leaked['features']}")
    print(f"- accuracy: {leaked['accuracy']:.4f}")
    print(f"- roc_auc : {leaked['roc_auc']:.4f}")

    print("\nStrict no-leakage setting (pre-match safe):")
    print(f"- features ({strict['n_features']}): {strict['features']}")
    print(f"- accuracy: {strict['accuracy']:.4f}")
    print(f"- roc_auc : {strict['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
