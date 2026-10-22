import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

Path("results").mkdir(exist_ok=True)

df = pd.read_csv("data/german_credit.csv")

X = df.drop(columns=["class"])
y = df["class"].map({1: 0, 2: 1})

categorical = X.select_dtypes(include=["object", "str"]).columns.tolist()
numerical = X.select_dtypes(include=["number"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

cv_results = []
test_results = []

for name, classifier in models.items():
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numerical", StandardScaler(), numerical)
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc"
        }
    )

    cv_results.append({
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1": scores["test_f1"].mean(),
        "ROC-AUC": scores["test_roc_auc"].mean()
    })

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    cost = 5 * fn + fp

    test_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "False Positives": fp,
        "False Negatives": fn,
        "Error Cost": cost
    })

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Good Credit", "Bad Credit"],
        cmap="Blues",
        ax=ax
    )

    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(
        f"results/{name.lower().replace(' ', '_')}_confusion_matrix.png",
        dpi=300
    )
    plt.close(fig)

cv_df = pd.DataFrame(cv_results).round(4)
test_df = pd.DataFrame(test_results).round(4)

cv_df.to_csv("results/cross_validation.csv", index=False)
test_df.to_csv("results/model_comparison.csv", index=False)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nCross-validation results:")
print(cv_df.to_string(index=False))

print("\nTest-set results:")
print(test_df.to_string(index=False))

print("\nBest model by F1:")
print(test_df.loc[test_df["F1"].idxmax(), "Model"])

print("\nBest model by lowest error cost:")
print(test_df.loc[test_df["Error Cost"].idxmin(), "Model"])

print("\nResults saved to results/")