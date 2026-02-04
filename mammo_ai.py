import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
from math import sqrt
from scipy.stats import norm
import time

def measure_inference_time(sample, n_runs=100):
    pipeline = joblib.load("breast_cancer_cfDNA_pipeline.pkl")
    label_encoder = joblib.load("breast_cancer_cfDNA_labelencoder.pkl")

    times = []
    for _ in range(n_runs):
        start = time.time()
        _ = pipeline.predict(pd.DataFrame([sample]))
        end = time.time()
        times.append(end - start)

    avg_time = np.mean(times)
    print(f"Average inference time over {n_runs} runs: {avg_time:.6f} seconds per sample")
    return avg_time


# 🔹 Function to compute margin of error
def compute_margin_of_error(accuracy, n, confidence=0.95):
    z = norm.ppf(1 - (1 - confidence) / 2)
    moe = z * sqrt((accuracy * (1 - accuracy)) / n)
    return moe


def visualize_data(df):
    plt.figure(figsize=(5,4))
    sns.countplot(x="Diagnosis", data=df, palette="Set2")
    plt.title("Diagnosis Distribution (Benign vs Malignant)")
    plt.show()

    plt.figure(figsize=(6,5))
    corr = df.drop("Diagnosis", axis=1).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.show()

    for feature in ["Mean Fragment Size", "Methylation Index", "Mutation Load"]:
        plt.figure(figsize=(6,4))
        sns.kdeplot(data=df, x=feature, hue="Diagnosis", fill=True)
        plt.title(f"{feature} Distribution by Diagnosis")
        plt.show()

    for feature in ["Mean Fragment Size", "Methylation Index", "Mutation Load"]:
        plt.figure(figsize=(6,4))
        sns.boxplot(x="Diagnosis", y=feature, data=df, palette="Set3")
        plt.title(f"{feature} by Diagnosis (Boxplot)")
        plt.show()

    plt.figure(figsize=(6,5))
    sns.scatterplot(
        x="Mean Fragment Size",
        y="Methylation Index",
        hue="Diagnosis",
        data=df,
        palette="coolwarm",
        alpha=0.7
    )
    plt.title("Scatter Plot: Mean Fragment Size vs Methylation Index")
    plt.show()

    plt.figure(figsize=(6,5))
    sns.scatterplot(
        x="Methylation Index",
        y="Mutation Load",
        hue="Diagnosis",
        data=df,
        palette="coolwarm",
        alpha=0.7
    )
    plt.title("Scatter Plot: Methylation Index vs Mutation Load")
    plt.show()

    plt.figure(figsize=(6,5))
    sns.scatterplot(
        x="Mean Fragment Size",
        y="Mutation Load",
        hue="Diagnosis",
        data=df,
        palette="coolwarm",
        alpha=0.7
    )

    # Pairplot (all features vs each other)
    sns.pairplot(df, hue="Diagnosis", palette="husl", diag_kind="kde", corner=True)
    plt.suptitle("Pairplot of All Features by Diagnosis", y=1.02)
    plt.show()


def train_and_save_model(csv_file):
    df = pd.read_csv(csv_file)

    visualize_data(df)

    # Encode diagnosis labels
    label_encoder = LabelEncoder()
    df["Diagnosis"] = label_encoder.fit_transform(df["Diagnosis"])

    print("Encoded classes:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
    print("Value counts:", df["Diagnosis"].value_counts())

    X = df.drop("Diagnosis", axis=1)
    y = df["Diagnosis"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))
    ])

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Predictions
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("\n--- Results ---")
    print("Accuracy:", acc)
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))

    # 🔹 Compute and display Margin of Error for Accuracy
    moe = compute_margin_of_error(acc, len(y_test), confidence=0.95)
    lower = acc - moe
    upper = acc + moe
    print(f"Margin of Error (95% CI): ±{moe:.3f}")
    print(f"Confidence Interval for Accuracy: [{lower:.3f}, {upper:.3f}]")

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    print("Confusion Matrix Table:\n", cm)

    # ROC Curve + AUC
    y_probs = pipeline.predict_proba(X_test)[:, 1]  # Probability of class=1 (Malignant)
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

    # Manual Trapezoidal AUC calculation
    auc_trap = 0.0
    for i in range(len(fpr)-1):
        auc_trap += (fpr[i+1] - fpr[i]) * (tpr[i+1] + tpr[i]) / 2
    print(f"Trapezoidal AUC: {auc_trap:.3f} (sklearn AUC: {auc_score:.3f})")

    # Feature importance
    model = pipeline.named_steps["model"]
    feature_importances = model.feature_importances_
    features = X.columns
    sorted_idx = np.argsort(feature_importances)[::-1]

    plt.figure(figsize=(6,4))
    sns.barplot(x=feature_importances[sorted_idx], y=features[sorted_idx], palette="viridis")
    plt.title("Feature Importance (Random Forest)")
    plt.show()

    # Cross-validation
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    print("Cross-validation Accuracy:", scores.mean())

    # Save pipeline + label encoder
    joblib.dump(pipeline, "breast_cancer_cfDNA_pipeline.pkl")
    joblib.dump(label_encoder, "breast_cancer_cfDNA_labelencoder.pkl")


def predict_new_sample(new_data_dict):
    pipeline = joblib.load("breast_cancer_cfDNA_pipeline.pkl")
    label_encoder = joblib.load("breast_cancer_cfDNA_labelencoder.pkl")

    new_data = pd.DataFrame([new_data_dict])

    pred = pipeline.predict(new_data)[0]
    pred_label = label_encoder.inverse_transform([pred])[0]

    pred_probs = pipeline.predict_proba(new_data)[0]
    print("Prediction:", pred_label)
    print("Class probabilities:", dict(zip(label_encoder.classes_, pred_probs)))

    return pred_label


if __name__ == "__main__":
    train_and_save_model("C:/Users/harro/mammo/Breast_cancer_cfDNA_10k.csv")

    # Example usage of new prediction function
    new_sample = {
        "Mean Fragment Size": 166,
        "Methylation Index": 0.72,
        "Mutation Load": 1.564
    }

    predict_new_sample(new_sample)
    measure_inference_time(new_sample, n_runs=100)