from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt


def load_ground_truth(file_path: str) -> Tuple[List[int], List[str]]:
    """Load the ground truth labels and return binary labels and texts.

    The ground truth CSV file must have a column named ``label`` with
    values indicating whether a tweet is a rumour (e.g. 'rumor') or not.

    Parameters
    ----------
    file_path: str
        Path to the ground truth CSV file.

    Returns
    -------
    Tuple[List[int], List[str]]
        A tuple containing two lists: binary labels (1 for rumour,
        0 for non-rumour) and the corresponding tweet texts.
    """
    df = pd.read_csv(file_path)
    # Strip whitespace from column names to avoid issues with trailing spaces
    df.columns = [c.strip() for c in df.columns]
    
    labels = []
    texts = []
    for _, row in df.iterrows():
        label = row.get("label")
        tweet = row.get("tweet")
        if pd.isna(label) or pd.isna(tweet):
            continue
        label_str = str(label).strip().lower()
        # The label may be 'rumor', 'non-rumor', 'true', 'false', etc.
        is_rumor = label_str in {"rumor", "true", "1", "yes"}
        labels.append(1 if is_rumor else 0)
        texts.append(str(tweet))
    
    return labels, texts


def load_predictions(file_path: str, all_texts: List[str]) -> List[int]:
    """Load predictions from the ``rumors.json`` file.

    Parameters
    ----------
    file_path: str
        Path to the JSON file containing rumour predictions. This file
        should be an array of objects with keys ``tweet`` and ``label``.
    all_texts: List[str]
        The list of tweets from the ground truth dataset. Used to
        generate a 0/1 label array matching the order of ground truth.

    Returns
    -------
    List[int]
        Binary predictions aligned with the ground truth texts (1 for
        rumour, 0 for non-rumour).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        preds = json.load(f)
    
    # Build a set of tweets labelled as rumour by the model
    rumor_tweets = {p["tweet"] for p in preds if p.get("label") == "rumor"}
    predictions = [1 if text in rumor_tweets else 0 for text in all_texts]
    
    return predictions


def evaluate(ground_truth_csv: str, predictions_json: str, output_dir: str = "evaluation") -> None:
    """Compute evaluation metrics and produce plots.

    Parameters
    ----------
    ground_truth_csv: str
        Path to the ground truth CSV file.
    predictions_json: str
        Path to the JSON file containing model predictions.
    output_dir: str
        Directory to save evaluation reports and plots.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load ground truth and predictions
    y_true, texts = load_ground_truth(ground_truth_csv)
    y_pred = load_predictions(predictions_json, texts)
    
    # Basic classification report
    unique_true = sorted(set(y_true))
    if not unique_true:
        # No valid labels found; skip classification report
        report = "No ground truth labels available."
    else:
        target_names = ["non-rumor" if label == 0 else "rumor" for label in unique_true]
        report = classification_report(y_true, y_pred, labels=unique_true, target_names=target_names)
    
    with open(Path(output_dir) / "classification_report.txt", "w") as f:
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["non-rumor", "rumor"])
    ax.set_yticklabels(["rumor","non-rumor"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "confusion_matrix.png")
    plt.close(fig)
    
    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    fig.savefig(Path(output_dir) / "roc_curve.png")
    plt.close(fig)
    
    print("Evaluation complete. Reports saved to", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rumour classification performance")
    parser.add_argument("--ground-truth", required=True, help="Path to the ground truth CSV file")
    parser.add_argument("--predictions", default="rumors.json", help="Path to the predictions JSON file")
    parser.add_argument("--output", default="evaluation", help="Directory to save evaluation outputs")
    args = parser.parse_args()
    evaluate(args.ground_truth, args.predictions, args.output)


if __name__ == "__main__":
    main()
