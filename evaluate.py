import pandas as pd
import numpy as np

# Files
GROUND_TRUTH = "data/ground_truth.csv"

CSS_FILE = "outputs/css_output.csv"
REGEX_FILE = "outputs/regex_output.csv"
LLM_FILE = "outputs/llm_output.csv"

FIELDS = [
    "name",
    "birth_date",
    "birth_place",
    "nationality",
    "occupation",
    "awards",
    "website"
]


# ---------------------------
# Helper functions
# ---------------------------
import re

def normalize(x):

    if pd.isna(x):
        return ""

    x = str(x).lower()

    # remove punctuation
    x = re.sub(r"[^\w\s]", "", x)

    # normalize country names
    replacements = {
        "usa": "united states",
        "us": "united states",
        "uk": "united kingdom"
    }

    for k, v in replacements.items():
        x = x.replace(k, v)

    return " ".join(x.split())



def load_and_align(pred_file, gt_file):

    pred = pd.read_csv(pred_file)
    gt = pd.read_csv(gt_file)

    pred.set_index("title", inplace=True)
    gt.set_index("title", inplace=True)

    common = pred.index.intersection(gt.index)

    pred = pred.loc[common]
    gt = gt.loc[common]

    return pred, gt


# ---------------------------
# Metric 1: Exact Match Accuracy
# ---------------------------

def exact_match_accuracy(pred, gt):

    correct = 0
    total = 0

    for field in FIELDS:

        p = pred[field].apply(normalize)
        g = gt[field].apply(normalize)

        for i in range(len(p)):

            if p.iloc[i] == g.iloc[i]:
                correct += 1

            total += 1

    return correct / total

# ---------------------------
# Metric 1B: Partial Match Accuracy
# ---------------------------

def partial_match_accuracy(pred, gt):

    correct = 0
    total = 0

    for field in FIELDS:

        p = pred[field].apply(normalize)
        g = gt[field].apply(normalize)

        for i in range(len(p)):

            pred_val = p.iloc[i]
            gt_val = g.iloc[i]

            # partial match condition
            if pred_val in gt_val or gt_val in pred_val:
                correct += 1

            total += 1

    return correct / total if total > 0 else 0


# ---------------------------
# Metric 2: Precision
# ---------------------------

def precision(pred, gt):

    true_positive = 0
    predicted_positive = 0

    for field in FIELDS:

        p = pred[field].apply(normalize)
        g = gt[field].apply(normalize)

        true_positive += ((p == g) & (p != "")).sum()
        predicted_positive += (p != "").sum()

    if predicted_positive == 0:
        return 0

    return true_positive / predicted_positive


# ---------------------------
# Metric 3: Recall
# ---------------------------

def recall(pred, gt):

    true_positive = 0
    actual_positive = 0

    for field in FIELDS:

        p = pred[field].apply(normalize)
        g = gt[field].apply(normalize)

        true_positive += ((p == g) & (g != "")).sum()
        actual_positive += (g != "").sum()

    if actual_positive == 0:
        return 0

    return true_positive / actual_positive


# ---------------------------
# Metric 4: F1 Score
# ---------------------------

def f1_score(p, r):

    if (p + r) == 0:
        return 0

    return 2 * p * r / (p + r)


# ---------------------------
# Metric 5: Schema Validity Rate
# ---------------------------

def schema_validity(pred):

    valid = 0

    for _, row in pred.iterrows():

        if all(field in row.index for field in FIELDS):
            valid += 1

    return valid / len(pred)


# ---------------------------
# Metric 6: Hallucination Rate
# ---------------------------

def hallucination_rate(pred, gt):

    hallucinations = 0
    opportunities = 0

    for field in FIELDS:

        p = pred[field].apply(normalize)
        g = gt[field].apply(normalize)

        hallucinations += ((p != "") & (g == "")).sum()
        opportunities += (g == "").sum()

    if opportunities == 0:
        return 0

    return hallucinations / opportunities


# ---------------------------
# Evaluation printer
# ---------------------------

def evaluate_method(name, file_path):

    pred, gt = load_and_align(file_path, GROUND_TRUTH)

    acc = exact_match_accuracy(pred, gt)
    partial_acc = partial_match_accuracy(pred, gt)


    prec = precision(pred, gt)

    rec = recall(pred, gt)

    f1 = f1_score(prec, rec)

    validity = schema_validity(pred)

    halluc = hallucination_rate(pred, gt)

    print(f"\n===== {name} RESULTS =====")

    print(f"Accuracy: {acc:.4f}")
    print(f"Partial Accuracy: {partial_acc:.4f}")

    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Schema Validity: {validity:.4f}")
    print(f"Hallucination Rate: {halluc:.4f}")


# ---------------------------
# Main
# ---------------------------

def main():

    print("\nStarting evaluation...\n")

    evaluate_method("CSS", CSS_FILE)

    evaluate_method("REGEX", REGEX_FILE)

    evaluate_method("LLM", LLM_FILE)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
