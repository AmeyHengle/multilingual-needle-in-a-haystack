import argparse
import os

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils import clean_str, remove_punctuation

# Argument parser setup
parser = argparse.ArgumentParser(
    description="Compute accuracy metrics for predictions."
)
parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help="Path to the input CSV file containing predictions.",
)
parser.add_argument(
    "--output_file",
    type=str,
    required=True,
    help="Path to save the output CSV file with computed accuracy.",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="all-mpnet-base-v2",
    help="Name of the SentenceTransformer model.",
)
parser.add_argument(
    "--similarity_threshold",
    type=float,
    default=0.95,
    help="Similarity threshold for semantic match.",
)
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu", "cuda"],
    default="cuda",
    help="Device to use for computation.",
)
parser.add_argument(
    "--question_lang",
    type=str,
    default="en",
    help="Language of the questions to filter by.",
)

args = parser.parse_args()

# Set up device and model
device = args.device if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(args.model_name, device=device)


def clean_text(text):
    if isinstance(text, float) or len(text) < 1:
        text = "<Empty>"

    if not isinstance(text, str):
        text = str(text)

    text = clean_str(text)
    text = remove_punctuation(text)
    text = text.lower().strip()
    text = text.replace("-", "")

    return text


def exact_match_binary(y_true, y_pred) -> tuple:
    y_pred = str(y_pred)

    if not isinstance(y_true, list):
        if y_true in y_pred:
            return 1, 1
        else:
            y_true = y_true.split()
            y_pred = y_pred.split()
            acc_prob = len(set(y_true).intersection(set(y_pred))) / len(y_true)
            acc_abs = 0 if acc_prob < 1 else 1
            return acc_abs, acc_prob
    else:
        acc_abs_list = []
        acc_prob_list = []

        for x in y_true:
            x = clean_text(x)
            if x in y_pred:
                return 1, 1
            else:
                x = x.split()
                y = y_pred.split()
                acc_prob = len(set(x).intersection(set(y))) / len(x)
                acc_abs = 0 if acc_prob < 1 else 1
                acc_abs_list.append(acc_abs)
                acc_prob_list.append(acc_prob)

        return max(acc_abs_list), max(acc_prob_list)


def semantic_match(y_true: list, y_pred: str):
    y_pred = str(y_pred)

    embeddings1 = model.encode(y_true)
    embeddings2 = model.encode([y_pred])
    similarities = model.similarity(embeddings1, embeddings2)
    return float(similarities.max())


def compute_accuracy(df, similarity_threshold=0.95):
    if "answer_text_format_multilingual" not in df:
        raise Exception("Groundtruth column absent")
    if "prediction" not in df:
        raise Exception("Prediction column absent")

    if "accuracy_prob" in df.columns:
        df = df.drop(columns=["accuracy_binary", "accuracy_prob"])

    df["answer_text_format_multilingual"] = df[
        "answer_text_format_multilingual"
    ].progress_apply(lambda x: eval(x))

    df["prediction"] = df["prediction"].progress_apply(lambda x: clean_text(x))

    df["accuracy_exact_match"] = df.progress_apply(
        lambda row: exact_match_binary(
            y_true=row["answer_text_format_multilingual"], y_pred=row["prediction"]
        )[0],
        axis=1,
    )

    df["accuracy_semantic_match"] = df.progress_apply(
        lambda row: (
            semantic_match(
                y_true=row["answer_text_format_multilingual"],
                y_pred=row["prediction"],
            )
            if row["accuracy_exact_match"] != 1
            else 1
        ),
        axis=1,
    )

    df["accuracy_semantic_match"] = df["accuracy_semantic_match"].apply(
        lambda x: 1 if x >= similarity_threshold else 0
    )

    if "predictions_translated" in df.columns:
        df["predictions_translated"] = df["predictions_translated"].progress_apply(
            lambda x: clean_text(x)
        )

        df["accuracy_exact_match_(y_pred_translated)"] = df.progress_apply(
            lambda row: exact_match_binary(
                y_true=row["answer_text_format_multilingual"],
                y_pred=row["predictions_translated"],
            )[0],
            axis=1,
        )

        df["accuracy_semantic_match_(y_pred_translated)"] = df.progress_apply(
            lambda row: (
                semantic_match(
                    y_true=row["answer_text_format_multilingual"],
                    y_pred=row["predictions_translated"],
                )
                if row["accuracy_exact_match_(y_pred_translated)"] != 1
                else 1
            ),
            axis=1,
        )

        df["accuracy_semantic_match_(y_pred_translated)"] = df[
            "accuracy_semantic_match_(y_pred_translated)"
        ].apply(lambda x: 1 if x >= similarity_threshold else 0)

    accuracy_cols = [col for col in df.columns if "accuracy_" in col]
    df["accuracy"] = df[accuracy_cols].max(axis=1)

    return df


def main():
    df_pred = pd.read_csv(args.input_file)
    df_pred = df_pred[df_pred["question_lang"] == args.question_lang]

    print(df_pred.shape)
    assert df_pred.shape[0] in [
        16800,
        4900,
        117600,
        700,
    ], "Unexpected number of rows in the input data. Please check your predictions."

    df_pred = compute_accuracy(df_pred, similarity_threshold=args.similarity_threshold)
    df_pred.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
