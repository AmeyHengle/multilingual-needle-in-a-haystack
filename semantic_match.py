import multiprocessing as mp
import os

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

tqdm.pandas()
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"
model = SentenceTransformer("all-mpnet-base-v2", device=device)


def semantic_match(row: tuple):
    y_true = row[0]
    y_pred = row[1]

    if isinstance(y_pred, list):
        print(y_pred)
    y_pred = str(y_pred)

    embeddings1 = model.encode(y_true)
    embeddings2 = model.encode([y_pred])
    similarities = model.similarity(embeddings1, embeddings2)
    return float(similarities.max())


def process_df(
    df: pd.DataFrame, true_col, pred_col, target_col, num_workers=32
) -> pd.DataFrame:

    # Apply the function using multiprocessing
    def apply_multiprocessing(df, func, true_col, pred_col, num_workers):
        with mp.Pool(num_workers) as pool:
            result = list(
                tqdm(
                    pool.imap(func, [row for row in zip(df[true_col], df[pred_col])]),
                    total=len(df),
                )
            )
        return result

    # Number of workers
    num_workers = mp.cpu_count()

    # Apply the function to the DataFrame
    df[target_col] = apply_multiprocessing(
        df, semantic_match, true_col, pred_col, num_workers
    )

    return df
