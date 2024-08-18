import ast
import glob
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from utils import (dashed_line, fname_to_experiment_dict, load_json,
                   postprocess_llm_pred)


def analysis2_context_lang(df_pred, acc_col):
    # Question1: Effect of changing language of needle

    """
    Here, we will analyse effect of controlling language of relevant information (needle passage) on the downstream multilingual QA task.
    The search criteria is defined as follows:
    1. Question language is to be kept a constant here, i.e., question_lang == noise_type or question_lang == en
    2. Needle position is to be ignored here, i.e., report accuracy as average of needle_position == start, middle, or end.
    3. Noise type is to be kept constant here, i.e., noise_type == question_lang
    4. Context lang (or lang of needle) is the only variable here, and can take any value from - [en, hi, es, du, vi, zh, ar]
    """
    df_pred = df_pred[df_pred["question_lang"] == "en"]
    df_query = df_pred[(df_pred["question_lang"] == df_pred["noise_type"])]

    print(f"Total query data: {df_query.shape[0]}")
    df_query.head(2)

    grouped_df = df_query.groupby(["context_lang"])[[acc_col]].agg(list).reset_index()
    print(f"Total permutations: {grouped_df.shape[0]}")

    # Calculate mean accuracy
    grouped_df["accuracy_mean"] = grouped_df[acc_col].apply(lambda x: np.mean(x))
    grouped_df["accuracy_std"] = grouped_df[acc_col].apply(lambda x: np.std(x))

    grouped_df.head(10)

    # Calculate the mean across each row
    x = grouped_df["context_lang"].tolist()
    y = grouped_df["accuracy_mean"].tolist()
    x_reorder = ["en", "de", "es", "zh", "vi", "hi", "ar"]
    length = range(len(x_reorder))
    y_reorder = [y[x.index(i)] for i in x_reorder]
    return x_reorder, y_reorder, grouped_df


def analysis2_noise_type(df_pred, acc_col):
    # Question 2: Effect of changing langauge of distractor passages.

    """
    Here, we will analyse effect of controlling language of distractor documents on the mQA task.
    The search criteria is defined as follows:
    1. Question language is to be kept a constant here, i.e., question_lang == context_lang or question_lang == en
    2. Needle position is to be ignored here, i.e., report accuracy as average of needle_position == start, middle, or end.
    3. Context lang (or lang of needle) is to be kept constant here, i.e., context_lang == question_lang
    4. Noise type is the only variable in this experiment, and can take any value from - [en, hi, es, du, vi, zh, ar, or multilingual]
    """

    df_pred = df_pred[df_pred["question_lang"] == "en"]
    df_query = df_pred[(df_pred["question_lang"] == df_pred["context_lang"])]

    print(f"Total query data: {df_query.shape[0]}")
    grouped_df = df_query.groupby(["noise_type"])[[acc_col]].agg(list).reset_index()
    print(f"Total permutations: {grouped_df.shape[0]}")

    # Calculate mean accuracy
    grouped_df["accuracy_mean"] = grouped_df[acc_col].apply(lambda x: np.mean(x))
    grouped_df["accuracy_std"] = grouped_df[acc_col].apply(lambda x: np.std(x))

    grouped_df.head(10)

    # Calculate the mean across each row
    x = grouped_df["noise_type"].tolist()
    y = grouped_df["accuracy_mean"].tolist()
    x_reorder = ["en", "es", "de", "zh", "vi", "hi", "ar", "multilingual"]
    # x_reorder = ['en', 'es', 'de', 'hi', 'ar', 'multilingual']
    length = range(len(x_reorder))
    y_reorder = [y[x.index(i)] for i in x_reorder]

    return x_reorder, y_reorder, grouped_df


def analysis2_noise_type_mono_bi_multi(df_pred, acc_col):
    # Question 2: Effect of changing langauge of distractor passages.

    """
    Here, we will analyse effect of controlling language of distractor documents on the mQA task.
    The search criteria is defined as follows:
    1. Question language is to be kept a constant here, i.e., question_lang == context_lang or question_lang == en
    2. Needle position is to be ignored here, i.e., report accuracy as average of needle_position == start, middle, or end.
    3. Context lang (or lang of needle) is to be kept constant here, i.e., context_lang == question_lang
    4. Noise type is the only variable in this experiment, and can take any value from - [en, hi, es, du, vi, zh, ar, or multilingual]
    """

    df_pred = df_pred[df_pred["question_lang"] == "en"]
    df_query = df_pred[(df_pred["question_lang"] == df_pred["context_lang"])]
    # df_query = df_query[(df_query['context_lang'] == 'en')]

    print(f"Total query data: {df_query.shape[0]}")
    df_query.head(2)
    grouped_df = df_query.groupby(["noise_type"])[[acc_col]].agg(list).reset_index()
    print(f"Total permutations: {grouped_df.shape[0]}")
    # Assert that total permutations equal to 49
    # assert grouped_df.shape[0] == 49

    # Calculate mean accuracy
    grouped_df["accuracy_mean"] = grouped_df[acc_col].apply(lambda x: np.mean(x))
    grouped_df["accuracy_std"] = grouped_df[acc_col].apply(lambda x: np.std(x))

    grouped_df.head(10)

    # Fine grained analysis - monolingual, bilingual, and multilingual cases.
    # Monolingual: noise_type == context_lang (eg: en:en)
    # Bilingual: noise_type != context_lang (eg en:hi)
    # Multilingual: noise_type == multilingual (eg en:multilingual)

    df_monolingual = grouped_df[(grouped_df["noise_type"] == "en")]
    monolingual_val = df_monolingual["accuracy_mean"].values.tolist()[0]

    df_bilingual = grouped_df[
        (grouped_df["noise_type"] != "en") & (grouped_df.index != "multilingual")
    ]
    bilingual_val = np.mean(df_bilingual["accuracy_mean"].values.tolist()[0])

    df_multilingual = grouped_df[(grouped_df["noise_type"] == "multilingual")]
    multilingual_val = df_multilingual["accuracy_mean"].values.tolist()[0]

    x = ["monolingual", "bilingual", "multilingual"]
    y = [monolingual_val, bilingual_val, multilingual_val]
    length = range(len(x))

    return x, y, grouped_df


def analysis2_needle_position(df_pred, acc_col, keep_context_lang="all"):
    # Question1: Effect of changing language of needle

    """
    Here, we will analyse effect of controlling language of relevant information (needle passage) on the downstream multilingual QA task.
    The search criteria is defined as follows:
    1. Question language is to be kept a constant here, i.e., question_lang == noise_type or question_lang == en
    2. Needle position is to be ignored here, i.e., report accuracy as average of needle_position == start, middle, or end.
    3. Noise type is to be kept constant here, i.e., noise_type == question_lang
    4. Context lang (or lang of needle) is the only variable here, and can take any value from - [en, hi, es, du, vi, zh, ar]
    """

    df_query = df_pred[df_pred["question_lang"] == "en"]

    if keep_context_lang != "all":
        df_query = df_query[(df_query["context_lang"].isin(keep_context_lang))]

    print(f"Total query data: {df_query.shape[0]}")
    df_query.head(2)

    grouped_df = (
        df_query.groupby(["needle_position"])[[acc_col]].agg(list).reset_index()
    )
    print(f"Total permutations: {grouped_df.shape[0]}")

    # Calculate mean accuracy
    grouped_df["accuracy_mean"] = grouped_df[acc_col].apply(lambda x: np.mean(x))
    grouped_df["accuracy_std"] = grouped_df[acc_col].apply(lambda x: np.std(x))

    grouped_df.head(10)

    # Calculate the mean across each row
    x = grouped_df["needle_position"].tolist()
    y = grouped_df["accuracy_mean"].tolist()
    x_reorder = ["start", "middle", "end"]
    length = range(len(x_reorder))
    y_reorder = [y[x.index(i)] for i in x_reorder]

    return x_reorder, y_reorder, grouped_df


def plot_heatmap_for_noise_type_needle_lang_correlation(df_pred, acc_col):
    # Question 2: Effect of changing langauge of distractor passages.

    """
    Here, we will analyse effect of controlling language of distractor documents on the mQA task.
    The search criteria is defined as follows:
    1. Question language is to be kept a constant here, i.e., question_lang == context_lang or question_lang == en
    2. Needle position is to be ignored here, i.e., report accuracy as average of needle_position == start, middle, or end.
    3. Context lang (or lang of needle) is to be kept constant here, i.e., context_lang == question_lang
    4. Noise type is the only variable in this experiment, and can take any value from - [en, hi, es, du, vi, zh, ar, or multilingual]
    """

    df_pred = df_pred[df_pred["question_lang"] == "en"]
    df_query = df_pred[df_pred["noise_type"] != "multilingual"]

    print(f"Total query data: {df_query.shape[0]}")
    grouped_df = (
        df_query.groupby(["context_lang", "noise_type"])[[acc_col]]
        .agg(list)
        .reset_index()
    )
    print(f"Total permutations: {grouped_df.shape[0]}")

    grouped_df["accuracy_mean"] = grouped_df[acc_col].apply(lambda x: np.mean(x))
    grouped_df["accuracy_std"] = grouped_df[acc_col].apply(lambda x: np.std(x))

    pivot_table = grouped_df.pivot_table(
        index="noise_type", columns="context_lang", values="accuracy_mean"
    )
    x_reorder = ["en", "de", "es", "zh", "vi", "hi", "ar"]
    pivot_table = pivot_table[x_reorder]
    pivot_table = pivot_table.reindex(x_reorder)

    return pivot_table


def analysis2_noise_type_contrast(df_pred, acc_col):
    # Question 2: Effect of changing langauge of distractor passages.

    df1 = df_pred[df_pred["context_lang"] == df_pred["noise_type"]]
    df2 = df_pred[df_pred["context_lang"] != df_pred["noise_type"]]

    langs = ["en", "de", "es", "zh", "vi", "hi", "ar"]
    acc1 = []
    acc2 = []
    for lang in langs:
        df = df_pred[df_pred["context_lang"] == lang]
        df1 = df[df["context_lang"] == df["noise_type"]]
        df2 = df[df["context_lang"] != df["noise_type"]]
        acc1.append(df1[df1[acc_col] == 1].shape[0] / df1.shape[0])
        acc2.append(df2[df2[acc_col] == 1].shape[0] / df2.shape[0])

    return langs, acc1, acc2
