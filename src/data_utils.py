import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.plot_utils import plot_ratios
import config
from datasets import Dataset


def load_df(df_path):
    df = pd.read_excel(df_path)
    print(df.info())
    return df


def analyze_data(df, labels):
    n = len(labels)
    classes_ratios = np.full((n, n), np.nan)
    for i1, class1 in enumerate(labels):
        for i2, class2 in enumerate(labels):
            if i2 > i1:
                break
            ratio = len(df[(df[class1] == 1) & (df[class2] == 1)]) / len(df)
            classes_ratios[i1][i2] = round(ratio, 3)
    plot_ratios(classes_ratios, labels)


def split_dataset(dataset, test_size=0.2, seed=42):
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=seed)
    return dataset.select(train_indices), dataset.select(test_indices)


def tokenize_and_map_labels(batch, tokenizer):
    batch_encoding = tokenizer(batch['original_text'], truncation=True, padding=True)
    labels_all = []
    for label in config.labels:
        labels_all.append([0 if val is None else 1 for val in batch[label]])
    batch_encoding['labels'] = np.array(labels_all).T
    return batch_encoding


def convert_df_to_train_eval_datasets(train_df, tokenizer):
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset, validation_dataset = split_dataset(train_dataset, test_size=config.train_validation_ratio)
    fn_kwargs = {'tokenizer': tokenizer}
    train_dataset_processed = train_dataset.map(tokenize_and_map_labels, batched=True, fn_kwargs=fn_kwargs)
    validation_dataset_processed = validation_dataset.map(tokenize_and_map_labels, batched=True, fn_kwargs=fn_kwargs)
    return train_dataset_processed, validation_dataset_processed


def apply_threshold(column, threshold=0.5):
    return column.apply(lambda value: 1 if value > threshold else None)


def save_labelled_test(df, scores):
    df[config.labels] = scores
    df[[f'{c} score' for c in config.labels]] = scores
    for col in config.labels:
        df[col] = apply_threshold(df[col])
    df.to_excel(os.path.join(config.results_path, 'Different_test_set_labelled.xlsx'), index=False, engine="openpyxl")