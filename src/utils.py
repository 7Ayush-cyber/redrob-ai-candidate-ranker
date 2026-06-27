import json
import re
from pathlib import Path
from time import perf_counter
from contextlib import contextmanager

import pandas as pd


# Basic text cleaning
def clean_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def unique_ordered(items):
    seen = set()
    result = []

    for item in items:
        text = clean_text(item)

        if not text:
            continue

        key = text.lower()

        if key not in seen:
            seen.add(key)
            result.append(text)

    return result


def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default



# Sentence counting
def _protect_internal_dots(text):
    """
    Protect:
    - decimals like 6.9
    - dotted company names like redrob.ai, verloop.io
    """
    # protect decimals
    text = re.sub(r"(?<=\d)\.(?=\d)", "§", text)

    # protect dotted names like redrob.ai, pvt.ltd, verloop.io
    pattern = r"\b[A-Za-z]{2,}(?:\.[A-Za-z]{2,})+\b"

    def replace_dots(match):
        return match.group(0).replace(".", "§")

    text = re.sub(pattern, replace_dots, text)
    return text


def count_sentences(text):
    """
    Count sentences in a robust way.

    Rules:
    - count line breaks as natural separators
    - ignore decimal dots like 6.9
    - ignore company/domain dots like redrob.ai
    - if a line has no clear ending punctuation, count it as 1 sentence
    """
    cleaned = clean_text(text)

    if cleaned == "":
        return 0

    lines = str(text).splitlines()
    total = 0

    for line in lines:
        line = line.strip()

        if line == "":
            continue

        protected = _protect_internal_dots(line)

        endings = re.findall(r"(?<!§)[.!?](?=\s+[A-Z]|$)", protected)

        if len(endings) > 0:
            total += len(endings)
        else:
            total += 1

    if total == 0:
        total = 1

    return total



# File helpers
def normalize_text(text):
    text = clean_text(text)
    if text == "":
        return ""
    return text.lower()

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data, path, indent=4):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df, path, index=False):
    ensure_dir(path.parent)
    df.to_csv(path, index=index, encoding="utf-8")


def load_csv(path):
    return pd.read_csv(path)


def save_parquet(df, path, index=False):
    ensure_dir(path.parent)
    df.to_parquet(path, index=index)


def load_parquet(path):
    return pd.read_parquet(path)



# DataFrame helpers
def pick_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError("None of these columns were found: " + str(candidates))


def maybe_get(df, column, default=None):
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)



# Simple timer
@contextmanager
def timer(label):
    start = perf_counter()
    print("[" + label + "] start")
    try:
        yield
    finally:
        end = perf_counter()
        print("[" + label + "] done in " + str(round(end - start, 2)) + "s")



# Quick check
if __name__ == "__main__":
    examples = [
        "Senior Data Scientist with 5.7 years. Availability looks good.",
        "Worked at Redrob.ai from Bangalore.",
        "Response rate is 0.92 and notice period is 30.0 days.",
        "One line without punctuation",
        "Two lines\nSecond line here."
    ]

    for text in examples:
        print("=" * 60)
        print(text)
        print("Sentences:", count_sentences(text))