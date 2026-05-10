import os
import pickle
import re
import warnings

import pandas as pd
import spacy
import tqdm
from spacy.tokens import Doc, DocBin
from spacy.util import filter_spans

POSSIBLE_RELATIONS = {
    "located in": 0.0,
    "interact": 0.0,
    "influence": 0.0,
    "change expression": 0.0,
    "part of": 0.0,
    "impact": 0.0,
    "produced by": 0.0,
    "administered": 0.0,
    "strike": 0.0,
    "change abundance": 0.0,
    "affect": 0.0,
    "is a": 0.0,
    "target": 0.0,
    "change effect": 0.0,
    "used by": 0.0,
    "is linked to": 0.0,
    "compared to": 0.0,
}

Doc.set_extension("rel", default={}, force=True)
Doc.set_extension("rel_kb", default={}, force=True)


def to_pickle(file_path: str, file_name: str, obj):
    """  Saves an object to a pickle file"""
    path = os.path.join(file_path, file_name)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_dataset(data: str, root_path: str) -> pd.DataFrame | None:
    """Loads the dataset and returns a pandas dataframe if succesful otherwise return None"""
    allowed_values = ["train", "dev"]
    if data not in allowed_values:
        raise ValueError(f"{data} value not allowed. Allowed values: {allowed_values}")

    data_path = os.path.join(root_path, "Data", "raw", "Annotations")
    if data == "dev":
        data_path = os.path.join(data_path, "Dev", "json_format", "dev.json")
        return pd.read_json(data_path, orient="index")
    elif data == "train":
        data_path = os.path.join(data_path, "Train")
        train_gold_df = pd.read_json(os.path.join(data_path, "gold_quality", "json_format", "train_gold.json"), orient="index")
        train_silver_df = pd.read_json(os.path.join(data_path, "silver_quality", "json_format", "train_silver.json"), orient="index")
        train_silver2025_df = pd.read_json(os.path.join(data_path, "silver_quality", "json_format", "train_silver_2025.json"), orient="index")
        train_bronze_df = pd.read_json(os.path.join(data_path, "bronze_quality/json_format/train_bronze.json"), orient="index")
        return pd.concat([train_gold_df, train_silver2025_df, train_silver_df, train_bronze_df], axis=0)
    return None


def clean_html_tags(text: str, replacement: str = "#")  -> str:
    """Cleans HTML tags for some text and replaces it with a special charater"""
    return re.sub(r"<\/?[a-z][a-z0-9]*[^<>]*>|<!--.*?-->",lambda match: replacement * len(match.group(0)), text)


def valid_text_span(text_span: str) -> bool:
    """Returns true or false depeding on wether the text span is valid
    This can be cases where the text only include special charaters"""
    text_span = text_span.lower()

    if len(text_span) == 1 and text_span not in ["C", "A"]:
        return False

    regex = re.match(r"[^\,\.\-\'\*\;\:\<\>\!\&\#\s]", text_span)
    if regex is not None:
        if len(regex.group()) == 0:
            return False

    entities = ["anatomical location", "animal", "statistical technique", "biomedical technique", "bacteria", "chemical", "dietary supplement", "disease", "disorder", "finding", "drug", "food", "gene", "human", "microbiome"]
    if text_span in entities:
        return False

    ends_with = r"(system|systems|axis|model|models|response|mechanism|theraphy|symptoms|pathogenesis|destruction)\b$"
    return re.match(ends_with, text_span) is None


def change_index(text: str, text_span: str, start_idx: int, end_idx: int) -> tuple[int, int]:
    """Takes a text and the text span along with the indexes and make sure the the text is correct and does not miss or include anything it shouldn't"""
    if re.match(r"\A(\)|\]|\s)", text_span) is not None:
        msg = f"text_span {text_span} starts with ) or ] which is not valid"
        warnings.warn(msg, stacklevel=2)
        start_idx += 1

    if text_span.startswith("#") and re.match(r"(?<=\w)#", text_span) is None:
        msg = f"text_span {text_span} has html in beggining but not anywhere else"
        warnings.warn(msg, stacklevel=2)
        start_idx += 1

    if start_idx >= 1:
        if re.match(r"[^\d\w]", text[start_idx - 1]) is None:
            start_idx -= 1
            msg = f"text_span {text_span} does not capture the entire word: {text[start_idx:end_idx]}"
            warnings.warn(msg, stacklevel=2)

    if end_idx < (len(text) - 1):
        char = text[end_idx + 1]
        if char.isdigit() or char.isalpha():
            end_idx += 1
            msg = f"text_span {text_span} does not capture the entire word: {text[start_idx:end_idx]}"
            warnings.warn(msg, stacklevel=2)

    return (start_idx, end_idx)


def to_docbin(data: list) -> DocBin:
    nlp = spacy.blank("en")
    db = DocBin()
    entities_skipped = 0
    for i, (text, metadata) in enumerate(tqdm(data)):
        token_mapping = {}
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label, _ in metadata["entities"]:
            if start > end:
                print(f"Wrong start: {start}, end: {end}")
                start, end = end, start
            span = doc.char_span(start, end, label=label, alignment_mode="expand")
            if span is None:
                entities_skipped += 1
            else:
                ents.append(span)
                token_mapping[start] = span.start_char

        ents = filter_spans(ents)
        try:
            doc.ents = ents
        except ValueError as e:
            print(f"Skipping doc {i}: {e}")
            continue

        relations = {}
        relations_kb = {}
        tokens = {token.idx: token.i for token in doc}
        for ent_1 in doc.ents:
            for ent_2 in doc.ents:
                if ent_1 != ent_2:
                    relations[(ent_1.start, ent_2.start)] = POSSIBLE_RELATIONS.copy()


        for subject_idx, subject_uri, predicate, object_idx, object_uri in metadata["relations"]:
            try:
                relations_kb[(subject_uri, object_uri)] = POSSIBLE_RELATIONS.copy()
                subject_idx = token_mapping[subject_idx]
                subject_id = tokens[subject_idx]

                object_idx = token_mapping[object_idx]
                object_id = tokens[object_idx]
                relations[(subject_id, object_id)][predicate] = 1.0
                relations_kb[(subject_uri, object_uri)][predicate] = 1.0
            except KeyError:
                print(f"Failed for document: {i}")
                continue

        doc._.rel = relations
        doc._.rel_kb = relations_kb
        db.add(doc)
    return db


