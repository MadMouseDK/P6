import os
import pickle
import subprocess
import sys

import spacy
from spacy.tokens import Doc, DocBin
from spacy.util import filter_spans
from tqdm import tqdm

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

def load_pickle(datatype: str) -> list:
    allowed_values = ["train", "dev"]
    if datatype not in allowed_values:
        raise ValueError(
            f"{datatype} value not allowed. Allowed values: {allowed_values}"
        )
    cwd = os.getcwd()
    folder = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    if cwd.endswith(folder):
        cwd = os.path.dirname(cwd)
    path = os.path.join(cwd, "Data", "processed", f"{datatype}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)
    return None


nlp = spacy.blank("en")

def create_spacy_docbin(data: list) -> DocBin:
    db = DocBin()
    entities_skipped = 0
    start_end_idx = []
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
                start_end_idx.append((start, end))
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


def save_spacy(data_spacy: DocBin, name: str) -> None:
    cwd = os.getcwd()
    folder = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    if cwd.endswith(folder):
        cwd = os.path.dirname(cwd)
    path = os.path.join(cwd, "Data", "processed", f"{name}_data_NER_V4.spacy")
    data_spacy.to_disk(path)


def train_spacy_model(model_name: str):

    cwd = os.getcwd()
    folder = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    if cwd.endswith(folder):
        project_dir = os.path.dirname(cwd)
    config_dir = os.path.join(cwd, "cfg")
    base_config_path = os.path.join(config_dir, f"base_config_{model_name}_NER.cfg")
    config_path = os.path.join(cwd, "cfg", f"config_{model_name}_NER.cfg")
    output_path = os.path.join(project_dir, "Models", f"{model_name}_NER", "output")


    subprocess.run(
        [
            sys.executable,
            "-m",
            "spacy",
            "init",
            "fill-config",
            base_config_path,
            config_path
        ],
        cwd = project_dir,
        check = True
    )


    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "spacy",
            "train",
            config_path, 
            "--output",
            output_path
        ],
        cwd = project_dir,
        check = True
    )

    if result.returncode != 0:
        print(f"Training failed with exit code {result.returncode}")

train_data = load_pickle("train")
dev_data = load_pickle("dev")

train = create_spacy_docbin(train_data)
dev = create_spacy_docbin(dev_data)

save_spacy(train, "train")
save_spacy(dev, "dev")

"""
train_spacy_model("RoBERTa")
train_spacy_model("SciBERT")
train_spacy_model("BlueBERT-pubmed_NER")
train_spacy_model("BlueBERT-pubmed-mimic_NER")
train_spacy_model("BioBERT-base_NER")
train_spacy_model("BioBERT-large_NER")
"""

