import os
import pandas as pd
import numpy as np
import warnings
import spacy
import re
from spacy.tokens import DocBin
from spacy.training import Example
from typing import Tuple, Iterator


def extract_entities(doc):
    return set((ent.start, ent.end, ent.label_) for ent in doc.ents)


def score_evaluation(examples: Iterator[Example]):
    for example in examples:
        true = example.reference.ents
        pred = example.predicted.ents

        if true == pred:
            pass
    

def macro_average_precesion() -> float:
    raise NotImplemented

def macro_average_recall() -> float:
    raise NotImplemented

def micro_average_precesion() -> float:
    raise NotImplemented

def micro_average_recall() -> float:
    raise NotImplemented

def average_f1_score(precesion, recall) -> float:
    return 2 * ((precesion * recall) / (precesion + recall))

def metrics(nlp, docs):
    all_y_true = []
    all_y_pred = []

    for doc in docs:
        y_true = extract_entities(doc)
        pred_doc = nlp(doc.text)
        y_pred = extract_entities(pred_doc)

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

    
    all_y_true_flat = [ent for sublist in all_y_true for ent in sublist]
    all_y_pred_flat = [ent for sublist in all_y_pred for ent in sublist]

    
    return all_y_true_flat, all_y_pred_flat


def calculate_metrics(all_y_true_flat, all_y_pred_flat):
    # Get all unique classes
    classes = list(set(all_y_true_flat + all_y_pred_flat))

    class_tp = {c: 0 for c in classes}
    class_fp = {c: 0 for c in classes}
    class_fn = {c: 0 for c in classes}

    for cls in classes:
        for t, p in zip(all_y_true_flat, all_y_pred_flat):
            if p == cls and t == cls:
                class_tp[cls] += 1
            elif p == cls and t != cls:
                class_fp[cls] += 1
            elif p != cls and t == cls:
                class_fn[cls] += 1

    class_precision = {}
    class_recall = {}
    class_f1 = {}
    for cls in classes:
        tp = class_tp[cls]
        fp = class_fp[cls]
        fn = class_fn[cls]

        class_precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0
        class_recall[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0
        p = class_precision[cls]
        r = class_recall[cls]
        class_f1[cls] = 2 * p * r / (p + r) if (p + r) > 0 else 0


    macro_p = sum(class_precision.values()) / len(classes)
    macro_r = sum(class_recall.values()) / len(classes)
    macro_f1 = sum(class_f1.values()) / len(classes)


    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    return {
        "class_precision": class_precision,
        "class_recall": class_recall,
        "class_f1": class_f1,
        "macro": (macro_p, macro_r, macro_f1),
        "micro": (micro_p, micro_r, micro_f1)
    }

