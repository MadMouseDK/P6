import os
import pandas as pd
import numpy as np
import warnings
import spacy
import re
from spacy.tokens import DocBin
from spacy.training import Example
from typing import Tuple, Iterator



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




#
def evaluate(model, dataset):

    total_pred = 0
    total_true = 0
    correct = 0

    for text, gold_entities in dataset:

        predicted = model.predict(text)

        total_pred += len(predicted)
        total_true += len(gold_entities)

        for ent in predicted:
            if ent in gold_entities:
                correct += 1

    precision = correct / total_pred
    recall = correct / total_true
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1