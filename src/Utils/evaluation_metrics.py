from typing import List, Dict, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass
import json


@dataclass # Using dataclass to automatically generate init and other methods
class EntityMetrics:
    tp: int = 0 
    fp: int = 0 
    fn: int = 0 
    
    @property # Makes the method look like an attribute
    def precision(self) -> float:#Precision = TP / (TP + FP)
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)
    
    @property
    def recall(self) -> float: #Recall = TP / (TP + FN)
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)
    
    @property
    def f1(self) -> float: #F1 = 2 * (Precision * Recall) / (Precision + Recall)
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    

def extract_entities(doc) -> Set[Tuple[int, int, str]]:
    return {(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents}


def calculate_ner_metrics(nlp, docs: List) -> Dict:
    # Accumulate metrics per class
    class_metrics: Dict[str, EntityMetrics] = defaultdict(EntityMetrics)
    
    for doc in docs:
        dev_entities = extract_entities(doc)
        pred_doc = nlp(doc.text)
        pred_entities = extract_entities(pred_doc)
        
        all_classes = set()# Used for base classes in DEV
        for start, end, label in dev_entities: 
            all_classes.add(label)
            
        for start, end, label in pred_entities: # Will add new classes if the model predicts something not in the gold data
            all_classes.add(label)
        
                
        # Calculate TP, FP, FN for each class
        for entity_class in all_classes:
            gold_class = {e for e in dev_entities if e[2] == entity_class}
            pred_class = {e for e in pred_entities if e[2] == entity_class}
            
            tp = len(gold_class & pred_class)
            fp = len(pred_class - gold_class)
            fn = len(gold_class - pred_class)
            
            class_metrics[entity_class].tp += tp
            class_metrics[entity_class].fp += fp
            class_metrics[entity_class].fn += fn
    
    classes = sorted(class_metrics.keys())
    
    # Calculate per-class metrics
    class_precision = {cls: class_metrics[cls].precision for cls in classes}
    class_recall = {cls: class_metrics[cls].recall for cls in classes}
    class_f1 = {cls: class_metrics[cls].f1 for cls in classes}
    
    # Calculate micro-averaged metrics 
    total_metrics = EntityMetrics(
        tp=sum(m.tp for m in class_metrics.values()),
        fp=sum(m.fp for m in class_metrics.values()),
        fn=sum(m.fn for m in class_metrics.values())
    )
    micro_p = total_metrics.precision
    micro_r = total_metrics.recall
    micro_f1 = total_metrics.f1
    
    # Calculate macro-averaged metrics
    if classes:
        macro_p = sum(class_precision.values()) / len(classes)
        macro_r = sum(class_recall.values()) / len(classes)
        macro_f1 = sum(class_f1.values()) / len(classes)
    else:
        macro_p = macro_r = macro_f1 = 0.0
    
    return {
        "class_precision": class_precision,
        "class_recall": class_recall,
        "class_f1": class_f1,
        "macro": (macro_p, macro_r, macro_f1),
        "micro": (micro_p, micro_r, micro_f1)
    }

    
    
def get_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_relations(data):
    data_list = []
    for pmid, text in data.items():
        for i in text["entities"]:
            data_list.append({
                "pmid": pmid,
                "text_span": i["text_span"],
                "label": i["label"],
                "uri": i["uri"],
                    
            })
    return data_list

def calculate_nerd_metrics(dev_path: str, pred_path: str):
    dev = get_json(dev_path)
    pred = get_json(pred_path)

    dev_list = get_relations(dev)
    pred_list = get_relations(pred)


    classes = set()
    for r in dev_list:
        classes.add(r["label"])
    for r in pred_list:
        classes.add(r["label"])

    per_class = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0

    for cls in classes:
        dev_cls = [r for r in dev_list if r["label"] == cls]
        pred_cls = [r for r in pred_list if r["label"] == cls]

        tp = 0
        fp = 0
        fn = 0

        for p in pred_cls:
            if p in dev_cls:
                tp += 1
            else:
                fp += 1

        for d in dev_cls:
            if d not in pred_cls:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn
        sum_precision += precision
        sum_recall += recall
        sum_f1 += f1

    # Micro: pool TP/FP/FN across all classes, then compute one P/R/F1
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    # Macro: average the per-class P/R/F1 (treats every class equally)
    n = len(classes)
    macro_precision = sum_precision / n if n > 0 else 0.0
    macro_recall = sum_recall / n if n > 0 else 0.0
    macro_f1 = sum_f1 / n if n > 0 else 0.0

    return [micro_precision, micro_recall, micro_f1, macro_precision,macro_recall,macro_f1]


def calculate_concept_relations(data):
    data_list = []
    for pmid, doc in data.items():
        for r in doc.get("concept_level_relations", []):
            data_list.append({
                "pmid": pmid,
                "subject_uri": r["subject_uri"],
                "subject_label": r["subject_label"],
                "predicate": r["predicate"],
                "object_uri": r["object_uri"],
                "object_label": r["object_label"],
            })
    return data_list


def calculate_concept_re_metrics(dev_path: str, pred_path: str):
    dev = get_json(dev_path)
    pred = get_json(pred_path)

    dev_list = calculate_concept_relations(dev)
    pred_list = calculate_concept_relations(pred)

    # Class = predicate (the relation type being classified)
    classes = set()
    for r in dev_list:
        classes.add(r["predicate"])
    for r in pred_list:
        classes.add(r["predicate"])

    per_class = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0

    for cls in classes:
        dev_cls = [r for r in dev_list if r["predicate"] == cls]
        pred_cls = [r for r in pred_list if r["predicate"] == cls]

        tp = 0
        fp = 0
        fn = 0

        for p in pred_cls:
            if p in dev_cls:
                tp += 1
            else:
                fp += 1

        for d in dev_cls:
            if d not in pred_cls:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn
        sum_precision += precision
        sum_recall += recall
        sum_f1 += f1

    # Micro: pool TP/FP/FN across all classes, then compute one P/R/F1
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    # Macro: average the per-class P/R/F1 (treats every class equally)
    n = len(classes)
    macro_precision = sum_precision / n if n > 0 else 0.0
    macro_recall = sum_recall / n if n > 0 else 0.0
    macro_f1 = sum_f1 / n if n > 0 else 0.0

    return [micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1]


def get_mention_relations(data):
    data_list = []
    for pmid, doc in data.items():
        for r in doc.get("mention_level_relations", []):
            data_list.append({
                "pmid": pmid,
                "subject_text_span": r["subject_text_span"],
                "subject_label": r["subject_label"],
                "predicate": r["predicate"],
                "object_text_span": r["object_text_span"],
                "object_label": r["object_label"],
            })
    return data_list


def calculate_mention_re_metrics(dev_path: str, pred_path: str):
    dev = get_json(dev_path)
    pred = get_json(pred_path)

    dev_list = get_mention_relations(dev)
    pred_list = get_mention_relations(pred)

    # Class = predicate (the relation type being classified)
    classes = set()
    for r in dev_list:
        classes.add(r["predicate"])
    for r in pred_list:
        classes.add(r["predicate"])

    per_class = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0

    for cls in classes:
        dev_cls = [r for r in dev_list if r["predicate"] == cls]
        pred_cls = [r for r in pred_list if r["predicate"] == cls]

        tp = 0
        fp = 0
        fn = 0

        for p in pred_cls:
            if p in dev_cls:
                tp += 1
            else:
                fp += 1

        for d in dev_cls:
            if d not in pred_cls:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn
        sum_precision += precision
        sum_recall += recall
        sum_f1 += f1

    # Micro: pool TP/FP/FN across all classes, then compute one P/R/F1
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    # Macro: average the per-class P/R/F1 (treats every class equally)
    n = len(classes)
    macro_precision = sum_precision / n if n > 0 else 0.0
    macro_recall = sum_recall / n if n > 0 else 0.0
    macro_f1 = sum_f1 / n if n > 0 else 0.0

    return [micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1]

