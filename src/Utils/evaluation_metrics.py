from typing import List, Dict
import json

def get_metrics(dev_list, pred_list, class_key):
    classes = set()
    for r in dev_list:
        classes.add(r[class_key])
    for r in pred_list:
        classes.add(r[class_key])

    per_class = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0

    for cls in classes:
        dev_cls = [r for r in dev_list if r[class_key] == cls]
        pred_cls = [r for r in pred_list if r[class_key] == cls]

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

def get_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_ner_entities(docs):
    data_list = []
    for doc_id, doc in enumerate(docs):
        for ent in doc.ents:
            data_list.append({
                "doc_id": doc_id,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_,
            })
    return data_list


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

def get_concept_relations(data):
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

def calculate_ner_metrics(nlp, docs: List):
    dev_list = get_ner_entities(docs)
    pred_list = get_ner_entities([nlp(doc.text) for doc in docs])
    
    metrics = get_metrics(dev_list, pred_list, class_key="label")
    return metrics

def calculate_nerd_metrics(dev_path: str, pred_path: str):
    dev = get_json(dev_path)
    pred = get_json(pred_path)
    
    dev_list = get_relations(dev)
    pred_list = get_relations(pred)
    
    metrics = get_metrics(dev_list, pred_list, class_key="label")
    return metrics

def calculate_mention_re_metrics(dev_path: str, pred_path: str):
    dev = get_json(dev_path)
    pred = get_json(pred_path)
    
    dev_list = get_mention_relations(dev)
    pred_list = get_mention_relations(pred)
    
    metrics = get_metrics(dev_list, pred_list, class_key="predicate")
    return metrics

def calculate_concept_re_metrics(dev_path: str, pred_path: str):
    dev = get_json(dev_path)
    pred = get_json(pred_path)
    
    dev_list = get_concept_relations(dev)
    pred_list = get_concept_relations(pred)
    
    metrics = get_metrics(dev_list, pred_list, class_key="predicate")
    return metrics
