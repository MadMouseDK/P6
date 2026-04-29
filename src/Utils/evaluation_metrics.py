from typing import List, Dict, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass


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


def calculate_metrics(nlp, docs: List) -> Dict:
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