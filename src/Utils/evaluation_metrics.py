

def extract_entities(doc):
    return set((ent.start_char, ent.end_char, ent.label_) for ent in doc.ents)


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


def calculate_iou(ent1, ent2): #Intersection over Union for same label only!
    start1, end1, label1 = ent1
    start2, end2, label2 = ent2
    
    if label1 != label2:
        return 0.0
    
    # Calculate intersection
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    
    if inter_end <= inter_start:
        return 0.0  
    
    intersection = inter_end - inter_start
    union = max(end1, end2) - min(start1, start2)
    
    return intersection / union if union > 0 else 0.0


def calculate_metrics(all_y_true_flat, all_y_pred_flat, iou_threshold=0.5):
    # Get all unique classes
    classes = list(set(label for _, _, label in all_y_true_flat + all_y_pred_flat))
    
    if not classes:
        return {
            "class_precision": {},
            "class_recall": {},
            "class_f1": {},
            "macro": (0, 0, 0),
            "micro": (0, 0, 0)
        }
    
    true_entities = set(all_y_true_flat)
    pred_entities = set(all_y_pred_flat)
    

    matched_pred = set()
    matched_true = set()
    
    class_tp = {c: 0 for c in classes}
    class_fp = {c: 0 for c in classes}
    class_fn = {c: 0 for c in classes}
    

    for true_ent in true_entities:
        best_match = None
        best_iou = 0
        
        for pred_ent in pred_entities:
            if pred_ent in matched_pred:
                continue
            
            iou = calculate_iou(true_ent, pred_ent)
            if iou > best_iou:
                best_iou = iou
                best_match = pred_ent
        
        if best_iou >= iou_threshold:
            class_tp[true_ent[2]] += 1
            matched_pred.add(best_match)
            matched_true.add(true_ent)
        else:
            class_fn[true_ent[2]] += 1
    

    for pred_ent in pred_entities:
        if pred_ent not in matched_pred:
            class_fp[pred_ent[2]] += 1
    

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
    

    macro_p = sum(class_precision.values()) / len(classes) if len(classes) > 0 else 0
    macro_r = sum(class_recall.values()) / len(classes) if len(classes) > 0 else 0
    macro_f1 = sum(class_f1.values()) / len(classes) if len(classes) > 0 else 0
    
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
