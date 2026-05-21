#Using the model on Dev data for output file

import json
import os
import sys

import torch
from transformers import AutoTokenizer

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from test import ( 
    ALLOWED_LABELS,
    E1_END,
    E1_START,
    E2_END,
    E2_START,
    LABELS,
    NO_REL,
    REModel,
    SPECIAL_TOKENS,
    insert_markers,
)


def _load_model(model_dir, device):
    ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location=device)
    labels = ckpt["labels"]
    model_name = ckpt["model_name"]
    max_length = ckpt["max_length"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if any(t not in tokenizer.get_vocab() for t in SPECIAL_TOKENS):
        tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    model = REModel(model_name, num_labels=len(labels))
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, tokenizer, labels, max_length


def _encode_pair(text, s, o, tokenizer, max_length):
    marked = insert_markers(text, s["start_idx"], s["end_idx"], o["start_idx"], o["end_idx"])
    enc = tokenizer(marked, truncation=True, max_length=max_length,
                    padding="max_length", return_tensors="pt")
    ids = enc["input_ids"].squeeze(0)
    attn = enc["attention_mask"].squeeze(0)
    e1_id = tokenizer.convert_tokens_to_ids(E1_START)
    e1c_id = tokenizer.convert_tokens_to_ids(E1_END)
    e2_id = tokenizer.convert_tokens_to_ids(E2_START)
    e2c_id = tokenizer.convert_tokens_to_ids(E2_END)
    e1m = torch.zeros_like(ids, dtype=torch.float)
    e2m = torch.zeros_like(ids, dtype=torch.float)
    in1 = in2 = False
    for i, t in enumerate(ids.tolist()):
        if t == e1_id: in1 = True; continue
        if t == e1c_id: in1 = False; continue
        if t == e2_id: in2 = True; continue
        if t == e2c_id: in2 = False; continue
        if in1: e1m[i] = 1.0
        if in2: e2m[i] = 1.0
    if e1m.sum() == 0:
        pos = (ids == e1_id).nonzero(as_tuple=True)[0]
        if len(pos): e1m[pos[0]] = 1.0
    if e2m.sum() == 0:
        pos = (ids == e2_id).nonzero(as_tuple=True)[0]
        if len(pos): e2m[pos[0]] = 1.0
    return ids, attn, e1m, e2m


@torch.no_grad()
def predict(model_dir, dev_path, out_path, batch_size=32, max_docs=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, labels, max_length = _load_model(model_dir, device)

    with open(dev_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    if max_docs:
        keys = list(documents.keys())[:max_docs]
        documents = {k: documents[k] for k in keys}
        print(f"Running on {max_docs} documents (subset mode)")

    submission = {}
    for pmid, doc in documents.items():
        meta = doc.get("metadata", {})
        texts = {"title": meta.get("title", ""), "abstract": meta.get("abstract", "")}
        ents = [e for e in (doc.get("entities") or []) if e.get("label") in ALLOWED_LABELS]

        # Group entities by location so spans index into the right text
        by_loc = {"title": [], "abstract": []}
        for e in ents:
            if e.get("location") in by_loc:
                by_loc[e["location"]].append(e)

        mention_rels = []
        concept_seen = set()
        concept_rels = []

        batch = []  # (s, o, text)
        def flush():
            if not batch:
                return
            ids_b, attn_b, e1_b, e2_b = [], [], [], []
            for s, o, text in batch:
                ids, attn, e1m, e2m = _encode_pair(text, s, o, tokenizer, max_length)
                ids_b.append(ids); attn_b.append(attn); e1_b.append(e1m); e2_b.append(e2m)
            out = model(
                input_ids=torch.stack(ids_b).to(device),
                attention_mask=torch.stack(attn_b).to(device),
                e1_mask=torch.stack(e1_b).to(device),
                e2_mask=torch.stack(e2_b).to(device),
            )
            preds = out["logits"].argmax(-1).cpu().tolist()
            for (s, o, _text), p in zip(batch, preds):
                lab = labels[p]
                if lab == NO_REL:
                    continue
                mention_rels.append({
                    "subject_text_span": s["text_span"],
                    "subject_label": s["label"],
                    "predicate": lab,
                    "object_text_span": o["text_span"],
                    "object_label": o["label"],
                })
                triple = (s.get("uri", ""), s["label"], lab, o.get("uri", ""), o["label"])
                if s.get("uri") and o.get("uri") and s["uri"] != o["uri"] and triple not in concept_seen:
                    concept_seen.add(triple)
                    concept_rels.append({
                        "subject_uri": s["uri"],
                        "subject_label": s["label"],
                        "predicate": lab,
                        "object_uri": o["uri"],
                        "object_label": o["label"],
                    })
            batch.clear()

        for loc, loc_ents in by_loc.items():
            text = texts[loc]
            if not text or len(loc_ents) < 2:
                continue
            for i, s in enumerate(loc_ents):
                for j, o in enumerate(loc_ents):
                    if i == j or (s["start_idx"] == o["start_idx"] and s["end_idx"] == o["end_idx"]):
                        continue
                    batch.append((s, o, text))
                    if len(batch) >= batch_size:
                        flush()
        flush()

        submission[pmid] = {
            "mention_level_relations": mention_rels,
            "concept_level_relations": concept_rels,
        }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}  (docs: {len(submission)})") #QoL print
