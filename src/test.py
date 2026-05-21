import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from sklearn.metrics import f1_score
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1" # "bert-base-cased" or "dmis-lab/biobert-base-cased-v1.1"
EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5
MAX_LENGTH = 256
NEG_RATIO = 2.0
MAX_DOCS = 0                             
SEED = 1337

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "Data", "processed", "output.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Models", "RE_BERT_test")


PREDICATES = [
    "located in", "interact", "influence", "change expression", "part of",
    "impact", "produced by", "administered", "strike", "change abundance",
    "affect", "is a", "target", "change effect", "used by", "is linked to",
    "compared to",
]

NO_REL = "no_relation"
LABELS = [NO_REL] + PREDICATES
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

ALLOWED_LABELS = {
    "anatomical location", "animal", "bacteria", "biomedical technique",
    "chemical", "DDF", "dietary supplement", "drug", "food", "gene",
    "human", "microbiome", "statistical technique",
}

E1_START, E1_END = "[E1]", "[/E1]"
E2_START, E2_END = "[E2]", "[/E2]"
SPECIAL_TOKENS = [E1_START, E1_END, E2_START, E2_END]


@dataclass
class REExample:
    text: str
    s_start: int
    s_end: int
    o_start: int
    o_end: int
    label: str


def build_examples(docs, max_neg_ratio, rng):
    examples = []
    for text, annot in docs:
        ents = [e for e in (annot.get("entities") or []) if e[2] in ALLOWED_LABELS]
        rels = annot.get("relations") or []
        gold = {}
        for r in rels:
            if len(r) == 5:
                s_start, _s, pred, o_start, _o = r
                gold[(s_start, o_start)] = pred

        positives, negatives = [], []
        for i, s in enumerate(ents):
            for j, o in enumerate(ents):
                if i == j or (s[0] == o[0] and s[1] == o[1]):
                    continue
                pred = gold.get((s[0], o[0]))
                ex = REExample(
                    text=text,
                    s_start=s[0], s_end=s[1] + 1,
                    o_start=o[0], o_end=o[1] + 1,
                    label=pred if pred in PREDICATES else NO_REL,
                )
                (positives if ex.label != NO_REL else negatives).append(ex)

        if positives:
            keep = int(len(positives) * max_neg_ratio)
            rng.shuffle(negatives)
            negatives = negatives[:keep]
        else:
            rng.shuffle(negatives)
            negatives = negatives[: min(8, len(negatives))]
        examples.extend(positives + negatives)
    return examples

#Entity-aware pooling
def insert_markers(text, s_start, s_end, o_start, o_end):
    spans = sorted(
        [(s_start, s_end, E1_START, E1_END), (o_start, o_end, E2_START, E2_END)],
        key=lambda x: x[0],
    )
    a_s, a_e, a_open, a_close = spans[0]
    b_s, b_e, b_open, b_close = spans[1]
    if b_s < a_e:
        b_s = max(b_s, a_e)
        b_e = max(b_e, b_s)
    return (
        text[:a_s] + a_open + text[a_s:a_e] + a_close
        + text[a_e:b_s] + b_open + text[b_s:b_e] + b_close
        + text[b_e:]
    )


class REDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples = examples
        self.tok = tokenizer
        self.max_length = max_length
        self.e1_id = tokenizer.convert_tokens_to_ids(E1_START)
        self.e2_id = tokenizer.convert_tokens_to_ids(E2_START)
        self.e1c_id = tokenizer.convert_tokens_to_ids(E1_END)
        self.e2c_id = tokenizer.convert_tokens_to_ids(E2_END)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        marked = insert_markers(ex.text, ex.s_start, ex.s_end, ex.o_start, ex.o_end)
        enc = self.tok(
            marked, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        ids = enc["input_ids"].squeeze(0)
        attn = enc["attention_mask"].squeeze(0)

        e1m = torch.zeros_like(ids, dtype=torch.float)
        e2m = torch.zeros_like(ids, dtype=torch.float)
        in1 = in2 = False
        for i, t in enumerate(ids.tolist()):
            if t == self.e1_id: in1 = True; continue
            if t == self.e1c_id: in1 = False; continue
            if t == self.e2_id: in2 = True; continue
            if t == self.e2c_id: in2 = False; continue
            if in1: e1m[i] = 1.0
            if in2: e2m[i] = 1.0
        if e1m.sum() == 0:
            pos = (ids == self.e1_id).nonzero(as_tuple=True)[0]
            if len(pos): e1m[pos[0]] = 1.0
        if e2m.sum() == 0:
            pos = (ids == self.e2_id).nonzero(as_tuple=True)[0]
            if len(pos): e2m[pos[0]] = 1.0

        return {
            "input_ids": ids,
            "attention_mask": attn,
            "e1_mask": e1m,
            "e2_mask": e2m,
            "labels": torch.tensor(LABEL2ID[ex.label], dtype=torch.long),
        }



# Model (R-BERT head over a HuggingFace encoder)
class REModel(nn.Module):
    def __init__(self, model_name, num_labels, class_weights=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.cls_proj = nn.Linear(h, h)
        self.e1_proj = nn.Linear(h, h)
        self.e2_proj = nn.Linear(h, h)
        self.classifier = nn.Linear(h * 3, num_labels)
        self.act = nn.GELU()
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(num_labels),
        )

    @staticmethod
    def _pool(hidden, mask):
        mask = mask.unsqueeze(-1)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def forward(self, input_ids, attention_mask, e1_mask, e2_mask, labels=None):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        v = torch.cat([
            self.act(self.cls_proj(self.dropout(h[:, 0]))),
            self.act(self.e1_proj(self.dropout(self._pool(h, e1_mask)))),
            self.act(self.e2_proj(self.dropout(self._pool(h, e2_mask)))),
        ], dim=-1)
        logits = self.classifier(self.dropout(v))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels, weight=self.get_buffer("class_weights"))
        return {"loss": loss, "logits": logits}



# Metrics used for training evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    pred_ids = [LABEL2ID[p] for p in PREDICATES]
    return {
        "macro_f1_pred": f1_score(labels, preds, labels=pred_ids, average="macro", zero_division=0),
        "micro_f1_pred": f1_score(labels, preds, labels=pred_ids, average="micro", zero_division=0),
    }


#Training the model
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_docs = json.load(f)
    if MAX_DOCS:
        all_docs = all_docs[:MAX_DOCS]

    pmids = sorted({d[1]["pmid"] for d in all_docs})
    rng.shuffle(pmids)
    n_dev = max(1, int(len(pmids) * 0.1))
    dev_pmids = set(pmids[:n_dev])
    train_docs = [d for d in all_docs if d[1]["pmid"] not in dev_pmids]
    dev_docs = [d for d in all_docs if d[1]["pmid"] in dev_pmids]
    print(f"Docs — train: {len(train_docs)}, dev: {len(dev_docs)}")

    train_ex = build_examples(train_docs, NEG_RATIO, rng)
    dev_ex = build_examples(dev_docs, NEG_RATIO, rng)
    print(f"Examples — train: {len(train_ex)}, dev: {len(dev_ex)}")
    print("Train label dist:", Counter(e.label for e in train_ex).most_common(5))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    counts = Counter(e.label for e in train_ex)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (len(LABELS) * max(counts.get(l, 1), 1)) for l in LABELS],
        dtype=torch.float,
    )

    model = REModel(MODEL_NAME, num_labels=len(LABELS), class_weights=weights)
    model.encoder.resize_token_embeddings(len(tokenizer))

    train_ds = REDataset(train_ex, tokenizer, MAX_LENGTH)
    dev_ds = REDataset(dev_ex, tokenizer, MAX_LENGTH)

    use_fp16 = device == "cuda"
    args = TrainingArguments(output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH_SIZE,       per_device_eval_batch_size=BATCH_SIZE, learning_rate=LR, weight_decay=0.01, warmup_ratio=0.1, eval_strategy="epoch",       save_strategy="epoch",  save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="macro_f1_pred",      greater_is_better=True, logging_steps=50, fp16=use_fp16, report_to="none", seed=SEED,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    final = trainer.evaluate()
    print("Final dev metrics:", final)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "labels": LABELS,
         "model_name": MODEL_NAME, "max_length": MAX_LENGTH},
        os.path.join(OUTPUT_DIR, "model.pt"),
    )
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
