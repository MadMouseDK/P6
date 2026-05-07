import pickle
import os
import random
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer, InputExample, losses, models, util
from pathlib import Path


base_path = Path.cwd()
output_path = base_path.parent / "Models" / "SciBERT_NERD"

with open(base_path.parent / "Output" / "kb_synonyms_labels.pickle" "rb") as f:
     kb_synonyms_labels = pickle.load(f)

all_examples = []

for entity_id, aliases in kb_synonyms_labels.items():
    aliases = list(set(aliases)) # remove duplicates
    
    for i in range(len(aliases)):
        for j in range(i + 1, len(aliases)):
            all_examples.append(
                InputExample(texts=[aliases[i], aliases[j]])
            )

train_examples, val_examples = train_test_split(all_examples, test_size=0.1, random_state=42)
train_loader = DataLoader(train_examples, shuffle=True, batch_size=32)

transformer = models.Transformer("allenai/scibert_scivocab_uncased")
transformer.max_seq_length = 256

pooling = models.Pooling(
     transformer.get_word_embedding_dimension(), 
     pooling_mode_mean_tokens=True )

model = SentenceTransformer(modules=[transformer, pooling])

loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_loader, loss)],
    epochs=25,
    warmup_steps=750
)

model.save(output_path)
