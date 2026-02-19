import spacy
from spacy.cli.download import download
from spacy.training.example import Example
from spacy.util import minibatch
import numpy as np
import random

data = np.load("train_data.npy", allow_pickle=True)

try:
    model = spacy.load("en_core_web_lg")
except:
    download("en_core_web_lg")
    model = spacy.load("en_core_web_lg")

# Adds ner to pipe if not present
if not model.has_pipe("ner"):
    model.add_pipe("ner")
ner = model.get_pipe("ner")

# Replace with a more automatic way of doing it.
labels = ["anatomical location", "animal", "statistical technique", "biomedical technique", "bacteria", "chemical", "dietary supplement", "DDF", "drug", "food", "gene", "human", "microbiome"]
for label in labels:
    ner.add_label(label)
#text = "Amyotrophic lateral sclerosis (ALS) is a neuromuscular disease. The ALS mice expressing human mutant of transactive response DNA binding protein of 43 kDa (hmTDP43) showed intestinal dysfunction before neuromuscular symptoms. We hypothesize that restoring the intestinal and microbial homeostasis with a bacterial metabolite or probiotics delays the ALS disease onset. We investigate the pathophysiological changes in the intestine and neurons, intestinal and blood-brain barriers, and inflammation during the ALS progression. We then cultured enteric glial cells (EGCs) isolated from TDP43 mice for mechanistic studies. TDP43 mice had significantly decreased intestinal mobility, increased permeability, and weakened muscle, compared with the age-matched wild-type mice. We observed increased hmTDP43 and Glial fibrillary acidic protein (GFAP), and decreased expression of \u03b1-smooth muscle actin (\u03b1-SMA), tight junction proteins (ZO-1 and Claudin-5) in the colon, spinal cord, and brain in TDP43 mice. TDP43 mice had reduced Butyryl-coenzyme A CoA transferase, decreased butyrate-producing bacteria <i>Butyrivibrio fibrisolvens</i>, and increased <i>Bacteroides fragilis</i>, compared to the WT mice. Serum inflammation cytokines (IL-6, IL-17, and IFN-\u03b3) and LPS were elevated in TDP43 mice. EGCs from TDP43 mice showed aggregation of hmTDP43 associated with increased GFAP and ionized calcium-binding adaptor molecule (IBA1, a microglia marker). TDP43 mice treated with butyrate or probiotic VSL#3 had significantly increased rotarod time, increased intestinal mobility and decreased permeability, compared to the untreated group. Butyrate or probiotics treatment decreased the expression of GFAP, TDP43, and increased \u03b1-SMA, ZO-1, and Claudin-5 in the colon, spinal cord, and brain. Also, butyrate or probiotics treatment enhanced the Butyryl-coenzyme A CoA transferase, <i>Butyrivibrio fibrisolvens</i>, and reduced inflammatory cytokines in TDP43 mice. The TDP43 EGCs treated with butyrate or probiotics showed reduced GFAP, IBA1, and TDP43 aggregation. Restoring the intestinal and microbial homeostasis by beneficial bacteria and metabolites provide a potential therapeutic strategy to treat ALS."

other_pipes = [pipe for pipe in model.pipe_names if pipe != "ner"]
with model.disable_pipes(*other_pipes):
    optimizer = model.initialize()
    epochs = 20

    for epoch in range(epochs):
        random.shuffle(data)
        losses = {}
        batches = minibatch(data, size=2)
        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = model.make_doc(text)
                example = Example.from_dict(doc, {"entities": annotations})
                examples.append(example)
            model.update(examples, drop=0.2, sgd=optimizer, losses=losses)
        print(f"Epoch: {epoch}, Losses: {losses}")


doc = model(data[0][0])

for ent in doc.ents:
    print(ent.text, ent.label_)
