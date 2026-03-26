from transformers import TrainingArguments, Trainer
from datasets import Dataset, Features, Value, ClassLabel
import numpy as np
from seqeval.metrics import classification_report
from transformers import BertTokenizerFast, BertForTokenClassification, ProgressCallback, TrainerCallback
import pickle
import spacy
from transformers import DataCollatorForTokenClassification
import os
import pandas as pd


'''
GitHub: https://github.com/ncbi-nlp/bluebert
HuggingFace: https://huggingface.co/bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12
    
Here it is this version implemented: BlueBERT-Base, Uncased, PubMed (bluebert_pubmed_uncased_L-12_H-768_A-12)

'''

'''
EQUAL WEIGHTS, NOTHING DONE ABOUT ASSIGNING DIFFERENT WEIGHTS TO GOLDEN, SILVER, BRONZE DATASETS.

Look into example.weight in spaCy, I didn't manage to do that.

Otherwise do resampling with replacement, if you wanna run with diffferent weight assigned to each dataset.
'''


'''
This is heavily based on this tutorial: https://medium.com/@maneyogesh065/ficne-tuning-biobert-for-custom-named-entity-recognition-a-complete-guide-a05b124edda0
'''

def load_pickle(datatype: str) -> pd.DataFrame:
    allowed_values = ["train", "dev"]
    if datatype not in allowed_values:
        raise ValueError(f"{datatype} value not allowed. Allowed values: {allowed_values}")
    cwd = os.getcwd()
    folder = os.path.basename(os.path.dirname(os.path.abspath("BlueBERT.py")))
    if cwd.endswith(folder):
        cwd = os.path.dirname(cwd)
        cwd = os.path.dirname(cwd)
        path = os.path.join(cwd, "Data", f"{datatype}", f"{datatype}.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

train_data = load_pickle("train")
dev_data = load_pickle("dev")

#tokenizing
nlp = spacy.blank("en")  # maybe only keep tokenizer (see documentation)
nlp.add_pipe('sentencizer') # needed for data preprocessing for BlueBERT

# preprocessing for BlueBERT with BIO tagging.

def doc_to_sentences(data):

    """
    data: list of (text, {"spans": [(start_char, end_char, label), ...]})
    returns: list of (token_list, label_list) per sentence

    """

    sentences = []
    labels_list = []

    for text, meta in data:
        spacy_doc = nlp(text)
        entities = meta.get("entities", [])

        # assign BIO labels "outside" ("O")
        doc_labels = ["O"] * len(spacy_doc)

        for tok_idx, tok in enumerate(spacy_doc):
            for start_char, end_char, label in entities:
                if tok.idx == start_char:
                    doc_labels[tok_idx] = f"B-{label}"
                elif start_char < tok.idx < end_char:
                    doc_labels[tok_idx] = f"I-{label}"

        # split into sentences
        for sent in spacy_doc.sents:
            start = sent.start
            tokens = [tok.text for tok in sent]
            labels = doc_labels[start: start + len(tokens)]
            sentences.append(tokens)
            labels_list.append(labels)

    return sentences, labels_list

sentences, labels_list = doc_to_sentences(train_data)

dev_sentences, dev_labels_list = doc_to_sentences(dev_data)

# getting all unique BIO tags

bio_tags = list(set(label for sublist in labels_list for label in sublist))

def convert_to_dataset(sentences, labels, bio_tags):
    
    '''
    Convert to HuggingFace Dataset format
    '''
    
    label_map = {tag: i for i, tag in enumerate(bio_tags)}

    label_ids = [[label_map[label] for label in sentence_labels] for sentence_labels in labels]
    
    return Dataset.from_dict({
        "tokens": sentences,
        "ner_tags": label_ids
    }, features = Features({
        "tokens": [Value("string")],
        "ner_tags": [ClassLabel(names = bio_tags)]
    }))


train_dataset = convert_to_dataset(sentences, labels_list, bio_tags)

dev_dataset = convert_to_dataset(dev_sentences, dev_labels_list, bio_tags)

# wordPiece tokenizer from BlueBERT (orgiginally from BERT)

tokenizer = BertTokenizerFast.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation = True,
        is_split_into_words = True,
        padding = "max_length",
        max_length = 128
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            # Special tokens have word_id set to None
            if word_idx is None:
                label_ids.append(-100)  # -100 will be ignored in loss calculation
            # For the first token of a word
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For subsequent subword tokens
            else:
                # If it's a B- tag, convert to I- for continuation
                if bio_tags[label[word_idx]].startswith("B-"):
                    i_tag = "I-" + bio_tags[label[word_idx]][2:]
                    label_ids.append(bio_tags.index(i_tag))
                else:
                    label_ids.append(label[word_idx])
            previous_word_idx = word_idx
            
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_train = train_dataset.map(tokenize_and_align_labels, batched = True)
tokenized_dev = dev_dataset.map(tokenize_and_align_labels, batched = True)


#Instantiating model

model = BertForTokenClassification.from_pretrained(
    "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",   #WHY DIFFERENT FROM THE PREVIOUS ONE
    num_labels = len(bio_tags))


'''
After instantiating the model you might get:

```
Some weights of BertForTokenClassification were not initialized from the model checkpoint at bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12 and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

It's fine, since BlueBERT is not tuned for NER, this is our job to do.

We basically have:
    
BlueBERT (pretrained)
    +
Fresh NER head (random weights)
    ↓
Fine-tune on our data
    ↓
BlueBERT-NER 
'''


training_args = TrainingArguments(
    output_dir = "./bluebert-ner-custom",
    eval_strategy = "epoch",
    learning_rate = 3e-5,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    num_train_epochs = 10,
    weight_decay = 0.01,
    save_strategy = "epoch",
    load_best_model_at_end = True,
    metric_for_best_model ="f1"
)

data_collator = DataCollatorForTokenClassification(tokenizer = tokenizer)


def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis = 2)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [bio_tags[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [bio_tags[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = classification_report(true_labels, true_predictions, output_dict=True)
    
    # Extract metrics
    overall_f1 = results["micro avg"]["f1-score"]
    return {
        "precision": results["micro avg"]["precision"],
        "recall": results["micro avg"]["recall"],
        "f1": overall_f1,
    }


trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_train,
    eval_dataset = tokenized_dev,
    data_collator = data_collator,
    compute_metrics = compute_metrics,
    callbacks = [ ProgressCallback()
                 ]
)

# Start training
try:
    trainer.train()
except KeyboardInterrupt:
    trainer.save_model("./bluebert-ner-best")
    tokenizer.save_pretrained("./bluebert-ner-best")


# saving the model
model.save_pretrained("./biobert-ner-final")
tokenizer.save_pretrained("./biobert-ner-final")

############################################################################
#################### E V A L U A T I O N ####################################
############################################################################


# test_results = trainer.evaluate(tokenized_dev)
# print(f"Test results: {test_results}")

# # Detailed entity-level evaluation


# predictions, labels, _ = trainer.predict(tokenized_test)
# predictions = np.argmax(predictions, axis=2)

# # Remove ignored index (special tokens)
# true_predictions = [
#     [bio_tags[p] for (p, l) in zip(prediction, label) if l != -100]
#     for prediction, label in zip(predictions, labels)
# ]
# true_labels = [
#     [bio_tags[l] for (p, l) in zip(prediction, label) if l != -100]
#     for prediction, label in zip(predictions, labels)
# ]

# # Print detailed classification report
# print(classification_report(true_labels, true_predictions))

# ##########################################################
# ############Model Deployment and Inference################
# ##########################################################



# #inference function

# def predict_entities(text, model, tokenizer):
#     # Tokenize the text
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
#     # Get predictions
#     with torch.no_grad():
#         outputs = model(**inputs)
#         predictions = torch.argmax(outputs.logits, axis=2)
    
#     # Convert predictions to labels
#     predicted_labels = [bio_tags[p] for p in predictions[0].numpy()]
    
#     # Align predictions with original tokens
#     tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
#     # Extract entities
#     entities = []
#     current_entity = None
    
#     for token, label in zip(tokens, predicted_labels):
#         # Skip special tokens
#         if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
#             continue
            
#         # Handle subwords
#         if token.startswith("##"):
#             if current_entity:
#                 current_entity["text"] += token[2:]
#             continue
        
#         if label.startswith("B-"):
#             if current_entity:
#                 entities.append(current_entity)
#             entity_type = label[2:]  # Remove B- prefix
#             current_entity = {"type": entity_type, "text": token}
#         elif label.startswith("I-") and current_entity and current_entity["type"] == label[2:]:
#             current_entity["text"] += " " + token
#         elif label == "O":
#             if current_entity:
#                 entities.append(current_entity)
#                 current_entity = None
    
#     # Don't forget the last entity
#     if current_entity:
#         entities.append(current_entity)
    
#     return entities
