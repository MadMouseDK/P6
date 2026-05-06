import os
import json
import pickle
import warnings
from collections import defaultdict
import spacy

warnings.filterwarnings("ignore")


allowed_entities = {
    ("anatomical location", "human"):          ["located in"],
    ("anatomical location", "animal"):         ["located in"],
    ("bacteria", "bacteria"):                  ["interact"],
    ("bacteria", "chemical"):                  ["interact"],
    ("bacteria", "drug"):                      ["interact"],
    ("bacteria", "DDF"):                       ["influence"],
    ("bacteria", "gene"):                      ["change expression"],
    ("bacteria", "human"):                     ["located in"],
    ("bacteria", "animal"):                    ["located in"],
    ("bacteria", "microbiome"):                ["part of"],
    ("chemical", "anatomical location"):       ["located in"],
    ("chemical", "human"):                     ["located in"],
    ("chemical", "animal"):                    ["located in"],
    ("chemical", "chemical"):                  ["interact"],
    ("chemical", "microbiome"):                ["impact"],
    ("chemical", "bacteria"):                  ["impact"],
    ("chemical", "DDF"):                       ["influence"],
    ("chemical", "gene"):                      ["change expression"],
    ("dietary supplement", "bacteria"):        ["impact"],
    ("dietary supplement", "microbiome"):      ["impact"],
    ("dietary supplement", "DDF"):             ["influence"],
    ("dietary supplement", "gene"):            ["change expression"],
    ("dietary supplement", "human"):           ["administered"],
    ("dietary supplement", "animal"):          ["administered"],
    ("drug", "chemical"):                      ["interact"],
    ("drug", "drug"):                          ["interact"],
    ("drug", "bacteria"):                      ["impact"],
    ("drug", "microbiome"):                    ["impact"],
    ("drug", "DDF"):                           ["change effect"],
    ("drug", "gene"):                          ["change expression"],
    ("drug", "human"):                         ["administered"],
    ("drug", "animal"):                        ["administered"],
    ("food", "bacteria"):                      ["impact"],
    ("food", "microbiome"):                    ["impact"],
    ("food", "DDF"):                           ["influence"],
    ("food", "gene"):                          ["change expression"],
    ("food", "human"):                         ["administered"],
    ("food", "animal"):                        ["administered"],
    ("DDF", "anatomical location"):            ["strike"],
    ("DDF", "bacteria"):                       ["change abundance"],
    ("DDF", "microbiome"):                     ["change abundance"],
    ("DDF", "chemical"):                       ["interact"],
    ("DDF", "DDF"):                            ["affect"],
    ("DDF", "human"):                          ["target"],
    ("DDF", "animal"):                         ["target"],
    ("human", "biomedical technique"):         ["used by"],
    ("animal", "biomedical technique"):        ["used by"],
    ("microbiome", "biomedical technique"):    ["used by"],
    ("microbiome", "anatomical location"):     ["located in"],
    ("microbiome", "human"):                   ["located in"],
    ("microbiome", "animal"):                  ["located in"],
    ("microbiome", "gene"):                    ["change expression"],
    ("microbiome", "DDF"):                     ["is linked to"],
    ("microbiome", "microbiome"):              ["compared to"],
}
allowed_labels = {
    "anatomical location", "animal", "bacteria", "biomedical technique",
    "chemical", "DDF", "dietary supplement", "drug", "food", "gene",
    "human", "microbiome", "statistical technique",
}

def get_path():
    path = os.getcwd()
    for _ in range(6):
        if os.path.basename(path) == "P6": #basename find name of last path  a/b/c -> "c"
            return path
        path = os.path.dirname(path)
    raise RuntimeError("Path Error")



# The pickle file maps:  URI -> [synonym1, synonym2, ...]
# We flip it to:         synonym -> URI
def build_kb_index(path):
    with open(path, "rb") as f:
        kb = pickle.load(f)  

    index = {}
    for uri, synonyms in kb.items():
        for synonym in synonyms:
            key = synonym.lower().strip()
            if key not in index:  # keep the first URI if there are duplicates
                index[key] = uri

    # Sorted with longest entity first, since it stops at first match
    # e.g. "Parkinson's disease" is preferred over just "disease" if both appear.
    sorted_synonyms = sorted(index.items(), key=lambda x: len(x[0]), reverse=True)
    return index, sorted_synonyms


def get_uri(text_span, kb_index, sorted_synonyms):
    key = text_span.lower().strip() #clean entity
    # 1. Exact match
    if key in kb_index:
        return kb_index[key]

    # 2. Try without trailing "s" (simple plural, e.g. "neurons" -> "neuron")
    if key.endswith("s") and len(key) > 3 and key[:-1] in kb_index:
        return kb_index[key[:-1]]

    # 3. Substring match: find the longest KB synonym that appears inside the
    for synonym, uri in sorted_synonyms:
        if synonym in key:
            return uri

    # If not in kb, return None
    return None


def load_documents(path, datatype):
    #Load all documents for the given dataset split. Returns a dict: pmid -> doc
    path = os.path.join(path, "Data", "raw", "Annotations")

    if datatype == "dev":
        path = os.path.join(path, "Dev", "json_format", "dev.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Train: merge all quality tiers into one dict
    train_files = [
        os.path.join(path, "Train", "gold_quality",   "json_format", "train_gold.json"),
        os.path.join(path, "Train", "silver_quality", "json_format", "train_silver_2025.json"),
        os.path.join(path, "Train", "silver_quality", "json_format", "train_silver.json"),
        os.path.join(path, "Train", "bronze_quality", "json_format", "train_bronze.json"),
    ]
    all_docs = {}
    for path in train_files:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                all_docs.update(json.load(f))
    return all_docs



# Extract relations using spacy
# For each document:
# -Run the NER model on the title and abstract
# -Link each predicted entity to a URI
# -For every pair of entities, check if allowed_entities has a predicate
# -Add unique (subject_uri, predicate, object_uri) triples to the results
def extract_from_predictions(documents, nlp, kb_index, sorted_synonyms):
    results = {}  # pmid -> list of relation dicts

    for idx, (pmid, doc_data) in enumerate(documents.items(), 1):

        # Get the title and abstract text
        meta = doc_data.get("metadata", {})
        texts = [s for s in [meta.get("title", ""), meta.get("abstract", "")] if s]

        seen = set()      # tracks which (subject, predicate, object) we've already added
        relations = []    # the relations we find for this document

        for text in texts:
            # Run NER model to find entity spans
            doc = nlp(text)
            entities = []
            for ent in doc.ents:
                if ent.label_ not in allowed_labels:
                    continue
                uri = get_uri(ent.text, kb_index, sorted_synonyms)
                if uri is not None:
                    entities.append({"text": ent.text, "label": ent.label_, "uri": uri})

            # Try every pair of entities
            for i, subject in enumerate(entities):
                for j, obj in enumerate(entities):
                    if i == j:
                        continue  # skip same entity
                    if subject["uri"] == obj["uri"]:
                        continue  # skip if they map to the same uri

                    # Look up the predicate for this pair of entity types
                    allowed = allowed_entities.get((subject["label"], obj["label"]))
                    if not allowed:
                        continue  # no relation defined for this pair
                    predicate = allowed[0]

                    # Deduplicate: only add this triple once per document
                    triple = (subject["uri"], subject["label"], predicate, obj["uri"], obj["label"])
                    if triple in seen:
                        continue
                    seen.add(triple)

                    relations.append({
                        "subject_uri":   subject["uri"],
                        "subject_label": subject["label"],
                        "predicate":     predicate,
                        "object_uri":    obj["uri"],
                        "object_label":  obj["label"],
                    })

        results[pmid] = relations

    return results



# Extract relations from ground truth (Gold annotations) 
# Reads the already-labelled mention-level relations from the JSON file and
# aggregates them to concept level (deduplicates by URI triple per document).

def extract_from_annotations(documents):
    results = {}

    for pmid, doc_data in documents.items():
        seen = set()
        relations = []

        for rel in doc_data.get("relations", []):
            s_uri   = rel.get("subject_uri", "")
            s_label = rel.get("subject_label", "")
            pred    = rel.get("predicate", "")
            o_uri   = rel.get("object_uri", "")
            o_label = rel.get("object_label", "")

            # Skip incomplete relations
            if not all([s_uri, s_label, pred, o_uri, o_label]):
                continue
            if s_label not in allowed_labels or o_label not in allowed_labels:
                continue

            # Deduplicate
            triple = (s_uri, s_label, pred, o_uri, o_label)
            if triple in seen:
                continue
            seen.add(triple)

            relations.append({
                "subject_uri":   s_uri,
                "subject_label": s_label,
                "predicate":     pred,
                "object_uri":    o_uri,
                "object_label":  o_label,
            })

        results[pmid] = relations

    return results


#Save result. json->submission and csv -> metrics/debug
def save_json(results, output_path):
    submission = {
        pmid: {"concept_level_relations": relations}
        for pmid, relations in results.items()
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

def save_csv(results, output_path):
    # Count how many times each (subject, predicate, object) triple appears across documents
    frequency = defaultdict(int)
    for relations in results.values():
        for rel in relations:
            key = (rel["subject_uri"], rel["subject_label"], rel["predicate"],
                   rel["object_uri"], rel["object_label"])
            frequency[key] += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("subjectConceptURI;subjectCategory;relationPredicate;objectConceptURI;objectCategory;frequency\n")
        for (s_uri, s_label, pred, o_uri, o_label), freq in sorted(frequency.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{s_uri};{s_label};{pred};{o_uri};{o_label};{freq}\n")



def print_top_relations(results, n=10):
    # Count how often each (subject type, predicate, object type) pattern appears
    pattern_counts = defaultdict(int)
    for relations in results.values():
        for rel in relations:
            pattern = (rel["subject_label"], rel["predicate"], rel["object_label"])
            pattern_counts[pattern] += 1

    print(f"{'Subject':<24} {'Predicate':<22} {'Object':<24} {'Count':>6}")
    print(f"{'-'*24} {'-'*22} {'-'*24} {'-'*6}")
    for (subject, predicate, obj), count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:n]:
        print(f"{subject:<24} {predicate:<22} {obj:<24} {count:>6}")



#Mode is either "predictions" or "annotations"
def main(model_name, datatype, mode):
    path = get_path()
    models = {
        "BioBERT-base":  os.path.join(path, "Models", "BioBERT-base_NER", "output", "model-best"),
        "BioBERT-large": os.path.join(path, "Models", "BioBERT-large_NER", "output", "model-best"), #Not implemented
        "BlueBERT-pubmed": os.path.join(path, "Models", "BlueBERT-pubmed_NER", "output", "model-best"),
        "BlueBERT-mimic": os.path.join(path, "Models", "BlueBERT-mimic_NER", "output", "model-best"),
        "RoBERTa": os.path.join(path, "Models", "RoBERTa_NER", "output", "model-best"),
        "RoBERTa-biomed": os.path.join(path, "Models", "RoBERTa-biomed_NER", "output", "model-best"),
        "SciBERT": os.path.join(path, "Models", "SciBERT_NER", "output", "model-best"),
        "SciBERT-cased": os.path.join(path, "Models", "SciBERT-cased_NER", "output", "model-best"),
    }

    model_path = models.get(model_name, "")
    if not os.path.exists(model_path):
        print(f" Model not found.")
        return

    kb_path = os.path.join(path, "Output", "kb_synonyms_labels.pickle")
    if not os.path.exists(kb_path):
        print(f"Knowledge base file not found")
        return

    # --- Load everything ---
    nlp = spacy.load(model_path) #Load Model
    kb_index, sorted_synonyms = build_kb_index(kb_path) #Build KB index
    documents = load_documents(path, datatype,) #Load docs

    print(f"Model: {model_name} | Dataset: {datatype} | Mode: {mode}")

    # --- Extract relations ---
    if mode == "predictions":
        results = extract_from_predictions(documents, nlp, kb_index, sorted_synonyms)
        tag = f"{model_name.lower().replace('-', '_')}_predictions"
    else:
        results = extract_from_annotations(documents)
        tag = "annotations"

    output   = os.path.join(path, "Output")
    json_path = os.path.join(output, f"{datatype}_concept_level_relations_{tag}.json") #
    csv_path  = os.path.join(output, f"{datatype}_concept_level_relations_{tag}.csv") #CSV if we want metrics in the future

    save_json(results, json_path)
    save_csv(results, csv_path)

    print_top_relations(results)


if __name__ == "__main__":
    main("BioBERT-base", "dev" , "predictions") 
    #main("BioBERT-large", "dev" , "predictions")
    main("BlueBERT-pubmed", "dev" , "predictions")
    main("BlueBERT-mimic", "dev" , "predictions")
    main("RoBERTa", "dev" , "predictions") 
    main("RoBERTa-biomed", "dev" , "predictions") 
    main("SciBERT", "dev" , "predictions") 
    main("SciBERT-cased", "dev" , "predictions") 
