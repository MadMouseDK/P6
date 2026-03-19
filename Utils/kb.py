import spacy
import os
from urllib.parse import urlparse
import rdflib
import re
import pickle
import json
import pandas as pd
# Only compatible with python 3.10, for newer version we will have to change the 
#from intermine.webservice import Service

path = os.getcwd()
if path.endswith("Utils"):
        cwd = os.path.dirname(path)
path = os.path.join(path, "Data", "raw", "GutBrainIE_Full_Collection_2026", "Annotations")

with open(os.path.join(path, "uris.csv"), "r") as f:
    URIs = [uri.rstrip() for i, uri in enumerate(f.readlines()) if i != 0]

URIs_per_domain = {}
for uri in URIs:
    domain = urlparse(uri).netloc
    if domain == "www.ebi.ac.uk":
        uri = uri.split("/")[-1]
        uri = "http://purl.obolibrary.org/obo/" + uri
    if domain not in URIs_per_domain.keys():
        URIs_per_domain[domain] = [uri]
        continue
    URIs_per_domain[domain].append(uri)

ontologies = {}
for uri in URIs_per_domain["purl.obolibrary.org"]:
    ontology = uri.split("/")[-1]
    ontology = re.match(r"([^\_]*)", ontology).group(0)
    if ontology not in ontologies.keys():
        ontologies[ontology] = [uri]
        continue
    ontologies[ontology].append(uri)
ontologies["efo"] = URIs_per_domain["www.ebi.ac.uk"]
ontologies["edam"] = URIs_per_domain["edamontology.org"]

synoynms_labls = {}
definitions = {}

def query_definition(graph: rdflib.Graph, query: str) -> None:
    query = query.replace("'", "")
    for item in graph.query(query):
        uri = str(item["s"])
        definition = str(item["definition"])
        definitions[uri] = definition


def query_synonyms(graph: rdflib.Graph, query: str) -> None:
    query = query.replace("'", "")
    for item in graph.query(query):
        uri = str(item["s"])
        label = str(item["label"])
        if item["synonym"] is None:
            synoynms_labls[uri] = [label]
        elif uri not in synoynms_labls.keys():
            synoynms_labls[uri] = [label]
            synoynms_labls[uri].append(str(item["synonym"]))
        else:
            synoynms_labls[uri].append(str(item["synonym"]))


path = os.getcwd()
path = os.path.join(path, "Data", "kb")
for ontology in ontologies.keys():
    if ontology.lower() == "omit":
        continue
    uris = ontologies[ontology]
    uris = tuple(f"<{uri}>" for uri in uris)
    graph = rdflib.Graph()
    graph_db = f"{ontology.lower()}.owl"
    graph.parse(os.path.join(path, graph_db), format="xml")

    query_definitions = f"""
        PREFIX obo-term: <http://purl.obolibrary.org/obo/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?s ?definition
        WHERE {{
            ?s rdfs:label ?label . 
            ?s obo-term:IAO_0000115 ?definition . #AIO_0000115 is code for definition
            FILTER (?s IN{uris})
        }}
        """
    query_definition(graph, query_definitions)

    if ontology.lower() == "foodon":
        query_labels_synonyms = f"""
        PREFIX obo-term: <http://purl.obolibrary.org/obo/>
        PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?s ?label ?synonym
        WHERE {{
            ?s rdfs:label ?label . 
            OPTIONAL {{?s oboInOwl:hasSynonym ?synonym .}}
            FILTER (?s IN{uris})
        }}
        """
    elif ontology.lower() == "stato":
        query_labels_synonyms = f"""
        PREFIX obo-term: <http://purl.obolibrary.org/obo/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?s ?label ?synonym
        WHERE {{
            ?s rdfs:label ?label . 
            OPTIONAL {{?s obo-term:IAO_0000118 ?synonym .}}
            FILTER (?s IN{uris})
        }}
        """
    else:
        query_labels_synonyms = f"""
        PREFIX obo-term: <http://purl.obolibrary.org/obo/>
        PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?s ?label ?synonym
        WHERE {{
            ?s rdfs:label ?label . 
            OPTIONAL {{?s oboInOwl:hasExactSynonym ?synonym .}}
            FILTER (?s IN{uris})
        }}
        """
    query_synonyms(graph, query_labels_synonyms)



uris = URIs_per_domain["id.nlm.nih.gov"]
uris = ["http://id.nlm.nih.gov/mesh/2026/" + uri.strip("/")[-1] for uri in uris]
uris = tuple(f"<{uri}>" for uri in uris)
graph = rdflib.Graph()
graph.parse(os.path.join(path, "mesh2026.nt"), format="nt")

q = f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
SELECT ?s ?label ?definition
WHERE {{
	?s rdfs:label ?label .
    OPTIONAL {{?s meshv:note ?definition}} 
    FILTER (?s IN{uris})
}}
"""
q = q.replace("'", "")
for item in graph.query(q):
    uri = str(item["s"])
    definition = str(item["definition"])
    label = str(item["label"])
    definitions[uri] = definition
    synoynms_labls[uri] = [label]


for uri in URIs_per_domain["www.ncbi.nlm.nih.gov"]:
    id = uri.split("/")[-1]
    if "gene" in uri:
        id = id + ".jsonl"
        with open(os.path.join(path, "NCBI,NLM", id), "r") as f:
            info = json.load(f)
        definitions[uri] = info["description"]
        try:
            synoynms_labls[uri] = info["synonyms"]
            synoynms_labls[uri].append(info["symbol"])
        except:
            synoynms_labls[uri] = info["symbol"]
    elif "mesh" in uri:
        id = id + ".txt"
        with open(os.path.join(path, "NCBI,NLM", id), "r") as f:
            info = f.readlines()
        definitions[uri] = info[2] + info[3]
        synoynms_labls[uri] = info[1]


for uri in URIs_per_domain["snomed.info"]:
    id = uri.split("/")[-1]
    file_name = f"{id}.json"
    base_path = os.path.join(os.getcwd(), "Data", "kb", "snomed", "concept", file_name)
    if not os.path.exists(base_path):
        continue
    with open(base_path, "r") as f:
        data = json.load(f)
    synoynms_labls[uri] = [data["fsn"]["term"]]
    definitions[uri] = ""

del graph

def load_and_clean() -> pd.DataFrame:
    df_start = pd.read_csv(os.path.join(path, "Data/kb/GutBrainIE 2026 - Concepts - Starting Concepts.csv"))
    df_added = pd.read_csv(os.path.join(path, "Data/kb/GutBrainIE 2026 - Concepts - Added Concepts.csv"))
    df_custom = pd.read_csv("Data/kb/GutBrainIE 2026 - Concepts - Custom Concepts.csv")

    column_names = {key: item for key, item in zip(df_added.columns, list(df_added.iloc[0]))}
    df_added = df_added.rename(column_names, axis=1)
    df_added = df_added.drop(columns="used in document(s)")
    df_added = df_added.drop(index=0)

    column_names = {key: item for key, item in zip(df_custom.columns, list(df_custom.iloc[0]))}
    df_custom = df_custom.rename(column_names, axis=1)
    df_custom = df_custom.drop(columns="used in document(s)")
    df_custom = df_custom.drop(index=0)
    df_custom = df_custom.dropna(subset="uri")

    df = pd.concat([df_start, df_custom, df_added])
    df = df.drop(columns="labels")
    return df

def extract_uris(df: pd.DataFrame, uris: list) -> pd.DataFrame:
    def replace_http(uris: list) -> list:
        for i, uri in enumerate(uris):
            uri = uri.removeprefix("http")
            uris[i] = "https" + uri
        return uris

    def contains(uri: str, acceptable_uri: list) -> bool:
        if not pd.notna(uri):
            return False
        return uri in acceptable_uri
    
    if any(["https" in uri for uri in uris]):
        uris = replace_http(uris)
    condition = df.apply(lambda row: contains(row["uri"], uris), axis=1)
    return df[condition]

def add_entries(df: pd.DataFrame):
    for idx, row in df.iterrows():
        if pd.notna(row["definition"]):
            definitions[row["uri"]] = row["definition"]
        if pd.notna(row["name"]):
            synoynms_labls[row["uri"]] = row["name"]

df = load_and_clean()

uris = URIs_per_domain["uts.nlm.nih.gov"]
df_extract = extract_uris(df, uris)
add_entries(df_extract)

uris = URIs_per_domain["w3id.org"]
df_extract = extract_uris(df, uris)
add_entries(df_extract)


# saves the definitions and labels/synonyms
with open("kb_definitions.pickle", "wb") as f:
    pickle.dump(definitions, f)

with open("kb_synonyms_labels.pickle", "wb") as f:
    pickle.dump(synoynms_labls, f)



