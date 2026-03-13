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


synoynms_labls = {}
definitions = {}
for ontology in ontologies.keys():
    if ontology.lower() == "omit":
        continue
    uris = ontologies[ontology]
    uris = tuple(f"<{uri}>" for uri in uris)

    graph = rdflib.Graph()
    path = os.getcwd()
    path = os.path.join(path, "Data", "kb")
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
    query_definitions = query_definitions.replace("'", "")

    for item in graph.query(query_definitions):
        uri = str(item["s"])
        definition = str(item["definition"])
        definitions[uri] = definition

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
    query_labels_synonyms = query_labels_synonyms.replace("'", "")
    for item in graph.query(query_labels_synonyms):
        uri = str(item["s"])
        label = str(item["label"])
        if item["synonym"] is None:
            synoynms_labls[uri] = [label]
        elif uri not in synoynms_labls.keys():
            synoynms_labls[uri] = [label]
            synoynms_labls[uri].append(str(item["synonym"]))
        else:
            synoynms_labls[uri].append(str(item["synonym"]))



uris = URIs_per_domain["edamontology.org"]
uris = tuple(f"<{uri}>" for uri in uris)
graph = rdflib.Graph()
path = os.getcwd()
path = os.path.join(path, "Data", "kb")
graph.parse(os.path.join(path, "edam.owl"), format="xml")

q = f"""
    PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT ?s ?definition
    WHERE {{
        ?s rdfs:label ?label . 
        ?s oboInOwl:hasDefinition ?definition . #AIO_0000115 is code for definition
        FILTER (?s IN{uris})
    }}
    """
q = q.replace("'", "")
for item in graph.query(q):
    uri = str(item["s"])
    definition = str(item["definition"])
    definitions[uri] = definition

query_labels_synonyms = f"""
    PREFIX obo-term: <http://purl.obolibrary.org/obo/>
    PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT ?s ?label ?synonym
    WHERE {{
        ?s rdfs:label ?label . 
        OPTIONAL {{?s oboInOwl:hasExactSynonym ?synonym .}}
        FILTER (?s IN{uris})
    }}"""
query_labels_synonyms = query_labels_synonyms.replace("'", "")
for item in graph.query(query_labels_synonyms):
    uri = str(item["s"])
    label = str(item["label"])
    if item["synonym"] is None:
        synoynms_labls[uri] = [label]
    elif uri not in synoynms_labls.keys():
        synoynms_labls[uri] = [label]
        synoynms_labls[uri].append(str(item["synonym"]))
    else:
        synoynms_labls[uri].append(str(item["synonym"]))



uris = URIs_per_domain["id.nlm.nih.gov"]
uris = ["http://id.nlm.nih.gov/mesh/2026/" + uri.strip("/")[-1] for uri in uris]
uris = tuple(f"<{uri}>" for uri in uris)
graph = rdflib.Graph()
path = os.getcwd()
path = os.path.join(path, "Data", "kb")
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


# saves the definitions and labels/synonyms
with open("kb_definitions.pickle", "wb") as f:
    pickle.dump(definitions, f)

with open("kb_synonyms_labels.pickle", "wb") as f:
    pickle.dump(synoynms_labls, f)



