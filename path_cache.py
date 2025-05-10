import json
import networkx as nx
import os
import time
import requests
import csv
import sys
import tiktoken
import spacy
from bs4 import BeautifulSoup
from googlesearch import search
from mistralai import Mistral

csv.field_size_limit(sys.maxsize)
nlp = spacy.load("en_core_web_sm")

MODEL_NAME = "mistral-large-latest"
os.environ["MISTRAL_API_KEY"] = "" # Add your Mistral API key here
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

TOKENS_PER_MINUTE = 500_000
SAFETY_MARGIN = 0.9

TRIP, FLAG = 0, False
desc_cache = {}

def estimate_tokens(prompt: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(prompt))

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            return soup.get_text(separator=" ", strip=True)
    except Exception:
        return None
    return None

def search_google(query):
    global TRIP, FLAG
    try:
        try:
            search_results = list(search(query, num=1, stop=1, pause=2))
            FLAG = False
            TRIP = 0
        except:
            print("Sleeping for 2 seconds for excessive calls!")
            if FLAG: TRIP += 1
            FLAG = True
            if TRIP > 10: exit(0)
            time.sleep(2)
            search_results = list(search(query, num=1, stop=1, pause=2))
        if search_results:
            return extract_text_from_url(search_results[0])
    except Exception:
        return None
    return None

def refine_description(identifier, kind, name, description):
    if len(description) > 5000:
        description = description[:5000] + "..."
    if not description.strip():
        return "No description available."
    prompt = f"""Summarize the following description in 1-3 lines using clear, scientific language, as if taken from a textbook. Retain key biological and medical terms while removing unnecessary details, references, or source mentions.

Entity Details:
- Identifier: {identifier}
- Kind: {kind}
- Name: {name}
- Original Description: {description}

Provide only the refined definition, without introductory phrases or extra formatting, without mentioning sources or where the data was obtained. If the description is not present, write a proper definition of the entity based on the name and kind with the initial instructions."""
    try:
        response = client.chat.complete(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        FLAG = False
        TRIP = 0
    except:
        print("Excessive API calls, sleeping for 60 seconds.")
        if FLAG: TRIP += 1
        FLAG = True
        if TRIP > 10: exit(0)
        time.sleep(60)
        response = client.chat.complete(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
    token_count = estimate_tokens(prompt)
    max_tokens = TOKENS_PER_MINUTE * SAFETY_MARGIN
    delay = max(1.0, 60 * token_count / max_tokens)
    print(f"[INFO] Estimated tokens: {token_count}, applying delay of {delay:.2f} seconds.")
    time.sleep(delay)
    return response.choices[0].message.content.strip()

def fetch_description(G, node_id):
    if node_id in desc_cache:
        return desc_cache[node_id]

    node = G.nodes[node_id]
    identifier = node.get("identifier", "")
    kind = node.get("kind", "")
    name = node.get("name", "")
    data_info = node.get("data", {})
    url = data_info.get("url", None)
    source = data_info.get("source", None)
    existing_description = data_info.get("description", None)

    description = None

    if url:
        description = extract_text_from_url(url)
    if not description and source:
        description = search_google(source)
    if not description and existing_description:
        description = existing_description
    if not description:
        description = search_google(name)
    if not description:
        description = "No description available."

    refined = refine_description(identifier, kind, name, description)
    desc_cache[node_id] = refined
    return refined

def load_hetionet(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    G = nx.MultiDiGraph()
    for node in data['nodes']:
        G.add_node(str(node['identifier']), **node)
    for edge in data['edges']:
        source = str(edge['source_id'][1])
        target = str(edge['target_id'][1])
        G.add_edge(source, target, **edge)
    print(f"Loaded {len(data['nodes'])} nodes and {len(data['edges'])} edges from Hetionet.")
    return G

def extract_phrases(query):
    doc = nlp(query)
    phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
    return phrases

def retrieve_nodes(G, query):
    phrases = extract_phrases(query)
    matches = {}
    for nid, d in G.nodes(data=True):
        name = d.get('name', '').lower()
        score = sum(phrase in name for phrase in phrases)
        if score > 0:
            matches[nid] = score
    sorted_matches = sorted(matches.keys(), key=lambda nid: -matches[nid])
    return sorted_matches

def find_paths(G, sources, targets, max_paths=5):
    paths = []
    for s in sources:
        for t in targets:
            if s != t and nx.has_path(G, s, t):
                path = nx.shortest_path(G, s, t)
                paths.append(path)
                if len(paths) >= max_paths:
                    return paths
    return paths

def paths_to_text(G, paths):
    output = []
    involved_nodes = set()
    for path in paths:
        involved_nodes.update(path)

    for node_id in involved_nodes:
        fetch_description(G, node_id)  # fetch + cache description for nodes in paths

    for path in paths:
        segs = []
        for i in range(len(path) - 1):
            a, b = path[i], path[i+1]
            edge_data = G.get_edge_data(a, b)
            if edge_data:
                edge = next(iter(edge_data.values()))
                rel = edge.get('kind', 'related_to').replace('_', ' ')
                s_name = G.nodes[a].get('name', 'Unknown')
                t_name = G.nodes[b].get('name', 'Unknown')
                segs.append(f"'{s_name}' {rel} '{t_name}'")
        output.append(' -> '.join(segs))
    return output

def generate_answer(prompt):
    token_count = estimate_tokens(prompt)
    max_tokens_per_request = TOKENS_PER_MINUTE * SAFETY_MARGIN
    delay = max(1.0, 60 * token_count / max_tokens_per_request)
    response = client.chat.complete(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"[INFO] Estimated tokens: {token_count}, applying delay of {delay:.2f} seconds.")
    time.sleep(delay)
    return response.choices[0].message.content.strip()

def process_query(json_path, query):
    G = load_hetionet(json_path)
    matched = retrieve_nodes(G, query)
    if not matched:
        return "No relevant nodes found."

    paths = find_paths(G, matched, matched)
    if not paths:
        return "No connecting paths found."

    print(f"\n[INFO] Found {len(paths)} paths connecting the nodes.\nPaths are:\n{paths}\n")

    texts = paths_to_text(G, paths)
    prompt = "You are a biomedical expert system. You have access to the following knowledge graph paths:\n\n"
    for i, t in enumerate(texts, 1):
        prompt += f"Path {i}: {t}\n"
    prompt += "\nUser's original query:\n"
    prompt += query

    return generate_answer(prompt)

# Example usage
if __name__ == "__main__":
    json_file = "hetionet-v1.0.json"
    user_query = "How does metformin affect Parkinson's disease?"
    answer = process_query(json_file, user_query)
    print("\nAnswer:\n", answer)