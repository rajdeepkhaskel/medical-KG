import json
import requests
import csv
import time
from bs4 import BeautifulSoup
from googlesearch import search  # Requires 'google' package

START_NODES = 1000
NUM_NODES = 47031  # Limit of nodes to crawl

TRIP, FLAG = 0, False

# Function to extract text from a URL
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            return soup.get_text(separator=" ", strip=True)  # Extract all text
    except Exception as e:
        return None
    return None

# Function to perform a Google search and extract content from the first result
def search_google(query):
    global TRIP, FLAG
    try:
        try:
            search_results = list(search(query, num=1, stop=1, pause=2))  # Fetch only top 1 result
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
            return extract_text_from_url(search_results[0]), search_results[0]
    except Exception as e:
        return None, None
    # Sleep to avoid Google rate-limiting
    # time.sleep(0.35)
    return None, None

# Load Hetionet JSON
with open("hetionet-v1.0.json", "r", encoding="utf-8") as f:
    data = json.load(f)

nodes = data.get("nodes", [])

# Prepare CSV file
output_file = f"hetionet_descriptions_{NUM_NODES}.csv"
fields = ["identifier", "kind", "name", "description", "url"]

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(fields)  # Write header

    count = 0  # Limit to NUM_NODES nodes
    for node in nodes:
        if count < START_NODES: 
            count += 1
            continue
        # if count >= NUM_NODES: break  # Stop after NUM_NODES nodes

        kind = node.get("kind", "")
        identifier = node.get("identifier", "")
        name = node.get("name", "")
        data_info = node.get("data", {})
        url = data_info.get("url", None)
        source = data_info.get("source", None)
        existing_description = data_info.get("description", None)
        url1 = url
        description = None

        # Step 1: Try extracting from the URL
        if url:
            description = extract_text_from_url(url)

        # Step 2: If no URL or failed fetch, search using the source
        if not description and source:
            description, url1 = search_google(source)

        # Step 3: If no source, use the existing description
        if not description and existing_description:
            description = existing_description

        # Step 4: If no description, search using the name
        if not description:
            description, url1 = search_google(name)

        # Step 5: If still no description, set as "Not Found"
        if not description:
            print(f"No description found for {name}, {count}")
            description = "Not Found"
        else:
            print(f"Found description for {name}, {count}")

        # Save to CSV
        writer.writerow([identifier, kind, name, description, url1])
        count += 1

print(f"Saved {count} nodes to {output_file}")