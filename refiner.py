import os
import time
import csv
import sys
import tiktoken
from mistralai import Mistral

csv.field_size_limit(sys.maxsize)

TRIP, FLAG = 0, False

# Constants
MODEL_NAME = "mistral-large-latest"
NUM_NODES = 47031
TOKENS_PER_MINUTE = 500_000
SAFETY_MARGIN = 0.9

# Setup Mistral API
os.environ["MISTRAL_API_KEY"] = "" # Add your Mistral API key here
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# Token estimation function
def estimate_tokens(prompt: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(prompt))

# Input & output files
INPUT_CSV = f"hetionet_descriptions_{NUM_NODES}.csv"
OUTPUT_CSV = f"hetionet_descriptions{NUM_NODES}.csv"

# Read input CSV
with open(INPUT_CSV, "r", encoding="utf-8") as infile, open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:
    reader = csv.DictReader(infile)
    fieldnames = ["identifier", "kind", "name", "refined_description", "url"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    c = 0
    for row in reader:
        identifier = row["identifier"]
        kind = row["kind"]
        name = row["name"]
        description = row["description"]
        url = row["url"]
        c += 1
        # if c <= 82:
        #     continue

        if len(description) > 5000:
            description = description[:5000] + "..."

        if not description.strip():
            refined_description = "No description available."
        else:
            prompt = f"""Summarize the following description in 1-3 lines using clear, scientific language, as if taken from a textbook. Retain key biological and medical terms while removing unnecessary details, references, or source mentions.

Entity Details:
- Identifier: {identifier}
- Kind: {kind}
- Name: {name}
- Original Description: {description}

Provide only the refined definition, without introductory phrases or extra formatting, without mentioning sources or where the data was obtained. If the description is not present, write a proper definition of the entity based on the name and kind with the initial instructions."""

            try:
                # Send prompt to Mistral
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
            refined_description = response.choices[0].message.content.strip()

            # Estimate tokens and apply dynamic delay after the call
            token_count = estimate_tokens(prompt)
            max_tokens = TOKENS_PER_MINUTE * SAFETY_MARGIN
            delay = max(1.0, 60 * token_count / max_tokens)
            print(f"[INFO] Estimated tokens: {token_count}, applying delay of {delay:.2f} seconds.")
            time.sleep(delay)

        print(f"Refined description for {name} ({identifier}) {c}:\n{refined_description}\n")
        writer.writerow({
            "identifier": identifier,
            "kind": kind,
            "name": name,
            "refined_description": refined_description,
            "url": url
        })

print(f"Refined descriptions saved to {OUTPUT_CSV}")
