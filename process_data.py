import json
import re
import os
from tqdm import tqdm
from dotenv import load_dotenv

from transformers import AutoTokenizer

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')


# Load environment variables from .env file
load_dotenv()

# Define file paths from environment variables or use defaults
RAW_INPUT_PATH = os.getenv("RAW_INPUT_PATH", "raw/data.json")
CLEANED_OUTPUT_PATH = os.getenv("CLEANED_OUTPUT_PATH", "cleaned/cleaned_data.json")

# Define a modular pipeline for data cleaning
class DataCleaningPipeline:
    def __init__(self, steps=None):
        self.steps = steps if steps else []

    def add_step(self, step):
        self.steps.append(step)

    def execute(self, data):
        for step in self.steps:
            data = step(data)
        return data

# Step: Extract address within article text
def extract_singapore_addresses(text):
    """
    Extracts possible Singapore addresses from text using regex patterns.
    """
    address_patterns = [
        # e.g. "123 Orchard Road, Singapore 238888"
        r"\d{1,4}\s+[\w\s\.\-']+,\s*Singapore\s*\d{6}",
        # e.g. "123 Telok Ayer Street, S123456"
        r"\d{1,4}\s+[\w\s\.\-']+,\s*S\d{6}",
        # e.g. "Block 123A Ang Mo Kio Ave 3, Singapore 560123"
        r"[Bb]lock\s*\d{1,4}[A-Z]?\s+[\w\s\.\-']+,\s*(Singapore\s*)?\d{6}",
        # e.g. "21 Tanjong Pagar Road S088444"
        r"\d{1,4}\s+[\w\s\.\-']+\s+S\d{6}"
    ]

    matches = []
    for pattern in address_patterns:
        found = re.findall(pattern, text)
        matches.extend(found)

    return list(set(matches))  # remove duplicates

def extract_addresses(data):
    """
    Extracts addresses from the article text in the data.
    """
    for article in data:
        if 'article_text' in article:
            addresses = extract_singapore_addresses(article['article_text'])
            article['addresses'] = addresses
    return data

# Filter out incomplete data
def filter_articles(data):
    filtered_data = []
    for article in data:
        if 'article_text' not in article or article['article_text'] == '':
            continue
        if 'addresses' not in article or len(article['addresses']) == 0:
            continue
        if 'name' not in article or article['name'] == '':
            continue
        if 'regions' not in article or len(article['regions']) == 0:
            continue
        filtered_data.append(article)
    return filtered_data
        
# Chunk article text into smaller segments
def chunk_text(data, chunk_size=500):
    def chunk_article_text(article):
        words = article['article_text'].split()
        article['article_text'] = [
            " ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)
        ]
        return article

    return [chunk_article_text(article) for article in data]

# Chunk article text into smaller segments based on sentences
def sentence_based_chunk_text(data, max_tokens=400, overlap=50):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    for article in data:
        text = article.get('article_text', '')
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_len = 0

        for i in range(len(sentences)):
            sent = sentences[i]
            token_len = len(tokenizer.tokenize(sent))
            if current_len + token_len > max_tokens and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))
                # Prepare overlap
                if overlap > 0:
                    overlap_tokens = 0
                    overlap_chunk = []
                    j = len(current_chunk) - 1
                    while j >= 0 and overlap_tokens < overlap:
                        sent_tokens = len(tokenizer.tokenize(current_chunk[j]))
                        overlap_tokens += sent_tokens
                        overlap_chunk.insert(0, current_chunk[j])
                        j -= 1
                    current_chunk = overlap_chunk.copy()
                    current_len = sum(len(tokenizer.tokenize(s)) for s in current_chunk)
                else:
                    current_chunk = []
                    current_len = 0
            else:
                current_chunk.append(sent)
                current_len += token_len

        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        article['article_text'] = chunks

    return data


# Classify Singapore region based on extracted postal code
def classify_sg_region_from_address(address):
    """Classify Singapore region (North, South, East, West, Central) based on postal code prefix."""
    match = re.search(r"\bS(?:ingapore)?\s*(\d{6})\b", address)
    if not match:
        return "Unknown"

    prefix = int(match.group(1)[:2])

    if 1 <= prefix <= 9 or prefix in [17, 18, 22, 23]:
        return "Central"
    elif 34 <= prefix <= 37 or 46 <= prefix <= 52:
        return "East"
    elif prefix in [60, 61] or 62 <= prefix <= 70 or prefix == 21:
        return "West"
    elif 75 <= prefix <= 78:
        return "North"
    elif 53 <= prefix <= 57:
        return "North-East"
    else:
        return "Unknown"

def classify_region(data):
    """
    Classify the region of each article based on the extracted addresses.
    """
    for article in data:
        if 'addresses' in article:
            regions = [classify_sg_region_from_address(addr) for addr in article['addresses']]
            article['regions'] = list(set(regions))  # remove duplicates
            # remove 'Unknown' regions
            article['regions'] = [region for region in article['regions'] if region != 'Unknown']
    return data

def classify_venue_type_from_text(text):
    text_lower = text.lower()

    if "hawker" in text_lower or "stall" in text_lower:
        return "hawker"
    elif "food court" in text_lower:
        return "food court"
    elif "bakery" in text_lower:
        return "bakery"
    elif "cafe" in text_lower:
        return "cafe"
    elif "bar" in text_lower or "pub" in text_lower:
        return "pub"
    elif "bistro" in text_lower:
        return "bistro"
    elif "canteen" in text_lower:
        return "canteen"
    elif "restaurant" in text_lower:
        return "restaurant"
    else:
        return "restaurant"  # default fallback

def classify_venue_type(data):
    """
    Classify the venue type of each article based on the article text.
    """
    for article in data:
        if 'article_text' in article:
            article['venue_type'] = classify_venue_type_from_text(article['article_text'])
    return data

# Remove duplicate articles with same url 
def remove_duplicates(data):
    seen_urls = set()
    unique_data = []
    for article in tqdm(data):
        if 'url' in article and article['url'] not in seen_urls:
            seen_urls.add(article['url'])
            unique_data.append(article)
    return unique_data

# Path to the JSON file
file_path = RAW_INPUT_PATH

# Read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

print(f"Loaded {len(data)} articles from {file_path}")

# Initialize the pipeline
pipeline = DataCleaningPipeline([
    remove_duplicates,
    extract_addresses,
    classify_region,
    classify_venue_type,
    filter_articles,
    # chunk_text
    sentence_based_chunk_text
])

# Execute the pipeline
cleaned_data = pipeline.execute(data)

# Debugging output
print(f'Processed {len(cleaned_data)} articles.')
# for article in cleaned_data:
    # print(article['name'])
    # print('chunks:', len(article['article_text']))


# Save output to a new JSON file
output_file_path = CLEANED_OUTPUT_PATH
with open(output_file_path, 'w') as file:
    json.dump(cleaned_data, file, indent=4)

print(f"Cleaned data saved to {output_file_path}")