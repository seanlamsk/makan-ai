import requests
from bs4 import BeautifulSoup
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def fetch_page(url):
    res = requests.get(url, headers=HEADERS)
    res.raise_for_status()
    return BeautifulSoup(res.text, "html.parser")


def parse_article(url, config):
    """Visit article page and extract structured info."""
    try:
        soup = fetch_page(url)

        name_tag = soup.select_one(config["name_selector"])
        location_tag = soup.select_one(config["location_selector"])
        cuisine_tag = soup.select_one(config["cuisine_selector"])

        article_paragraphs = soup.select(config["article_paragraph_selector"])
        article_text = "\n".join([p.get_text(strip=True) for p in article_paragraphs])

        return {
            "url": url,
            "name": name_tag.get_text(strip=True) if name_tag else "",
            "location": location_tag.get_text(strip=True) if location_tag else "",
            "cuisine_type": cuisine_tag.get_text(strip=True) if cuisine_tag else "",
            "article_text": article_text
        }

    except Exception as e:
        print(f"⚠️ Error parsing article {url}: {e}")
        return None


def scrape(config, max_pages=1, delay=1.5):
    """Scrape reviews from a paginated listing page."""
    results = []
    next_url = config["start_url"]

    for i in range(1,max_pages+1):
        print(f"🔎 Scraping listing page: {next_url}")

        soup = fetch_page(next_url)
        article_links = soup.select(config["article_link_selector"])

        article_urls = [
            link["href"] if link["href"].startswith("http") else config["base_url"] + link["href"]
            for link in article_links if link.has_attr("href")
        ]

        for url in tqdm(article_urls):
            data = parse_article(url, config)
            if data:
                results.append(data)
                time.sleep(delay)

        next_page_selector = config.get("next_page_selector", "")
        if not next_page_selector:
            print("⚠️ No next page selector provided. Stopping pagination.")
            break
        next_button = soup.select_one(next_page_selector)
        if next_button and next_button.has_attr("href"):
            # next_url = config["base_url"] + next_button["href"]
            # next_url = next_button["href"]
            next_url = config.get('start_url','') + f'page/{i+1}/'
        else:
            print("⚠️ No more pages to scrape.")
            break

    return results


def save_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(data)} entries to {filename}")


load_dotenv()

config = {
    "start_url": os.getenv("START_URL"),
    "base_url": os.getenv("BASE_URL"),
    "article_link_selector": os.getenv("ARTICLE_LINK_SELECTOR"),
    "next_page_selector": os.getenv("NEXT_PAGE_SELECTOR"),
    "name_selector": os.getenv("NAME_SELECTOR"),
    "location_selector": os.getenv("LOCATION_SELECTOR"),
    "cuisine_selector": os.getenv("CUISINE_SELECTOR"),
    "article_paragraph_selector": os.getenv("ARTICLE_PARAGRAPH_SELECTOR")
}

RAW_OUTPUT_PATH = os.getenv("RAW_OUTPUT_PATH", "raw/data.json")

if __name__ == "__main__":
    data = scrape(config, max_pages=5)
    save_to_json(data, RAW_OUTPUT_PATH)
    print("Scraping completed.")