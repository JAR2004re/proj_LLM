import json
import os

from wiki_people_scraper import (
    get_page_html,
    page_url,
    get_last_updated,
    extract_infobox,
    extract_lead_and_sections,
    count_references
)

from bs4 import BeautifulSoup

OUTPUT = "data/raw_records.json"

def extract(title):

    page_id, html = get_page_html(title)

    soup = BeautifulSoup(html, "html.parser")

    return {

        "title": title,
        "page_id": page_id,
        "url": page_url(title),
        "last_updated": get_last_updated(title),
        "references_count": count_references(soup),
        "infobox_raw": extract_infobox(soup),
        "sections": extract_lead_and_sections(soup)
    }

def main():

    os.makedirs("data", exist_ok=True)

    titles = [
        "Elon Musk",
        "Ada Lovelace",
        "Narendra Modi"
    ]

    data = []

    for t in titles:

        print("Extracting:", t)

        data.append(extract(t))

    with open(OUTPUT, "w", encoding="utf-8") as f:

        json.dump(data, f, indent=2)

    print("Saved raw_records.json")


if __name__ == "__main__":
    main()
