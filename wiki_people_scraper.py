import os
import re
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Optional LLM mode (same pattern as your rto_scrape.py)
LLM_AVAILABLE = True
try:
    import asyncio
    from pydantic import BaseModel, Field
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig
    from crawl4ai.extraction_strategy import LLMExtractionStrategy
except Exception:
    LLM_AVAILABLE = False


WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_PAGE = "https://en.wikipedia.org/wiki/"


def mw_get(params: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    params = dict(params)
    params["format"] = "json"
    r = requests.get(
        WIKI_API,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "wiki-kg-scraper/1.0"}
    )
    r.raise_for_status()
    return r.json()


def normalize_title(title: str) -> str:
    return title.strip().replace(" ", "_")


def page_url(title: str) -> str:
    return WIKI_PAGE + normalize_title(title)


def get_last_updated(title: str) -> str:
    data = mw_get({
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvprop": "timestamp",
        "rvlimit": 1
    })
    pages = data.get("query", {}).get("pages", {})
    for _, p in pages.items():
        revs = p.get("revisions", [])
        if revs:
            return revs[0].get("timestamp", "") or ""
    return ""


def get_page_html(title: str) -> Tuple[Optional[int], str]:
    data = mw_get({
        "action": "parse",
        "page": title,
        "prop": "text",
        "redirects": 1
    })
    page_id = data.get("parse", {}).get("pageid")
    html = data.get("parse", {}).get("text", {}).get("*", "") or ""
    return page_id, html


def clean_text(s: str) -> str:
    s = re.sub(r"\[\d+\]", "", s)      # remove [1], [2]
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_infobox(soup: BeautifulSoup) -> Dict[str, str]:
    infobox = soup.select_one("table.infobox")
    if not infobox:
        return {}

    out: Dict[str, str] = {}
    for row in infobox.select("tr"):
        th = row.find("th")
        td = row.find("td")
        if not th or not td:
            continue

        key = clean_text(th.get_text(" ", strip=True))
        val = clean_text(td.get_text(" ", strip=True))

        if key and val:
            if key in out and val not in out[key]:
                out[key] = out[key] + " | " + val
            else:
                out[key] = val

    return out


def count_references(soup: BeautifulSoup) -> int:
    refs = soup.select("ol.references li")
    if refs:
        return len(refs)

    refs2 = soup.select("sup.reference")
    return len(refs2)


def extract_lead_and_sections(soup: BeautifulSoup) -> Dict[str, str]:
    """
    lead: paragraphs before first H2
    sections: Education, Career, Awards, Personal life (best-effort)
    """
    content = soup.select_one("div.mw-parser-output")
    if not content:
        return {"lead": "", "education": "", "career": "", "awards": "", "personal_life": ""}

    # lead
    lead_parts: List[str] = []
    for el in content.find_all(["p", "h2"], recursive=False):
        if el.name == "h2":
            break
        if el.name == "p":
            txt = clean_text(el.get_text(" ", strip=True))
            if txt:
                lead_parts.append(txt)
    lead = "\n".join(lead_parts).strip()

    section_text: Dict[str, str] = {"education": "", "career": "", "awards": "", "personal_life": ""}

    target_map = {
        "education": {"education", "early life", "early_life", "background"},
        "career": {"career", "work", "professional career", "playing career", "political career"},
        "awards": {"awards", "honours", "honors", "recognition", "achievements"},
        "personal_life": {"personal life", "personal_life", "family", "private life"}
    }

    current_key: Optional[str] = None
    buf: List[str] = []

    def flush():
        nonlocal current_key, buf
        if current_key and buf:
            section_text[current_key] = "\n".join(buf).strip()
        current_key = None
        buf = []

    for el in content.find_all(["h2", "p", "h3", "ul"], recursive=True):
        if el.name == "h2":
            flush()
            headline = clean_text(el.get_text(" ", strip=True)).lower().replace("[edit]", "").strip()
            for key, names in target_map.items():
                if headline in names:
                    current_key = key
                    break
            continue

        if current_key:
            if el.name == "p":
                txt = clean_text(el.get_text(" ", strip=True))
                if txt:
                    buf.append(txt)
            elif el.name == "ul":
                items = [clean_text(li.get_text(" ", strip=True)) for li in el.find_all("li")]
                items = [i for i in items if i]
                if items:
                    buf.append(" • " + " • ".join(items))

    flush()

    return {
        "lead": lead,
        "education": section_text["education"],
        "career": section_text["career"],
        "awards": section_text["awards"],
        "personal_life": section_text["personal_life"],
    }


def get_category_members(category: str, limit: int = 50) -> List[str]:
    if not category.lower().startswith("category:"):
        category = "Category:" + category

    titles: List[str] = []
    cmcontinue = None

    while len(titles) < limit:
        params: Dict[str, Any] = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": min(50, limit - len(titles)),
            "cmtype": "page"
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        data = mw_get(params)
        members = data.get("query", {}).get("categorymembers", [])
        for m in members:
            t = m.get("title")
            if t:
                titles.append(t)

        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break

        time.sleep(0.2)

    return titles


# ---------------- LLM schema for normalized output (optional) ----------------
if LLM_AVAILABLE:
    class PersonNormalized(BaseModel):
        name: str = Field(default="")
        birth_date: str = Field(default="")
        death_date: str = Field(default="")
        birth_place: str = Field(default="")
        nationality: str = Field(default="")
        occupation: List[str] = Field(default_factory=list)
        known_for: List[str] = Field(default_factory=list)
        education: List[str] = Field(default_factory=list)
        alma_mater: List[str] = Field(default_factory=list)
        awards: List[str] = Field(default_factory=list)
        spouse: List[str] = Field(default_factory=list)
        children: List[str] = Field(default_factory=list)
        organizations: List[str] = Field(default_factory=list)
        website: str = Field(default="")

    class PersonRecord(BaseModel):
        title: str = Field(default="")
        page_id: Optional[int] = Field(default=None)
        url: str = Field(default="")
        last_updated: str = Field(default="")
        references_count: int = Field(default=0)
        infobox_raw: Dict[str, str] = Field(default_factory=dict)
        infobox_normalized: PersonNormalized = Field(default_factory=PersonNormalized)
        sections: Dict[str, str] = Field(default_factory=dict)


def llm_normalize_record(record: Dict[str, Any], prompt_path: str) -> Dict[str, Any]:
    """
    Uses Crawl4AI LLMExtractionStrategy to map messy extracted text into strict schema.
    """
    if not LLM_AVAILABLE:
        raise RuntimeError("LLM mode not available (crawl4ai/pydantic not installed).")

    with open(prompt_path, "r", encoding="utf-8") as f:
        instruction = f.read()

    md = f"""
# TITLE
{record.get("title", "")}

# INFOBOX_RAW (JSON)
{json.dumps(record.get("infobox_raw", {}), ensure_ascii=False)}

# LEAD
{record.get("sections", {}).get("lead", "")}

# EDUCATION
{record.get("sections", {}).get("education", "")}

# CAREER
{record.get("sections", {}).get("career", "")}

# AWARDS
{record.get("sections", {}).get("awards", "")}

# PERSONAL LIFE
{record.get("sections", {}).get("personal_life", "")}
""".strip()

    llm_cfg = LLMConfig(
        provider="openai/gpt-4o-mini",
        api_token=os.getenv("OPENAI_API_KEY"),
    )

    strategy = LLMExtractionStrategy(
        llm_config=llm_cfg,
        schema=PersonRecord.model_json_schema(),
        extraction_type="schema",
        instruction=instruction,
        input_format="markdown",
        apply_chunking=False,
        chunk_token_threshold=2000,
    )

    crawl_cfg = CrawlerRunConfig(
        extraction_strategy=strategy,
        cache_mode=CacheMode.ENABLED,
        exclude_external_links=True,
        page_timeout=120000,
    )

    async def _run() -> Dict[str, Any]:
        browser_cfg = BrowserConfig(headless=True, verbose=False, text_mode=True)
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            dummy_html = f"<html><body><pre>{md}</pre></body></html>"
            url = "raw://" + dummy_html

            for attempt in range(2):  # one retry
                try:
                    res = await crawler.arun(url, config=crawl_cfg)

                    if not res.success:
                        raise RuntimeError(res.error_message)

                    return json.loads(res.extracted_content)[0]

                except Exception as e:
                    if attempt == 1:
                        raise
                    await asyncio.sleep(1.0)


    return asyncio.run(_run())


def record_to_csv_row(record: Dict[str, Any]) -> Dict[str, Any]:
    norm = record.get("infobox_normalized", {}) or {}
    sections = record.get("sections", {}) or {}

    def j(x):
        if isinstance(x, list):
            return "; ".join([str(i) for i in x if str(i).strip()])
        return x

    return {
        "title": record.get("title", ""),
        "page_id": record.get("page_id", ""),
        "url": record.get("url", ""),
        "last_updated": record.get("last_updated", ""),
        "references_count": record.get("references_count", 0),

        "name": norm.get("name", ""),
        "birth_date": norm.get("birth_date", ""),
        "death_date": norm.get("death_date", ""),
        "birth_place": norm.get("birth_place", ""),
        "nationality": norm.get("nationality", ""),
        "occupation": j(norm.get("occupation", [])),
        "known_for": j(norm.get("known_for", [])),
        "education_list": j(norm.get("education", [])),
        "alma_mater": j(norm.get("alma_mater", [])),
        "awards": j(norm.get("awards", [])),
        "spouse": j(norm.get("spouse", [])),
        "children": j(norm.get("children", [])),
        "organizations": j(norm.get("organizations", [])),
        "website": norm.get("website", ""),

        "lead": sections.get("lead", ""),
        "education_section": sections.get("education", ""),
        "career_section": sections.get("career", ""),
        "awards_section": sections.get("awards", ""),
        "personal_life_section": sections.get("personal_life", ""),

        "infobox_raw_json": json.dumps(record.get("infobox_raw", {}), ensure_ascii=False)
    }


def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--titles", nargs="*", default=[], help="Wikipedia page titles (space separated).")
    ap.add_argument("--category", default="", help="Wikipedia category name or full Category:... title.")
    ap.add_argument("--limit", type=int, default=25, help="Max pages from category.")
    ap.add_argument("--use-llm", action="store_true", help="Use LLM to normalize to strict schema.")
    ap.add_argument("--prompt", default="wiki_prompt.txt", help="Prompt file for LLM mode.")
    ap.add_argument("--out-json", default="wiki_people_output.json")
    ap.add_argument("--out-csv", default="wiki_people_output.csv")
    args = ap.parse_args()

    titles = list(args.titles)

    if args.category:
        titles.extend(get_category_members(args.category, limit=args.limit))

    seen = set()
    titles = [t for t in titles if not (t in seen or seen.add(t))]

    results: List[Dict[str, Any]] = []

    for idx, title in enumerate(titles, start=1):
        print(f"[{idx}/{len(titles)}] Scraping: {title}")

        # ✅ 1. Protect page HTML fetch
        try:
            page_id, html = get_page_html(title)
        except Exception as e:
            print(f"[WARN] Failed to fetch page HTML for {title}: {e}")
            continue

        if not html:
            print(f"[WARN] Empty HTML for {title}")
            continue

        soup = BeautifulSoup(html, "html.parser")

        infobox_raw = extract_infobox(soup)
        sections = extract_lead_and_sections(soup)

        # ✅ 2. Protect last_updated call
        try:
            last_updated = get_last_updated(title)
        except Exception as e:
            print(f"[WARN] Failed to get last_updated for {title}: {e}")
            last_updated = ""

        # ✅ 3. Build record safely
        record = {
            "title": title,
            "page_id": page_id,
            "url": page_url(title),
            "last_updated": last_updated,
            "references_count": count_references(soup),
            "infobox_raw": infobox_raw,
            "infobox_normalized": {
                "name": "",
                "birth_date": "",
                "death_date": "",
                "birth_place": "",
                "nationality": "",
                "occupation": [],
                "known_for": [],
                "education": [],
                "alma_mater": [],
                "awards": [],
                "spouse": [],
                "children": [],
                "organizations": [],
                "website": ""
            },
            "sections": sections
        }

        # ✅ 4. Protect LLM call (very important)
        if args.use_llm:
            try:
                record = llm_normalize_record(record, prompt_path=args.prompt)
            except Exception as e:
                print(f"[WARN] LLM failed for {title}: {e}")

        results.append(record)
        time.sleep(0.2)



    # write JSON
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON saved: {args.out_json}")

    # write CSV
    import csv
    rows = [record_to_csv_row(r) for r in results]
    if rows:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"✅ CSV saved: {args.out_csv}")


if __name__ == "__main__":
    main()
