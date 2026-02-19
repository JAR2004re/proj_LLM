from wiki_people_scraper import llm_normalize_record

def normalize_llm(record):

    result = llm_normalize_record(
        record,
        prompt_path="wiki_prompt.txt"
    )

    norm = result.get("infobox_normalized", {})

    return {

        "title": result.get("title", ""),

        "name": norm.get("name", ""),

        "birth_date": norm.get("birth_date", ""),

        "birth_place": norm.get("birth_place", ""),

        "nationality": norm.get("nationality", ""),

        "occupation": ";".join(norm.get("occupation", [])),

        "awards": ";".join(norm.get("awards", [])),

        "website": norm.get("website", "")
    }
