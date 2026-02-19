import re
from typing import Dict, Any, List

def split_list(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"\s*[,;|]\s*", s.strip())
    return [p for p in parts if p]

def normalize_css(record: Dict[str, Any]) -> Dict[str, Any]:

    info = record.get("infobox_raw", {})

    born = info.get("Born", "")
    nationality = info.get("Nationality", "")
    occupation = info.get("Occupation", "")
    awards = info.get("Awards", "")
    website = info.get("Website", "")

    birth_date = ""
    birth_place = ""

    if born:
        parts = born.split(" ", 3)

        if len(parts) >= 3:
            birth_date = " ".join(parts[:3])
            birth_place = born.replace(birth_date, "").strip()

    return {
        "title": record.get("title", ""),
        "name": record.get("title", ""),
        "birth_date": birth_date,
        "birth_place": birth_place,
        "nationality": nationality,
        "occupation": ";".join(split_list(occupation)),
        "awards": ";".join(split_list(awards)),
        "website": website,
    }
