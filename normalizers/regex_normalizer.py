import re
from typing import Dict, Any

def normalize_regex(record: Dict[str, Any]):

    info = record.get("infobox_raw", {})

    born = info.get("Born", "")

    date_match = re.search(r'\d{1,2}\s\w+\s\d{4}', born)

    birth_date = date_match.group(0) if date_match else ""

    birth_place = born.replace(birth_date, "").strip()

    occupation = info.get("Occupation", "")

    occupation = re.split(r',|;', occupation)

    occupation = ";".join([o.strip() for o in occupation if o.strip()])

    return {

        "title": record.get("title", ""),
        "name": record.get("title", ""),
        "birth_date": birth_date,
        "birth_place": birth_place,
        "nationality": info.get("Nationality", ""),
        "occupation": occupation,
        "awards": info.get("Awards", ""),
        "website": info.get("Website", "")
    }
