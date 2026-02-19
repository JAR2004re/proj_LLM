import json
import csv
import os

from normalizers.css_normalizer import normalize_css
from normalizers.regex_normalizer import normalize_regex
from normalizers.llm_normalizer import normalize_llm

INPUT_FILE = "data/raw_records.json"

CSS_OUT = "outputs/css_output.csv"
REGEX_OUT = "outputs/regex_output.csv"
LLM_OUT = "outputs/llm_output.csv"

COLUMNS = [
"title",
"name",
"birth_date",
"birth_place",
"nationality",
"occupation",
"awards",
"website"
]

def write_csv(path, rows):

    with open(path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=COLUMNS)

        writer.writeheader()

        for r in rows:
            writer.writerow(r)


def main():

    os.makedirs("outputs", exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:

        records = json.load(f)

    css_rows = []
    regex_rows = []
    llm_rows = []

    for r in records:

        css_rows.append(normalize_css(r))

        regex_rows.append(normalize_regex(r))

        llm_rows.append(normalize_llm(r))

    write_csv(CSS_OUT, css_rows)

    write_csv(REGEX_OUT, regex_rows)

    write_csv(LLM_OUT, llm_rows)

    print("Done normalization")


if __name__ == "__main__":
    main()
