import os
import re
import time
from collections import defaultdict
from typing import List, Tuple, TypedDict

import pandas as pd
from rapidfuzz import fuzz, process


class ProcessedWikidata(TypedDict):
    name: str
    nicknames: List[str]
    instance: str
    wd_id: str
    instances: List[str]


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHOOLS_CSV_PATH = os.path.join(BASE_DIR, "static", "schools.csv")


def read_csv(csv_file_path: str):
    # read csv into dataframe and dict
    df_schools = pd.read_csv(csv_file_path, dtype=str)
    name_to_ids = defaultdict(list)
    law_name_to_ids = defaultdict(list)
    business_name_to_ids = defaultdict(list)
    med_name_to_ids = defaultdict(list)
    eng_name_to_ids = defaultdict(list)

    categories = ["law", "business", "medical", "engineer"]
    category_dicts = [law_name_to_ids, business_name_to_ids, med_name_to_ids, eng_name_to_ids]

    for _, row in df_schools.iterrows():
        wd_id = row["wd_id"]
        # Normalize name and nicknames for consistency
        name = row["name"].strip().lower()

        if pd.notna(row["nickname"]):  # Check if 'nicknames' is not NaN
            nicknames = [
                nickname.strip().strip("'").lower()
                for nickname in str(row["nickname"])[1:-1].split("', '")
                if nickname.strip()
            ]
        else:
            nicknames = []

        # Add wd_id to the main name and nicknames
        name_to_ids[name].append(wd_id)
        for nickname in nicknames:
            name_to_ids[nickname].append(wd_id)

        for i, category in enumerate(categories):
            if category in row["instance of"]:
                category_dicts[i][name].append(wd_id)
                for nickname in nicknames:
                    category_dicts[i][nickname].append(wd_id)

    # Ensure unique wd_ids for each name/nickname
    string_id_map = {key: list(set(value)) for key, value in name_to_ids.items()}
    for i, dict in enumerate(category_dicts):
        globals()[f"string_id_map_{categories[i]}"] = {key: list(set(value)) for key, value in dict.items()}
    return df_schools, string_id_map, *category_dicts


DF_SCHOOLS, STRING_ID_MAP, LAW_NAME_TO_IDS, BUSINESS_NAME_TO_IDS, MED_NAME_TO_IDS, ENG_NAME_TO_IDS = read_csv(
    SCHOOLS_CSV_PATH
)


def fetch_csv_properties(entity_str):
    """
    Fetch specified properties from a CSV based on keywords in the entity string.

    Parameters:
        entity_str (str): The entity string (e.g., 'Harvard University').
        csv_file_path (str): The path to the CSV file containing relevant institution data.

    Returns:
        pd.DataFrame: A DataFrame containing rows from the CSV that match the criteria.
    """
    str_to_search = STRING_ID_MAP
    # Define the categories or keywords to search for in the entity string
    if " medical" in entity_str.casefold() or "med" in entity_str.casefold():
        str_to_search = MED_NAME_TO_IDS
    if " law" in entity_str.casefold():
        str_to_search = LAW_NAME_TO_IDS
    if " engineering" in entity_str.casefold():
        str_to_search = ENG_NAME_TO_IDS
    if " business" in entity_str.casefold():
        str_to_search = BUSINESS_NAME_TO_IDS

    matches = process.extract(entity_str, str_to_search.keys(), scorer=fuzz.WRatio, limit=50)

    if matches[0][1] == 100:
        return [
            (matched_key, score, STRING_ID_MAP[matched_key])
            for matched_key, score, _ in matches
            if score == 100
        ]
    return [
        (matched_key, score, STRING_ID_MAP[matched_key]) for matched_key, score, _ in matches if score >= 86
    ]


def retrieve_school_object(match_id: str):
    match = DF_SCHOOLS[DF_SCHOOLS["wd_id"] == match_id]

    if not match.empty:
        return match.iloc[0].to_dict()

    print(f"No match found for wd_id: {match_id}")
    return None


def custom_scoring_function(query, candidate, *, score_cutoff=None):
    ignore_words = {"school", "college", "law", "business", "medical", "of", "the"}

    def remove_parentheses(s):
        return re.sub(r"\(.*?\)", "", s).strip()

    query = remove_parentheses(query.lower().strip())
    candidate = remove_parentheses(candidate.lower().strip())

    query_words = set(query.split())
    candidate_words = set(candidate.split())

    important_query_words = query_words - ignore_words
    important_candidate_words = candidate_words - ignore_words

    base_score = fuzz.token_sort_ratio(query, candidate)

    coverage_score = len(query_words & candidate_words) / len(query_words) * 100 if query_words else 0
    important_coverage_score = (
        len(important_query_words & important_candidate_words) / len(important_query_words) * 100
        if important_query_words
        else 0
    )

    length_difference_penalty = abs(len(query_words) - len(candidate_words)) * 2  # Reduced penalty

    # Bonus for fully containing the query
    containment_bonus = 20 if query in candidate else 0

    final_score = (
        (0.3 * base_score + 0.3 * coverage_score + 0.4 * important_coverage_score)
        - length_difference_penalty
        + containment_bonus
    )

    if score_cutoff is not None and final_score < score_cutoff:
        return 0

    return max(0, final_score)


def choose_best_id(match, score, ids):
    if len(ids) == 1:
        return ids[0]

    # sort on name > nickname, etc. if needed using entire object
    schools = []
    name_matches = []
    nickname_matches = []
    for id in ids:
        school = retrieve_school_object(id)
        if school:
            school["score"] = score
            if match == school["name"].casefold():
                name_matches.append(school)
            else:
                nickname_matches.append(school)
            school["matched_on_name"] = match == school["name"]
            schools.append(school)
    if len(name_matches) == 1:
        return name_matches[0]["wd_id"]
    else:
        return ids


def find_best_match(query: str, data: List[Tuple[str, float, List[str]]], verbose=False):
    if len(data) == 1:
        if len(data[0][2]) == 1 and data[0][1] > 95:
            return data[0][2][0]
        else:
            return choose_best_id(*data[0])

    # If all strings refer to the same place, return
    all_match = True
    while all_match:
        for match in data:
            for id in match[2]:
                if id != data[0][2][0]:
                    all_match = False
    if all_match:
        print("all ids the same. returning.")
        return choose_best_id(*data[0])

    strings_to_match = [datum[0] for datum in data]
    new_scores = process.extract(query, strings_to_match, scorer=custom_scoring_function, score_cutoff=70)

    if new_scores:
        threshold = 0
        new_scores = [
            (matched_key, score, STRING_ID_MAP[matched_key])
            for matched_key, score, _ in new_scores
            if score >= threshold
        ]

        if verbose:
            print("new scores")
        max_score = new_scores[0][1]
        if verbose:
            print("max scores:", max_score)
        max_scores = [
            (matched_key, score, STRING_ID_MAP[matched_key])
            for matched_key, score, _ in new_scores
            if score >= max_score
        ]

        if verbose:
            print(max_scores)

        return choose_best_id(*max_scores[0])
    else:
        return None


# Main function to resolve entity
def match_string_to_school_id(query, verbose=False):
    """
    This is the function that returns the id of the school in question.
    If multiple schools exactly match, returns an array of ids.
    If no schools match well enough, returns None.
    """
    candidates = fetch_csv_properties(query.casefold())
    if verbose:
        print("candidates", len(candidates), candidates)
    if candidates:
        best_match = find_best_match(query, candidates, verbose)
        return best_match
        # if best_match:
        #     match = retrieve_school_object(best_match)
        #     return match
    return None


def test_random_schools():
    import random

    cleaner_schools = [
        # Universities
        "Harvard University",
        "Stanford University",
        "University of Oxford",
        "Massachusetts Institute of Technology (MIT)",
        "University of Cambridge",
        "University of California, Berkeley",
        "Yale University",
        "University of Chicago",
        "Princeton University",
        "Columbia University",
        # Colleges
        "Macalester College",
        "Amherst College",
        "Williams College",
        "Swarthmore College",
        "Pomona College",
        "Grinnell College",
        "Wellesley College",
        "Middlebury College",
        "Oberlin College",
        "Bowdoin College",
        # Business Schools
        "Harvard Business School",
        "Stanford Graduate School of Business",
        "Wharton School (University of Pennsylvania)",
        "MIT Sloan School of Management",
        "Kellogg School of Management (Northwestern University)",
        "Columbia Business School",
        "Booth School of Business (University of Chicago)",
        "Tuck School of Business (Dartmouth College)",
        "INSEAD Business School",
        "London Business School",
        # Law Schools
        "Harvard Law School",
        "Yale Law School",
        "Stanford Law School",
        "Columbia Law School",
        "University of Oxford Law School",
        "New York University School of Law",
        "University of Chicago Law School",
        "University of California, Berkeley Law",
        "University of Cambridge Faculty of Law",
        "Georgetown University Law Center",
        # High Schools
        "Phillips Academy Andover",
        "Harvard-Westlake School",
        "The Lawrenceville School",
        "The Hotchkiss School",
        "The Brearley School",
        "The Spence School",
        "Trinity School (New York City)",
        "The Dalton School",
        "St. Paul's School",
        "The Chapin School",
        # Colleges within Universities
        "College of William & Mary (part of William & Mary)",
        "College of Arts & Sciences (University of Washington)",
        "College of Engineering (University of Michigan)",
        "College of Fine Arts (University of Texas)",
        "College of Charleston (University of Charleston)",
        "College of Education (University of Pennsylvania)",
        "College of Liberal Arts (Texas A&M University)",
        "College of Public Health (University of Iowa)",
        "College of Medicine (University of Florida)",
        "College of Business (University of Nebraska–Lincoln)",
    ]

    random.shuffle(cleaner_schools)
    shortened_cleaner_schools = [
        "College of Liberal Arts (Texas A&M University)",
        "Trinity School (New York City)",
        "The Spence School",
        "Booth School of Business (University of Chicago)",
        "Harvard-Westlake School",
        "College of Public Health (University of Iowa)",
    ]
    shortened_cleaner_schools.extend(cleaner_schools[: len(cleaner_schools) // 2])

    for query in shortened_cleaner_schools:
        print(query)
        start_time = time.time()
        entity = match_string_to_school_id(query)
        end_time = time.time()
        print(entity)
        print(f"{end_time-start_time}s to execute\n")


if __name__ == "__main__":
    raw_input = input(">>> ")

    while raw_input.casefold() != "exit":
        print(match_string_to_school_id(raw_input))
        raw_input = input(">>> ")
