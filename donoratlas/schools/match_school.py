import json
import os
import re
import time
from collections import defaultdict
from typing import Any, List, Optional, Tuple, TypedDict

import pandas as pd
from pydantic import BaseModel
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

    # Mapping from any colloquial name to the school's ID
    name_to_ids = defaultdict(list)

    # Individual mappings for each category, to speed up lookup when type is already known
    law_name_to_ids = defaultdict(list)
    business_name_to_ids = defaultdict(list)
    med_name_to_ids = defaultdict(list)
    eng_name_to_ids = defaultdict(list)
    high_school_ids = defaultdict(list)

    categories = ["law", "business", "medical", "engineer", "high school"]
    category_dicts = [
        law_name_to_ids,
        business_name_to_ids,
        med_name_to_ids,
        eng_name_to_ids,
        high_school_ids,
    ]

    for _, row in df_schools.iterrows():
        wd_id = row["wd_id"]
        # Normalize name and nicknames for consistency
        name = row["name"].encode("utf-8").decode("unicode_escape").strip().lower()

        if pd.notna(row["altLabels"]):  # Check if 'nicknames' is not NaN
            nicknames = [
                nickname.encode("utf-8").decode("unicode_escape").strip().strip("'").strip('"').lower()
                for nickname in json.loads(row["altLabels"])
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


(
    DF_SCHOOLS,
    STRING_ID_MAP,
    LAW_NAME_TO_IDS,
    BUSINESS_NAME_TO_IDS,
    MED_NAME_TO_IDS,
    ENG_NAME_TO_IDS,
    HIGH_SCHOOL_IDS,
) = read_csv(SCHOOLS_CSV_PATH)


def fetch_csv_properties(entity_str):
    """
    Fetch specified properties from a CSV based on keywords in the entity string.

    Parameters:
        entity_str (str): The entity string (e.g., 'Harvard University').
        csv_file_path (str): The path to the CSV file containing relevant institution data.

    Returns:
        pd.DataFrame: A DataFrame containing rows from the CSV that match the criteria.
    """
    print(f"fetching csv entries for {entity_str}")
    str_to_search = STRING_ID_MAP
    # Define the categories or keywords to search for in the entity string
    if " medical" in entity_str.casefold() or "med" in entity_str.casefold():
        str_to_search = MED_NAME_TO_IDS
        print("searching med ids")
    if " law" in entity_str.casefold():
        str_to_search = LAW_NAME_TO_IDS
        print("searching law ids")
    if " engineering" in entity_str.casefold():
        str_to_search = ENG_NAME_TO_IDS
        print("searching eng ids")
    if " business" in entity_str.casefold():
        str_to_search = BUSINESS_NAME_TO_IDS
        print("searching business ids")
    if " high school" in entity_str.casefold():
        str_to_search = HIGH_SCHOOL_IDS
        print("searching high school ids")

    entity_str = entity_str.casefold().strip()
    matches = process.extract(entity_str, str_to_search.keys(), scorer=fuzz.WRatio, limit=50, score_cutoff=70)

    if matches[0][1] == 100:
        return [
            (matched_key, score, STRING_ID_MAP[matched_key])
            for matched_key, score, _ in matches
            if score == 100
        ]
    return [
        (matched_key, score, STRING_ID_MAP[matched_key]) for matched_key, score, _ in matches
    ]


def retrieve_school_object(match_id: str):
    match = DF_SCHOOLS[DF_SCHOOLS["wd_id"] == match_id]

    if not match.empty:
        return match.iloc[0].to_dict()

    print(f"No match found for wd_id: {match_id}")
    return None


def custom_scoring_function(query, candidate, *, score_cutoff=None):
    # print(query, candidate)
    ignore_words = {
        "school",
        "college",
        "university",
        "state",
        "law",
        "business",
        "medical",
        "of",
        "the",
        "program",
        "junior",
        "senior",
        "high",
    }

    def normalize_text(s):
        """Lowercase, strip, and remove parentheses."""
        return re.sub(r"\(.*?\)", "", s).strip().lower()

    query = normalize_text(query.lower().strip())
    candidate = normalize_text(candidate.lower().strip())

    query_words = set(query.split())
    candidate_words = set(candidate.split())

    important_query_words = query_words - ignore_words
    important_candidate_words = candidate_words - ignore_words

    base_score = fuzz.token_sort_ratio(query, candidate)
    # print('base score:', base_score)

    coverage_score = len(query_words & candidate_words) / len(query_words) * 100 if query_words else 0
    # print('coverage score:', coverage_score)
    important_coverage_score = (
        len(important_query_words & important_candidate_words) / len(important_query_words) * 100
        if important_query_words
        else 0
    )
    # print('important coverage score:', important_coverage_score)

    length_difference_penalty = abs(len(query_words) - len(candidate_words)) * 2
    # print('length_difference_penalty:', length_difference_penalty)

    # Bonus for fully containing the query
    containment_bonus = 20 if query in candidate or candidate in query else 0
    # print('containment_bonus:', containment_bonus)

    final_score = (
        (0.3 * base_score + 0.2 * coverage_score + 0.4 * important_coverage_score)
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


def find_best_match(
    query: str, data: List[Tuple[str, float, List[str]]], verbose=False, accept_substring_score=False
):
    if verbose:
        print(f"finding best match for {query} from {data}")
    if len(data) == 1:
        if len(data[0][2]) == 1 and data[0][1] > 95:
            return data[0][2][0], data[0][1]
        else:
            return choose_best_id(*data[0]), None
    max_scores = [datum for datum in data if datum[1] > 95]
    if len(max_scores) == 1:
        return max_scores[0][2][0], max_scores[0][1]
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
    new_scores_cutoff = process.extract(
        query, strings_to_match, scorer=custom_scoring_function, score_cutoff=70
    )

    if new_scores_cutoff:
        threshold = 0
        new_scores_filtered = [
            (matched_key, score, STRING_ID_MAP[matched_key])
            for matched_key, score, _ in new_scores_cutoff
            if score >= threshold
        ]

        print("new scores", new_scores_filtered)
        max_score = new_scores_filtered[0][1]
        print("max scores:", max_score)
        max_scores = [
            (matched_key, score, STRING_ID_MAP[matched_key])
            for matched_key, score, _ in new_scores_filtered
            if score >= max_score
        ]

        if verbose:
            print(max_scores)

        return choose_best_id(*max_scores[0]), max_score
    elif accept_substring_score:
        scores = []
        new_scores = process.extract(query, strings_to_match, scorer=custom_scoring_function)
        for candidate in new_scores:
            print(query, candidate[0])
            if query.casefold() in candidate[0].casefold():
                scores.append(candidate)
            elif candidate[0].casefold() in query.casefold():
                scores.append(candidate)
            else:
                print("did not add")
        scores = [(matched_key, score, STRING_ID_MAP[matched_key]) for matched_key, score, _ in scores]
        print("substring scores", scores)
        max_score = max(scores, key=lambda x: x[1])
        print("max score", max_score)
        return max_score[2][0], max_score[1]

    else:
        return None, None


def preprocess_query(query: str) -> str:
    """
    Remove extra whitespace, remove parentheses, and remove newlines.

    Parameters
    ----------
        query : str
            The query to preprocess.

    Returns
    -------
        str
            The preprocessed query.
    """
    query = re.sub(r"\s+", " ", re.sub(r"\(\d+\)|\n", " ", query)).strip().strip('"')
    query = query.encode("utf-8").decode("unicode_escape")
    return query


def split_query(query: str):
    """
    Split the query into a list of strings.

    Parameters
    ----------
        query : str
            The query to split.

    Returns
    -------
        list[str]
            The split query.
    """
    print("splitting query")
    try:
        # Use re.split to split on the specified words
        new_query = re.split(r"\b(from|at)\b", query)
        print('splitting on "from", "at"', new_query)
        did_split_on_from_at = new_query[0] != query

        if new_query[0] == query:
            print("splitting query on (")
            new_query = query.split("(")
            new_query = [query.strip().strip("(").strip(")") for query in new_query]

        if new_query[0] == query:
            print("splitting query on ,")
            new_query = query.split(",")

        if new_query[0] == query:
            print("splitting query on -")
            new_query = query.split("-")

        if new_query[0] == query:
            print("splitting query on :")
            new_query = query.split(":")

        if new_query[0] == query:
            print("returning original query")
            return None

        words_to_check = ["college", "university", "school", "institute", "academy"]
        to_return = []
        for query in new_query:
            if (
                any(word in query.casefold() for word in words_to_check)
                or bool(re.search(r"U [A-Z]", query))
                or bool(re.search(r"U[A-Z]", query))
            ):
                # Strip 'the'
                if query.strip().lower().startswith("the "):
                    query = query.strip()[3:].strip()
                to_return.append(query.strip())
        if to_return == [] and did_split_on_from_at:
            first_split_index = [i for i, w in enumerate(new_query) if w in ["from", "at"]][0]
            if first_split_index is not None:
                return new_query[first_split_index + 1 :]
            return new_query[-1]
        print("returning new_query", to_return)
        return to_return
        # print('new query:', new_query)
        # return new_query
    except:
        print("Error splitting query. Returning original")
        return None


# Main function to resolve entity
def match_string_to_school_id(query, verbose=False):
    """
    This is the function that returns the id of the school in question.
    If multiple schools exactly match, returns an array of ids.
    If no schools match well enough, returns None.
    """
    query = preprocess_query(query)
    print(query)
    candidates = fetch_csv_properties(query.casefold())
    print("candidates", len(candidates), candidates)
    if candidates:
        best_match = find_best_match(query, candidates, verbose=True)
        print("best_match", best_match)
        if best_match[0]:
            return best_match[0]
        else:
            try:
                queries = split_query(query)
                if not queries:
                    return None
                best_matches = []
                for query in queries:
                    candidates = fetch_csv_properties(query.casefold())
                    if candidates:
                        best_match, max_score = find_best_match(
                            query, candidates, verbose=True, accept_substring_score=True
                        )
                        if best_match:
                            best_matches.append((best_match, max_score))
                    else:
                        print("no candidates found (line 408)")
                print("best matches", best_matches)
                if best_matches == []:
                    return None
                elif len(best_matches) == 1:
                    return best_matches[0][0]
                else:
                    max_tuple = max(best_matches, key=lambda x: x[1])
                    return max_tuple[0]
            except:
                return None
        # if best_match:
        #     match = retrieve_school_object(best_match)
        #     return match
    try:
        queries = split_query(query)
        if not queries:
            return None
        candidates = [fetch_csv_properties(query.casefold()) for query in queries][0]
        if candidates:
            best_match, max_score = find_best_match(
                queries[-1], candidates, verbose=True, accept_substring_score=True
            )
            if best_match:
                return best_match
            else:
                return None
        else:
            print("no candidates found (line 298)")
    except:
        return (None,)
    return None


def test_random_schools():
    class Education(BaseModel):
        """A single educational institution relationship"""

        school_ids: list[str]
        raw_name: Optional[str] = None
        degree: Optional[str] = None
        major: Optional[str] = None
        grad_year: Optional[int] = None
        candidates: Optional[list[Any]] = None

    educations = []
    with open("raw_names_3.txt", "r") as file:
        education = [
            "MA in folklore from UNC-Chapel Hill",
            "Associate in Arts, SUNY Broome",
            "Executive Education - Wharton School at the University of Pennsylvania",
        ]

        education.extend(list(set([line for line in file])))
        print(len(education))

    with open("matches_3.txt", "w") as file:
        for edu in education:
            start_time = time.time()
            school_id = match_string_to_school_id(edu, verbose=True)
            end_time = time.time()
            edu_obj = Education(
                school_ids=(
                    [] if school_id is None else (school_id if isinstance(school_id, list) else [school_id])
                ),
                raw_name=edu,
            )
            file.write(
                json.dumps(
                    {
                        "ids": edu_obj.school_ids,
                        "raw_name": edu,
                        "processed_name": (
                            retrieve_school_object(edu_obj.school_ids[0]).get("name")
                            if len(edu_obj.school_ids) > 0
                            else None
                        ),
                    }
                )
                + "\n"
            )
            educations.append(edu_obj)
            print("school_id", school_id)
            print(f"{end_time-start_time}s to execute\n")

        # for query in education:
        #     print(query.strip())
        #     entity = match_string_to_school_id(query)
        #     print(entity)


if __name__ == "__main__":
    # print(test_random_schools())
    raw_input = input(">>> ")

    while raw_input.casefold() != "exit":
        print(match_string_to_school_id(raw_input))
        raw_input = input(">>> ")
