import json
import os
import re
from typing import Optional

from nameparser import HumanName
from nicknames import NickNamer
from pydantic import BaseModel
from rapidfuzz import fuzz

nick_namer = NickNamer()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATE_NAME_TO_ABBREV = json.load(open(os.path.join(BASE_DIR, "static", "states.json"), encoding="utf-8"))


def get_nicknames(name: str) -> set[str]:
    """
    Get the nicknames for a given name.

    Parameters
    ----------
        name (str): The name for which to get the nicknames.

    Returns
    -------
        set[str]: A set of nicknames for the given name.
    """
    return nick_namer.nicknames_of(name)


def get_formal_names(name: str) -> set[str]:
    """
    Get the formal names for a given name.

    Parameters
    ----------
        name (str): The name for which to get the formal names.

    Returns
    -------
        set[str]: A set of formal names for the given name.
    """
    return nick_namer.canonicals_of(name)


def get_all_names(name: str, include_self: bool = True) -> set[str]:
    """
    Get all the names for a given name.

    Parameters
    ----------
        name (str): The name for which to get the names.
        include_self (bool): Whether to include the name itself in the set of names.

    Returns
    -------
        set[str]: A set of names for the given name.
    """
    return get_nicknames(name) | get_formal_names(name) | {name.casefold()}


def alpha_only(s):
    return "".join(filter(str.isalpha, s))


class PersonName(BaseModel):
    """A person's name."""

    first: Optional[str] = None
    middle: Optional[str] = None  # Either a middle name or initial
    last: str
    nicknames: Optional[list[str]] = None
    suffix: Optional[str] = None
    title: Optional[str] = None

    strict: bool = False

    def standardize(self):
        parsed_name = HumanName(str(self))
        parsed_name.capitalize(force=True)
        self.first = None if parsed_name.first == "" else parsed_name.first
        self.middle = None if parsed_name.middle == "" else parsed_name.middle
        self.last = None if parsed_name.last == "" else parsed_name.last
        self.suffix = None if parsed_name.suffix == "" else parsed_name.suffix
        self.title = None if parsed_name.title == "" else parsed_name.title
        self.nicknames = None if self.first is None else list(get_all_names(self.first, include_self=False))

    def set_strict(self):
        self.strict = True

    def __str__(self) -> str:
        return (
            " ".join(
                [
                    "" if i is None else i
                    for i in [self.title, self.first, self.middle, self.last, self.suffix]
                ]
            )
            .replace(r"\s+", " ")
            .strip()
        )

    def __eq__(self, other) -> bool:
        """Check if two names could refer to the same person"""
        if not isinstance(other, PersonName):
            return False

        self_first = None if self.first is None else alpha_only(self.first.casefold())
        other_first = None if other.first is None else alpha_only(other.first.casefold())

        self_middle = None if self.middle is None else alpha_only(self.middle.casefold())
        other_middle = None if other.middle is None else alpha_only(other.middle.casefold())

        self_last = None if self.last is None else alpha_only(self.last.casefold())
        other_last = None if other.last is None else alpha_only(other.last.casefold())

        if self.strict and any(item is None for item in [self_first, self_last, other_first, other_last]):
            return False

        # If self has one parsed field and the other has both, check if the self matches either
        if (self_first is None or self_last is None) and (other_first is not None and other_last is not None):
            single_compare = self_first if self_first is not None else self_last
            return (
                single_compare == other_first
                or single_compare == other_last
                or single_compare
                in [i.casefold() for i in (set() if other.nicknames is None else set(other.nicknames))]
            )

        # And vice versa
        if (other_first is None or other_last is None) and (self_first is not None and self_last is not None):
            single_compare = other_first if other_first is not None else other_last
            return (
                single_compare == self_first
                or single_compare == self_last
                or single_compare
                in [i.casefold() for i in (set() if self.nicknames is None else set(self.nicknames))]
            )

        firsts_same = (
            (self_first is None or other_first is None)
            or (min(len(self_first), len(other_first)) == 1 and self_first[0] == other_first[0])
            or self_first == other_first
            or self_first
            in [i.casefold() for i in (set() if other.nicknames is None else set(other.nicknames))]
            or other_first
            in [i.casefold() for i in (set() if self.nicknames is None else set(self.nicknames))]
        )

        middles_same = (
            self_middle is None
            or other_middle is None
            or (min(len(self_middle), len(other_middle)) == 1 and self_middle[0] == other_middle[0])
            or self_middle == other_middle
        )

        last_same = self_last is None or other_last is None or self_last.casefold() == other_last.casefold()

        return firsts_same and middles_same and last_same


def name_similarity(name1: PersonName, name2: PersonName) -> float:
    """
    Calculate the similarity between two names.

    Parameters
    ----------
        name1 (PersonName): The first name.
        name2 (PersonName): The second name.

    Returns
    -------
        float: The similarity between the two names.

    Notes
    -----
    We use the following thresholds:
      - 1.0 means exact match on all fields which could have, at least first and last (nickname match is allowed)
      - 0.5 or above can only be achieved with either one perfect field match, or two partial field matches
      - 0 means no match on any field
    """
    name1_first = None if name1.first is None else alpha_only(name1.first.casefold())
    name2_first = None if name2.first is None else alpha_only(name2.first.casefold())

    name1_middle = None if name1.middle is None else alpha_only(name1.middle.casefold())
    name2_middle = None if name2.middle is None else alpha_only(name2.middle.casefold())

    name1_last = None if name1.last is None else alpha_only(name1.last.casefold())
    name2_last = None if name2.last is None else alpha_only(name2.last.casefold())

    score = 0

    if name1_first is not None and name2_first is not None:
        if (
            name1_first == name2_first
            or name1_first
            in [i.casefold() for i in (set() if name2.nicknames is None else set(name2.nicknames))]
            or name2_first
            in [i.casefold() for i in (set() if name1.nicknames is None else set(name1.nicknames))]
        ):
            score += 0.5
        else:
            first_score = fuzz.WRatio(name1_first, name2_first) / 100
            score += first_score - 0.5

    if name1_last is not None and name2_last is not None:
        if name1_last == name2_last:
            score += 0.5
        else:
            last_score = fuzz.WRatio(name1_last, name2_last) / 100
            score += last_score - 0.5

    if name1_middle is not None and name2_middle is not None:
        # Check for full name match
        if (
            (len(name1_middle) == 1 and len(name2_middle) == 1 and name1_middle == name2_middle)
            or (len(name1_middle) == 1 and len(name2_middle) > 1 and name1_middle == name2_middle[0])
            or (len(name1_middle) > 1 and len(name2_middle) == 1 and name1_middle[0] == name2_middle)
        ):
            score += 0.2
        elif (len(name1_middle) == 1 and len(name2_middle) == 1 and name1_middle != name2_middle) or (
            (len(name1_middle) > 1 or len(name2_middle) > 1) and name1_middle[0] != name2_middle[0]
        ):
            score -= 0.2
        elif len(name1_middle) > 1 and len(name2_middle) > 1 and name1_middle == name2_middle:
            score += 0.3
        else:
            middle_score = fuzz.WRatio(name1_middle, name2_middle) / 400
            score += middle_score - 0.125

    return score


def parse_name(name: str) -> PersonName:
    parsed_name = HumanName(name)
    parsed_name.capitalize(force=True)

    ret = PersonName(
        first=parsed_name.first,
        middle=parsed_name.middle,
        last=parsed_name.last,
        nicknames=list(get_all_names(parsed_name.first, include_self=False)),
        suffix=parsed_name.suffix,
        title=parsed_name.title,
    )
    ret.standardize()

    return ret


class NameTyper:
    def __init__(self):
        # Load the first and last names
        self.first_scores: dict[str, int] = json.load(
            open(os.path.join(BASE_DIR, "static", "census_baby_names", "first_to_score.json"))
        )
        self.last_scores: dict[str, int] = json.load(
            open(os.path.join(BASE_DIR, "static", "census_surnames", "last_to_score.json"))
        )

        # Load the unigram scores
        self.word_scores: dict[str, int] = json.load(
            open(os.path.join(BASE_DIR, "static", "english_word_freq", "word_to_score.json"))
        )

    def is_individual(self, name: str, verbose: bool = False) -> bool:
        """
        Determines whether a name is an individual.

        Parameters
        ----------
            name (str): The name to check.

        Returns
        -------
            bool: Whether the name is an individual.

        Notes
        -----
            Separate into words
            For each word, decide if it's a name or word (can be a score?). If it's in both, assume name
            Based on all the scores, decide the final bool: +scores for names, -scores for words, take sign of final value
        """
        name = str(name).casefold()
        common_suffixes = ["jr", "sr", "ii", "iii", "iv"]
        ignore_suffixes = ["p.a.", "o.d."]
        common_company_suffixes = [
            "llc",
            "corp",
            "inc",
            "llp",
            "ltd",
            "assn",
            "club",
            "assoc",
            "fund",
            "assocs",
        ]

        # These are words where if they are included as any substring, the name is likely a company
        common_company_any_inclusions = [
            "company",
            "group",
            "assoc's",
            "association",
            "associates",
            "partners",
            "partnership",
            "corporation",
            "solutions",
            "consulting",
            "holdings",
            "ventures",
            "investments",
            "collective",
            "enterprises",
            "industries",
            "incorporated",
            "international",
            "services",
            "systems",
            "innovations",
            "technologies",
            "worldwide",
            "finance",
            "capital",
            "health",
            "pharmaceutical",
            "insurance",
            "technology",
            "energy",
            "automotive",
            "agency",
            "digital",
            "financial",
            "media",
            "design",
            "productions",
            "logistics",
            "marketing",
            "management",
            "development",
            "creative",
            "resources",
            "advisory",
            "brokerage",
            "distribution",
            "realty",
            "real estate",
            "trading",
            "construction",
            "engineering",
            "biotech",
            "manufacturing",
            "security",
        ]

        if any(word in name for word in common_company_any_inclusions):
            if verbose:
                print(f"{name} had common company any inclusion")
            return False

        common_comm_suffixes = ["pac", "comm", "committee", "party", "pty"]
        state_vals = [i.casefold() for i in STATE_NAME_TO_ABBREV.values()]

        ENGLISH_WORD_MULTIPLIER = 1

        for suffix in ignore_suffixes:
            name = name.replace(suffix, "")

        words = re.split(r"[,\ \(\)\./\\\s]+", name)
        words = [word for word in words if word != ""]
        name_score = 0
        for word in words:
            if word in common_company_suffixes + common_comm_suffixes + state_vals:
                if verbose:
                    print(f"{word} is a common non-individual word")
                return False
            if word in common_suffixes:
                if verbose:
                    print(f"{word} is a common suffix")
                name_score += 50000

            word_english_score = self.word_scores.get(word, 0)
            word_first_score = self.first_scores.get(word, 0)
            word_last_score = self.last_scores.get(word, 0)
            word_name_score = max(word_first_score, word_last_score)

            total_score = -word_english_score * ENGLISH_WORD_MULTIPLIER + word_name_score

            if total_score == 0:
                total_score = -100

            if verbose:
                print(
                    f"{word}: (english score) {word_english_score}, (first score) {word_first_score}, (last score) {word_last_score}, (total score) {total_score}"
                )

            name_score += total_score

        if verbose:
            print(f"Final score: {name_score}")

        return name_score > 0


if __name__ == "__main__":
    typer = NameTyper()
    str_input = input(">>> ")
    while str_input != "exit":
        print(typer.is_individual(str_input, verbose=True))
        str_input = input(">>> ")
