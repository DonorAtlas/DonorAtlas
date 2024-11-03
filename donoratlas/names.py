from typing import Optional

from pydantic import BaseModel
from nameparser import HumanName
from nicknames import NickNamer

from rapidfuzz import fuzz

nick_namer = NickNamer()


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
    return get_nicknames(name) | get_formal_names(name) | ({name.casefold()} if include_self else set())


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


def name_similarity(name1: Optional[str], name2: Optional[str], name1_parsed: Optional[PersonName] = None, name2_parsed: Optional[PersonName] = None) -> dict[str, float]:
    """
    Calculate the similarity between two names.

    Parameters
    ----------
        name1 (str): The first name.
        name2 (str): The second name.
        name1_parsed (PersonName): The first name parsed.
        name2_parsed (PersonName): The second name parsed.

    Returns
    -------
        dict[str, float]: A dictionary mapping each field to the similarity score for that field. Keys are either just "full", or "first", "last", and "full".

    Notes
    -----
    A score of ~1.0 is taken to mean that "with all the information we have, there is no evidence that these are different people".
        Thus, two names with the same first and last name are considered as good of a match as if they both had matching middle names too.

    A score of ~0.5 is taken to mean that "these people are very unlikely to be the same person, but have some common attribute (same first, same last, etc.)".

    A score of 0 means "that there is no evidence to suggest that these could be the same person".

    TODO
    ----
    Account for the rareness of the names, and initials, etc.
    """
    if name1_parsed is None:
        assert name1 is not None, "name1 must be provided if name1_parsed is not provided"
        name1_parsed = parse_name(name1)
    if name2_parsed is None:
        assert name2 is not None, "name2 must be provided if name2_parsed is not provided"
        name2_parsed = parse_name(name2)
    if name1 is None:
        assert name1_parsed is not None, "name1 must be provided if name1_parsed is not provided"
        name1 = str(name1_parsed)
    if name2 is None:
        assert name2_parsed is not None, "name2 must be provided if name2_parsed is not provided"
        name2 = str(name2_parsed)

    name1_first = None if name1_parsed.first is None else alpha_only(name1_parsed.first.casefold())
    name2_first = None if name2_parsed.first is None else alpha_only(name2_parsed.first.casefold())

    name1_middle = None if name1_parsed.middle is None else alpha_only(name1_parsed.middle.casefold())
    name2_middle = None if name2_parsed.middle is None else alpha_only(name2_parsed.middle.casefold())

    name1_last = None if name1_parsed.last is None else alpha_only(name1_parsed.last.casefold())
    name2_last = None if name2_parsed.last is None else alpha_only(name2_parsed.last.casefold())

    # These scores are between -1 and 1.
    score = {"first": 0, "middle": 0, "last": 0}

    if name1_first is not None and name2_first is not None:
        if name1_first == name2_first:
            score["first"] = 1
        elif (name1_first in [i.casefold() for i in (set() if name2_parsed.nicknames is None else set(name2_parsed.nicknames))]
            or name2_first in [i.casefold() for i in (set() if name1_parsed.nicknames is None else set(name1_parsed.nicknames))]
        ):
            # Calculate the similarity score as if they were the same name, and scale between 0.9 and 1.
            first_score = fuzz.WRatio(name1_first, name2_first) / 1000
            score["first"] = first_score + 0.9
        else:
            first_score = fuzz.WRatio(name1_first, name2_first) / 50
            score["first"] = first_score - 1

    if name1_last is not None and name2_last is not None:
        if name1_last == name2_last:
            score["last"] = 1
        else:
            last_score = fuzz.WRatio(name1_last, name2_last) / 50
            score["last"] = last_score - 1

    if name1_middle is not None and name2_middle is not None:
        # Check for full name match
        if (
            (len(name1_middle) == 1 and len(name2_middle) == 1 and name1_middle == name2_middle)
            or (len(name1_middle) == 1 and len(name2_middle) > 1 and name1_middle == name2_middle[0])
            or (len(name1_middle) > 1 and len(name2_middle) == 1 and name1_middle[0] == name2_middle)
        ):
            score["middle"] = 0.9
        elif (len(name1_middle) == 1 and len(name2_middle) == 1 and name1_middle != name2_middle) or (
            (len(name1_middle) > 1 or len(name2_middle) > 1) and name1_middle[0] != name2_middle[0]):
            score["middle"] = -1
        elif len(name1_middle) > 1 and len(name2_middle) > 1 and name1_middle == name2_middle:
            score["middle"] = 1
        else:
            middle_score = fuzz.WRatio(name1_middle, name2_middle) / 50
            score["middle"] = middle_score - 1

    # Also calculate the full WRatio score, in case of bad parsing
    full_score = fuzz.WRatio(name1, name2) / 100 - 0.1
    full_parsed_score = min(1, (score["first"] / 2) + (score["middle"] / 4) + (score["last"] / 2))

    if full_score > full_parsed_score:
        return {
            "full": full_score
        }
    else:
        return {
            "first": score["first"],
            "last": score["last"],
            "full": min(1, full_parsed_score),
        }


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


if __name__ == "__main__":
    # Test how long the parse_name function takes
    import time
    start_time = time.time()
    for i in range(1000):
        parse_name("Dr. John Smith")
    print(f"Parsing: average time taken: {(time.time() - start_time) / 1000} seconds")

    start_time = time.time()
    for i in range(1000):
        name_similarity("Dr. John Smith", "John Smith")
    print(f"Similarity: average time taken: {(time.time() - start_time) / 1000} seconds")
