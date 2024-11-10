import json
import os
import random
import re
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import Callable, Optional

import pandas as pd
import swifter
from nameparser.config.conjunctions import CONJUNCTIONS
from nameparser.config.prefixes import PREFIXES
from nameparser.config.regexes import REGEXES
from nameparser.config.suffixes import SUFFIX_ACRONYMS
from nameparser.config.titles import TITLES
from tabulate import tabulate
from termcolor import colored

REGEX_MAP: dict[str, re.Pattern] = {regex[0]: regex[1] for regex in REGEXES}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class NameParts(Enum):
    TITLE = "title"
    FIRST = "first"
    MIDDLE = "middle"
    LAST = "last"
    SUFFIX = "suffix"
    IGNORE = "ignore"


class ParseOptions(Enum):
    IGNORE = "ignore"
    MARK_DELETE = "mark_delete"
    FALLBACK = "fallback"


MAX_N_PARTS = 10


class NameParser:
    def __init__(self, names: Optional[pd.Series] = None):
        self.first_scores: dict[str, int] = json.load(
            open(os.path.join(BASE_DIR, "static", "census_baby_names", "first_to_score.json"))
        )
        self.last_scores: dict[str, int] = json.load(
            open(os.path.join(BASE_DIR, "static", "census_surnames", "last_to_score.json"))
        )

        SURE_TITLES = set(
            [
                "mr",
                "mrs",
                "ms",
                "miss",
                "mister",
                "dr",
                "prof",
            ]
        )
        self.title_scores: dict[str, int] = {
            # 100% sure titles
            **{title: 100_000 for title in SURE_TITLES},
            **{title: 1_000 for title in (set(TITLES) - SURE_TITLES)},
        }

        SURE_SUFFIXES = set(
            [
                "jr",
                "sr",
                "ii",
                "iii",
                "iv",
                "v",
                "md",
                "pa",
                "pdh",
                "esq",
                "esquire",
                "junior",
                "jnr",
                "snr",
            ]
        )
        self.suffix_scores: dict[str, int] = {
            **{suffix: 100_000 for suffix in SURE_SUFFIXES},
            **{suffix: 1_000 for suffix in (set(SUFFIX_ACRONYMS) - SURE_SUFFIXES)},
        }

        self.name_parts_to_mapping: dict[NameParts, Callable[[str], int]] = {
            NameParts.TITLE: lambda x: self.title_scores.get(
                x, -100_000
            ),  # Titles are more enummed than other fields
            NameParts.SUFFIX: lambda x: self.suffix_scores.get(x, -100_000),  # As are suffixes
            NameParts.FIRST: lambda x: self.first_scores.get(x, 0),
            NameParts.LAST: lambda x: self.last_scores.get(x, 0),
            NameParts.MIDDLE: lambda x: self.first_scores.get(x, 0)
            / 3,  # This ensures we never choose a middle name before a first name
        }

        self.names = names
        self.DELIMETERS = [" ", ",", "(", ")", "&"]
        self.KEEP_REGEX = re.compile(r"[^a-zA-Z0-9 ,\(\)&\-]+")

        # The parse map maps formats to a list of possible name part assignments
        self.parse_map: dict[str, list[list[NameParts]] | ParseOptions] = None

        self.format_map: dict[str, list[int]] = None

        if self.names is not None:
            # Make an output dataframe with original and processed names
            self.df_output = pd.DataFrame(
                {
                    "original": self.names,
                    "processed": self.names,
                    "action": pd.NA,
                    "title": pd.NA,
                    "first": pd.NA,
                    "middle": pd.NA,
                    "last": pd.NA,
                    "suffix": pd.NA,
                }
            )

            # Re-index
            self.df_output.reset_index(drop=True, inplace=True)

            # Pre-process the names
            empty_re = re.compile("")
            for _re in (
                REGEX_MAP["quoted_word"] or empty_re,
                REGEX_MAP["double_quotes"] or empty_re,
                REGEX_MAP["parenthesis"] or empty_re,
                REGEX_MAP["emoji"] or empty_re,
            ):
                # Replace these in the series
                self.df_output["processed"] = self.df_output["processed"].str.replace(_re, "", regex=True)

            # Remove double spaces and non-alphanumeric characters
            self.df_output["processed"] = (
                self.df_output["processed"]
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
                .str.replace(self.KEEP_REGEX, "", regex=True)
                .str.casefold()
            )

            # Remove duplicate words
            self.df_output["processed"] = (
                self.df_output["processed"].str.split().apply(lambda x: list(dict.fromkeys(x))).str.join(" ")
            )

            print(f"Pre-processed {len(self.names):,} names. Scanning for formats...")

            self._scan()

    def _detect_string_format(self, string: str) -> str:
        """
        Detect the format of a string.

        Parameters
        ----------
        string: str
            The string to detect the format of.

        Returns
        -------
        str
            The format of the string.
        """
        format = "X"
        # Iterate through, adding X or a delimeter until the end
        for char in string:
            if char in self.DELIMETERS:
                format += char
            else:
                format += "X"
        format = re.sub(r"X+", "X", format)
        return format

    def _scan(self):
        """
        Find all unique name formats.

        Spaces and commas are considered delimeters.

        Parameters
        ----------
        values: pd.Series
            A pandas series of names.
        """
        format_map = {}
        for idx, name in enumerate(self.df_output["processed"]):
            format = self._detect_string_format(name)
            if format in format_map:
                format_map[format].append(idx)
            else:
                format_map[format] = [idx]

        self.format_map = format_map

    def _auto_assign_formats(self, sample_pct: float = 0.01):
        """
        Automatically guess the formats of the names and assign a default parse map.

        Parameters
        ----------
        sample_pct: float
            The percentage of names to sample from each format.
        """
        format_to_option_counts: dict[str, dict[tuple[NameParts], list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for format, indices in self.format_map.items():
            num_sample = int(max(len(indices) * sample_pct, min(100, len(indices))))
            sample_indices = random.choices(indices, k=num_sample)
            samples = self.df_output.loc[sample_indices, "processed"]
            for sample in samples:
                try:
                    _, name_parts, _ = self.parse_individual_name(sample, verbose=False, use_heuristics=False)
                    format_to_option_counts[format][name_parts].append(sample)
                except Exception as e:
                    print(f"Error parsing {sample}: {e}")

        # Rank by popularity of format from format_map
        formats_ordered = sorted(self.format_map.items(), key=lambda x: len(x[1]), reverse=True)
        for format, indices in formats_ordered:
            option_counts = format_to_option_counts[format]
            print(f"Format: {format}")
            total = len(indices)
            total_this_format = sum([len(samples) for samples in option_counts.values()])
            for option, samples in sorted(option_counts.items(), key=lambda x: len(x[1]), reverse=True):
                if len(samples) / total_this_format > 0.01:
                    sample_str = ", ".join(f'"{sample}"' for sample in samples[: min(3, len(samples))])
                    print(f"\tOption: {option} ({len(samples) / total_this_format:.2%}) ({sample_str})")

        # Any format with more than 10% can be assigned to the parse map
        format_map: dict[str, list[list[NameParts]]] = defaultdict(list)
        for format, option_counts in format_to_option_counts.items():
            total = sum(option_counts.values())
            for option, count in option_counts.items():
                if count / total > 0.1:
                    format_map[format].append(option)

        self.parse_map = format_map

        self.print_parse_map()

    def print_parse_map(self):
        """
        Print the parse map.
        """
        for format, options in self.parse_map.items():
            print(f"'{format}': {options}")

    def print_formats(self, min_pct: float = 0.5):
        """
        Print the formats in the parse map.

        Parameters
        ----------
        min_pct: float
            The minimum percentage of names that must be in the format to print it.
        """
        accounted_for = 0
        name_map = dict(sorted(self.format_map.items(), key=lambda x: len(x[1]), reverse=True))
        num_skipped = 0
        unaccounted_for = 0
        for format, indices in list(name_map.items()):
            if len(indices) / len(self.df_output) >= min_pct / 100:
                examples = [f'"{i}"' for i in self.sample(format, 3)]
                print(
                    f"'{format}' ({len(indices):,} - {len(indices) / len(self.df_output):.2%}). Examples: {', '.join(examples)}"
                )
                accounted_for += len(indices)
            else:
                num_skipped += 1
                unaccounted_for += len(indices)
        print(
            f"And {num_skipped:,} more formats, accounting for {unaccounted_for / len(self.df_output):.2%} of the names."
        )

    def set(self, parse_map: dict[str, list[NameParts]], delete_on_too_long: bool = True):
        """
        Set the parse map.

        Parameters
        ----------
        parse_map: dict[str, list[NameParts]]
            The new parse map.
        """
        if delete_on_too_long:
            for format in parse_map:
                if format.count("X") > MAX_N_PARTS:
                    parse_map[format] = ParseOptions.MARK_DELETE

        self.parse_map = parse_map

    def update_parse_map(self, parse_map: dict[str, list[NameParts]]):
        """
        Update the parse map.

        Parameters
        ----------
        parse_map: dict[str, list[NameParts]]
            The new parse map.
        """
        self.parse_map.update(parse_map)

    def sample(self, format: str, n: int = 10):
        """
        Sample n random names for a given format.

        Parameters
        ----------
        format: str
            The format to sample from.
        n: int
            The number of names to sample.

        Returns
        -------
        list[str]
            The sampled names.
        """
        indices = random.choices(self.format_map[format], k=n)
        return [str(name) for name in self.df_output.loc[indices, "processed"]]

    def _add_heuristic_scores(
        self, part_to_category_scores: dict[str, dict[NameParts, int]], format: str, parts: list[str]
    ):
        """
        Add rule-based heuristic scores to the matrix. This allows accounting for the fact
        that names of a certain format are more likely to be parsed a certain way, initials are more likely to be middle or first names, etc.

        This function edits part_to_category_scores in place.

        Parameters
        ----------
        part_to_category_scores: dict[str, dict[NameParts, int]]
            The matrix of part to category scores.
        format: str
            The format of the name.
        parts: list[str]
            The parts of the name.
        """
        # Make a mapping from formats to the bumps that each category gets at each position
        bump_map = {
            "X": {
                NameParts.TITLE: [0],
                NameParts.FIRST: [10_000],
                NameParts.MIDDLE: [0],
                NameParts.LAST: [25_000],
                NameParts.SUFFIX: [0],
            },
            "X, X": {
                NameParts.TITLE: [0, 0],
                NameParts.FIRST: [0, 25_000],
                NameParts.MIDDLE: [0, 0],
                NameParts.LAST: [25_000, 0],
                NameParts.SUFFIX: [0, 0],
            },
            "X X": {
                NameParts.TITLE: [0, 0],
                NameParts.FIRST: [25_000, 0],
                NameParts.MIDDLE: [0, 0],
                NameParts.LAST: [0, 25_000],
                NameParts.SUFFIX: [0, 0],
            },
            "X X, X": {
                NameParts.TITLE: [0, 0, 0],
                NameParts.FIRST: [0, 0, 25_000],
                NameParts.MIDDLE: [15_000, 15_000, 0],
                NameParts.LAST: [15_000, 15_000, 0],
                NameParts.SUFFIX: [0, 0, 0],
            },
            "X, X X": {
                NameParts.TITLE: [0, 0, 0],
                NameParts.FIRST: [0, 25_000, 0],
                NameParts.MIDDLE: [0, 0, 25_000],
                NameParts.LAST: [25_000, 0, 0],
                NameParts.SUFFIX: [0, 0, 0],
            },
            "X X X": {
                NameParts.TITLE: [10_000, 0, 0],
                NameParts.FIRST: [25_000, 0, 0],
                NameParts.MIDDLE: [0, 10_000, 0],
                NameParts.LAST: [0, 10_000, 25_000],
                NameParts.SUFFIX: [0, 0, 10_000],
            },
            "X X X X": {
                NameParts.TITLE: [25_000, 0, 0, 0],
                NameParts.FIRST: [0, 25_000, 0, 0],
                NameParts.MIDDLE: [0, 0, 25_000, 0],
                NameParts.LAST: [0, 0, 0, 25_000],
                NameParts.SUFFIX: [0, 0, 0, 0],
            },
        }

        if format in bump_map:
            for category, bump_scores in bump_map[format].items():
                for i, bump_score in enumerate(bump_scores):
                    part_to_category_scores[parts[i]][category] += bump_score

        # Initials get a middle name bump
        for part in parts:
            if len(part) == 1:
                part_to_category_scores[part][NameParts.MIDDLE] += 25_000

    def _choose_best_assignment(self, name: str, format: str, options: list[list[NameParts]]):
        """
        Choose the best assignment of parts to categories.

        Parameters
        ----------
        name: str
            The name to assign.
        format: str
            The format of the name.
        options: list[list[NameParts]]
            The options to choose from.

        Returns
        -------
        pd.Series[str]
            The assigned parts of the name.
        """
        process_result = self._process_name(name)
        if process_result is None:
            return pd.Series({})
        parts, part_to_category_scores, original_parts, original_part_to_new_idx = process_result

        adjusted_scores: dict[str, dict[NameParts, int]] = self._recalculate_scores(
            part_to_category_scores, parts, list(set(NameParts) - {NameParts.IGNORE})
        )

        if len(parts) != len(options[0]):
            _, parsed_parts, _ = self._parse_name(
                format,
                parts,
                adjusted_scores,
                original_parts,
                original_part_to_new_idx,
                verbose=False,
                use_heuristics=False,
            )

            # Reform parsed_parts into the title, first, middle, last, suffix order
            name_mapping: dict[str, str] = defaultdict(lambda: "")
            for i in range(len(parts)):
                name_mapping[parsed_parts[i].value] += parts[i]
            return pd.Series(name_mapping)

        best_score = -float("inf")
        best_option = None
        for option in options:
            score = sum(adjusted_scores[parts[i]][option[i]] for i in range(len(parts)))
            if score > best_score:
                best_score = score
                best_option = option

        name_mapping: dict[str, str] = defaultdict(lambda: "")
        for i in range(len(parts)):
            name_mapping[best_option[i].value] += parts[i]
        return pd.Series(name_mapping)

    def _display_matrix(
        self, scores: dict[str, dict[NameParts, int]], parts: list[str], categories: list[NameParts]
    ):
        """
        Display the matrix of part to category scores.

        Parameters
        ----------
        scores: dict[str, dict[NameParts, int]]
            The matrix of part to category scores.
        parts: list[str]
            The parts of the name.
        categories: list[NameParts]
            The categories of the name.
        """
        table = []
        table.append(["Category"] + parts)
        for category in categories:
            table.append([category.value] + [scores[part][category] for part in parts])
        print(tabulate(table, headers="firstrow") + "\n")

    def _process_name(self, name: str):
        """
        Process a name to get the parts and matrix.

        Parameters
        ----------
        name: str
            The name to process.

        Returns
        -------
        tuple[list[str], dict[str, dict[NameParts, int]], list[str], dict[str, int]]
            The parts of the name, the part to category scores, the original parts, and the original part to new index.
        """
        # Split the name by the delimeters
        parts = []
        part = ""
        for char in name:
            if char in self.DELIMETERS:
                parts.append(part)
                part = ""
            else:
                part += char
        parts.append(part)
        parts = [part for part in parts if part]
        original_parts = deepcopy(parts)

        if len(parts) > MAX_N_PARTS:
            return None

        # Combine parts: combine prefixes with their next words (van der, de la, etc.)
        original_part_to_new_idx: dict[str, int] = {}
        i = 0
        offset = 0
        while i < len(parts):
            if parts[i] in PREFIXES | CONJUNCTIONS:
                j = i + 1
                while j < len(parts) and parts[j] in PREFIXES | CONJUNCTIONS:
                    parts[i] = parts[i] + " " + parts[j]
                    parts[j] = ""
                    original_part_to_new_idx[original_parts[j]] = i - offset
                    original_part_to_new_idx[original_parts[i]] = i - offset
                    offset += 1
                    j += 1
                # Combine the prefix with the next word
                if j < len(parts):
                    parts[i] = parts[i] + " " + parts[j]
                    parts[j] = ""
                    original_part_to_new_idx[original_parts[j]] = i - offset
                    original_part_to_new_idx[original_parts[i]] = i - offset
                    offset += 1
            i += 1

        parts = [part for part in parts if part]

        unmapped_categories: set[NameParts] = set(NameParts) - {NameParts.IGNORE}

        # 4. Classify the parts. We don't want to assign a part to a category if there's another that would assign better.
        possible_subname_delimiters = [" ", "-"]
        part_to_category_scores: dict[str, dict[NameParts, int]] = defaultdict(lambda: defaultdict(int))
        for part in parts:
            for category in unmapped_categories:
                for subname_delimiter in possible_subname_delimiters:
                    if subname_delimiter in part:
                        score = max(
                            self.name_parts_to_mapping[category](part),
                            self.name_parts_to_mapping[category](part.split(subname_delimiter)[0]),
                        )
                        break
                else:
                    score = self.name_parts_to_mapping[category](part)
                part_to_category_scores[part][category] = score

        return parts, part_to_category_scores, original_parts, original_part_to_new_idx

    def _recalculate_scores(
        self, scores: dict[str, dict[NameParts, int]], parts: list[str], categories: list[NameParts]
    ) -> dict[str, dict[NameParts, int]]:
        """
        Recalculate a score matrix.

        Parameters
        ----------
        scores: dict[str, dict[NameParts, int]]
            The score matrix to recalculate.
        parts: list[str]
            The parts of the name.
        categories: list[NameParts]
            The categories of the name.

        Returns
        -------
        dict[str, dict[NameParts, int]]
            The recalculated score matrix.
        """
        new_part_to_category_scores = defaultdict(lambda: defaultdict(int))
        for part in parts:
            for category in categories:
                new_part_to_category_scores[part][category] = scores[part][category] - (
                    (sum(max(0, val) for cat, val in scores[part].items() if cat != category) / len(parts))
                    + (
                        sum(
                            max(0, scores[other_part][category])
                            for other_part in scores
                            if other_part != part
                        )
                        / len(categories)
                    )
                )
        return new_part_to_category_scores

    def process_and_parse_name(self, name: str, verbose: bool = False, use_heuristics: bool = True):
        """
        Process and parse a name.

        Parameters
        ----------
        name: str
            The name to process and parse.
        verbose: bool
            Whether to display the matrix of part to category scores.
        use_heuristics: bool
            Whether to use the heuristic scores.

        Returns
        -------
        pd.Series[str]
            The parts of the name (a mapping from category to part).
        """
        format = self._detect_string_format(name)
        process_result = self._process_name(name)
        if process_result is None:
            return pd.Series({})
        parts, part_to_category_scores, original_parts, original_part_to_new_idx = process_result

        _, name_parts, _ = self._parse_name(
            format,
            parts,
            part_to_category_scores,
            original_parts,
            original_part_to_new_idx,
            verbose,
            use_heuristics,
        )

        name_mapping: dict[str, str] = defaultdict(lambda: "")
        for i in range(len(original_parts)):
            name_mapping[name_parts[i].value] += original_parts[i]

        return pd.Series(name_mapping)

    def _parse_name(
        self,
        format: str,
        parts: list[str],
        part_to_category_scores: dict[str, dict[NameParts, int]],
        original_parts: list[str],
        original_part_to_new_idx: dict[str, int],
        verbose: bool = False,
        use_heuristics: bool = True,
    ):
        """
        Parse a single name.

        Parameters
        ----------
        parts: list[str]
            The parts of the name.
        part_to_category_scores: dict[str, dict[NameParts, int]]
            The part to category scores.
        original_parts: list[str]
            The original parts of the name.
        original_part_to_new_idx: dict[str, int]
            The original part to new index.
        verbose: bool
            Whether to display the matrix of part to category scores.
        use_heuristics: bool
            Whether to use the heuristic scores.

        Returns
        -------
        dict[str, NameParts]
            A dictionary with the parts of the name.
        tuple[NameParts]
            The parts of the name.
        str
            The format of the name.

        TODO
        ----
        - Re-score after each removal. Once a player (part) is drafted, the field resets.
        """
        part_to_category: dict[str, NameParts] = {}
        unmapped_categories: set[NameParts] = set(NameParts) - {NameParts.IGNORE}
        unmapped_parts: set[str] = set(parts)

        # Bump
        if verbose:
            print(colored("Initial scores:", "yellow"))
            self._display_matrix(part_to_category_scores, parts, list(unmapped_categories))
        if use_heuristics:
            if format.count("X") == len(parts):
                self._add_heuristic_scores(part_to_category_scores, format, parts)
            if verbose:
                print(colored("After heuristic scores:", "yellow"))
                self._display_matrix(part_to_category_scores, parts, list(unmapped_categories))

        # 4.1 Update scores to reflect the other parts and categories
        part_to_category_scores = self._recalculate_scores(
            part_to_category_scores, unmapped_parts, unmapped_categories
        )
        if verbose:
            print(colored("After recalculating scores:", "yellow"))
            self._display_matrix(part_to_category_scores, parts, list(unmapped_categories))

        while unmapped_parts and unmapped_categories:
            # Find the most "obvious" match. Continue to find the most obvious match until all parts are assigned.
            # "Obvious" is defined as the cell with the highest sedoku score (relative to its row and column)
            max_score = -float("inf")
            max_part = ""
            max_category = None

            for part in part_to_category_scores:
                for category in unmapped_categories:
                    if part_to_category_scores[part][category] > max_score:
                        max_score = part_to_category_scores[part][category]
                        max_part = part
                        max_category = category

            # Never assign the middle name before the first name and last name
            if max_category == NameParts.MIDDLE and NameParts.FIRST in unmapped_categories:
                max_category = NameParts.FIRST
            elif max_category == NameParts.MIDDLE and NameParts.LAST in unmapped_categories:
                max_category = NameParts.LAST

            part_to_category[max_part] = max_category
            unmapped_parts.discard(max_part)
            if max_category not in [NameParts.SUFFIX]:
                unmapped_categories.discard(max_category)

            if verbose:
                print(colored(f"Assigning {max_part} to {max_category}:", "green"))

            # Re-calculate the scores without the max part and max category
            part_to_category_scores = self._recalculate_scores(
                part_to_category_scores, unmapped_parts, unmapped_categories
            )

        final_name_parts_list: list[NameParts] = []
        for part in original_parts:
            if part in original_part_to_new_idx:
                new_idx = original_part_to_new_idx[part]
                new_part = parts[new_idx]
                final_name_parts_list.append(part_to_category[new_part])
            else:
                final_name_parts_list.append(part_to_category[part])

        return part_to_category, tuple(final_name_parts_list), format

    def parse_individual_name(self, name: str, verbose: bool = False, use_heuristics: bool = True):
        """
        Parse a single name without context.

        Parameters
        ----------
        name: str
            The name to parse.
        verbose: bool
            Whether to display the matrix of part to category scores.

        Returns
        -------
        dict
            A dictionary with the parts of the name.
        name_parts: tuple[NameParts]
            The parts of the name.
        str
            The format of the name.
        """
        empty_re = re.compile("")
        for _re in (
            REGEX_MAP["quoted_word"] or empty_re,
            REGEX_MAP["double_quotes"] or empty_re,
            REGEX_MAP["parenthesis"] or empty_re,
            REGEX_MAP["emoji"] or empty_re,
        ):
            if _re.search(name):
                name = _re.sub("", name)

        # Remove double spaces
        name = re.sub(self.KEEP_REGEX, "", re.sub(r"\s+", " ", name).strip()).casefold()

        # Remove duplicate words
        name = " ".join(list(dict.fromkeys(name.split(" "))))

        return self.process_and_parse_name(name, verbose, use_heuristics)

    def _pattern_to_regex(self, format: str):
        # Create a regex fragment for 'X' by excluding unallowed characters
        # Assuming 'X' should match one or more allowed characters
        excluded = "".join(re.escape(c) for c in self.DELIMETERS)
        x_pattern = f"[^{excluded}]+"

        parts = re.split("(X)", format)
        regex_parts = []
        for part in parts:
            if part == "X":
                regex_parts.append(f"({x_pattern})")
            else:
                regex_parts.append(part)

        regex_pattern = "".join(regex_parts)

        # Compile the final regex
        final_regex = f"^{regex_pattern}$"

        return final_regex

    def parse(
        self, on_no_format: ParseOptions = ParseOptions.FALLBACK, print_pct: float = 0.5
    ) -> pd.DataFrame:
        """
        Parse the dataframe.

        Parameters
        ----------
        on_no_format: ParseOptions
            The action to take if no format is found.
        print_pct: float
            The percentage of names that must be in the format to print it.
        """
        if self.parse_map is None:
            raise ValueError("No parse map set.")

        num_not_verbose = 0

        # For each format, do a vectorized regex parse
        for format, parse_action in self.parse_map.items():
            verbose = (len(self.format_map[format]) / len(self.df_output)) * 100 >= print_pct
            num_not_verbose += not verbose

            if isinstance(parse_action, list) and isinstance(parse_action[0], NameParts):
                if verbose:
                    print(f"Using regex for format {format} with only one parsing option.")
                regex = re.compile(self._pattern_to_regex(format))
                result = self.df_output.loc[self.format_map[format], "processed"].str.extract(regex)

                # Assign the parts in the order of parse_action
                inserted_parts: set[NameParts] = set()
                for i, part in enumerate(parse_action):
                    if part == NameParts.IGNORE:
                        continue
                    if part not in inserted_parts:
                        self.df_output.loc[self.format_map[format], part.value] = result[i]
                        inserted_parts.add(part)
                    else:
                        # Append to the strings in this column
                        self.df_output.loc[self.format_map[format], part.value] = (
                            self.df_output.loc[self.format_map[format], part.value] + " " + result[i]
                        )

                self.df_output.loc[self.format_map[format], "action"] = "completed - regex"
            elif isinstance(parse_action, list) and isinstance(parse_action[0], list):
                if verbose:
                    print(f"Multiple options for format {format}. Scoring each name individually.")
                self.df_output.loc[
                    self.format_map[format], ["title", "first", "middle", "last", "suffix"]
                ] = self.df_output.loc[self.format_map[format], "processed"].swifter.apply(
                    lambda name: self._choose_best_assignment(name, format, parse_action)
                )
                self.df_output.loc[self.format_map[format], "action"] = "completed - chose"
            elif parse_action == ParseOptions.FALLBACK:
                if verbose:
                    print(f"Falling back to process_and_parse_name for format {format}.")
                self.df_output.loc[
                    self.format_map[format], ["title", "first", "middle", "last", "suffix"]
                ] = self.df_output.loc[self.format_map[format], "processed"].swifter.apply(
                    self.process_and_parse_name
                )
                self.df_output.loc[self.format_map[format], "action"] = "completed - explicit fallback"
            elif parse_action == ParseOptions.MARK_DELETE:
                if verbose:
                    print(f"Marking {self.format_map[format]} for deletion.")
                self.df_output.loc[self.format_map[format], "action"] = "delete"

        if num_not_verbose > 0:
            print(f"And {num_not_verbose:,} more formats with <{print_pct}% of names each.")

        # For any rows that have no action, perform the on no format action
        if on_no_format == ParseOptions.MARK_DELETE:
            self.df_output.loc[self.df_output["action"].isna(), "action"] = "delete"
        elif on_no_format == ParseOptions.FALLBACK:
            print("Performing catchall fallback.")
            # Parse the names that have no action
            self.df_output.loc[
                self.df_output["action"].isna(), ["title", "first", "middle", "last", "suffix"]
            ] = self.df_output.loc[self.df_output["action"].isna(), "processed"].swifter.apply(
                self.process_and_parse_name
            )
            self.df_output.loc[self.df_output["action"].isna(), "action"] = "completed - catchall fallback"

        return self.df_output
