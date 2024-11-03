from typing import Optional
from scourgify import NormalizeAddress
from pydantic import BaseModel
from rapidfuzz import fuzz
from postal.parser import parse_address as postal_parse_address


class PostalAddress(BaseModel):
    house_number: Optional[str] = None
    road: Optional[str] = None
    unit: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postcode: Optional[str] = None


def parse_address(address: str) -> PostalAddress:
    ret = PostalAddress()
    parsed_address = postal_parse_address(address)
    for value, field in parsed_address:
        if field in PostalAddress.model_fields:
            setattr(ret, field, value)
    return ret


def address_similarity(address1: str, address2: str) -> float:
    """
    Calculate the similarity between two addresses.

    Parameters
    ----------
    address1 : str
        The first address to compare.
    address2 : str
        The second address to compare.

    Returns
    -------
    dict
        A dictionary containing the similarity scores for the address components.
    """
    full_score_str = fuzz.WRatio(address1, address2) / 100 - 0.2

    try:
        parsed_address1 = parse_address(address1)
        parsed_address2 = parse_address(address2)
    except Exception as e:
        print("returning full score", e)
        return full_score_str
    
    house_number_score = None if parsed_address1.house_number is None or parsed_address2.house_number is None else ((fuzz.ratio(parsed_address1.house_number, parsed_address2.house_number) / 100)**3 * 2 - 1)
    road_score = None if parsed_address1.road is None or parsed_address2.road is None else ((fuzz.WRatio(parsed_address1.road, parsed_address2.road) / 100) * 2 - 1)
    city_score = None if parsed_address1.city is None or parsed_address2.city is None else ((fuzz.WRatio(parsed_address1.city, parsed_address2.city) / 100) * 2 - 1)
    state_score = None if parsed_address1.state is None or parsed_address2.state is None else (1 if parsed_address1.state == parsed_address2.state else -1)
    zip_score = None if parsed_address1.postcode is None or parsed_address2.postcode is None else ((fuzz.ratio(parsed_address1.postcode, parsed_address2.postcode) / 100)**2 * 2 - 1)

    print(house_number_score, road_score, city_score, state_score, zip_score)

    to_count = [x for x in [house_number_score, road_score, city_score, state_score, zip_score] if x is not None]

    if all(part is None for part in to_count):
        return full_score_str

    full_score = max(0, sum(to_count) / len(to_count))

    return max(full_score, full_score_str)


if __name__ == "__main__":
    command = input(">>>")
    while command != "q":
        address2 = input(">>>")
        print(address_similarity(command, address2))
        command = input(">>>")
