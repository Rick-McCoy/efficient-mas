import re
from re import Match

from util.text.korean_dict import (
    COUNT_LIST,
    COUNT_REGEX,
    DIGIT_LIST,
    ENGLISH_DICT,
    HAN_NUM_DICT,
    NUMBER_REGEX,
    NUMERICAL_NAME_LIST,
    SUBDIVISION_LIST,
    TENS_NUMERAL_LIST,
    UNIT_DICT,
    UNIT_REGEX,
)


def sanitize(text: str):
    """Remove nonstandard characters from text.
    Example:
        '한글은\n위대하다.'  --> '한글은 위대하다.'
        '한글은\\n위대하다.'  --> '한글은 위대하다.'
    """
    text = text.replace("\\n", "\n")
    text = " ".join(text.split())
    return text


def normalize_english(text: str):
    def fn(m: Match[str]):
        word = m.group()
        return ENGLISH_DICT.get(word, word)

    text = re.sub(r"([A-Za-z]+)", fn, text)
    return text


def normalize_quote(text: str):
    quotes = "'＂“‘’”`\""
    for quote in quotes:
        text = text.replace(quote, " ")
    text = sanitize(text)
    return text


def normalize_number(text: str):
    text = re.sub(NUMBER_REGEX + UNIT_REGEX, lambda x: number_to_korean(x, True), text)
    text = re.sub(NUMBER_REGEX + COUNT_REGEX, lambda x: number_to_korean(x, True), text)
    text = re.sub(NUMBER_REGEX, lambda x: number_to_korean(x, False), text)
    return text


def division_to_korean(division: int, count=False):
    assert 0 <= division < 10000
    if count:
        assert (
            division < 100
        ), f"division must be less than 100 when count is True: {division}"
        tens_numeral = division // 10
        digit = division % 10
        return TENS_NUMERAL_LIST[tens_numeral] + COUNT_LIST[digit]

    if division == 0:
        return ""

    result = ""
    for i in range(4):
        digit = (division % 10 ** (i + 1)) // 10**i
        if digit == 1 and i > 0:
            result = SUBDIVISION_LIST[i] + result
        elif digit > 0:
            result = DIGIT_LIST[digit] + SUBDIVISION_LIST[i] + result

    return result


def lead_zeros(text: str):
    if text == "0":
        return False
    non_zero_index = next((i for i, c in enumerate(text) if c != "0"), None)
    if non_zero_index == 1 and text[1] == ".":
        return False
    return non_zero_index != 0


def number_to_korean(num_match: Match, is_count=False):
    if is_count:
        num_str, unit_str = num_match.group(1), num_match.group(2)
    else:
        num_str, unit_str = num_match.group(), ""

    assert isinstance(num_str, str)
    assert isinstance(unit_str, str)
    if unit_str in UNIT_DICT:
        is_count = False
        unit_str = UNIT_DICT[unit_str]

    num_str = num_str.replace(",", "")
    plus = num_str.startswith("+")
    num_str = num_str.lstrip("+")
    minus = num_str.startswith("-")
    num_str = num_str.lstrip("-")

    zero_lead = lead_zeros(num_str)
    korean_num = ""

    if zero_lead:
        for digit in num_str:
            if digit in HAN_NUM_DICT:
                korean_num += HAN_NUM_DICT[digit]
            elif digit == ".":
                korean_num += "쩜 "
            else:
                raise ValueError(f"Invalid digit: {digit} in {num_str}")
            korean_num = korean_num.replace(HAN_NUM_DICT["0"], "공")

    elif num_str:
        try:
            num = int(num_str)
        except ValueError:
            num = float(num_str)

        num_str = str(num)

        if isinstance(num, int):
            digit_str, float_str = num_str, None
            if num >= 100:
                is_count = False
        else:
            digit_str, float_str = num_str.split(".")
            is_count = False

        size = len(digit_str)
        remainder = (size - 1) % 4 + 1
        division_list = [digit_str[:remainder]]
        division_list += [digit_str[i : i + 4] for i in range(remainder, size, 4)]
        division_list = [
            division_to_korean(int(division), is_count) for division in division_list
        ]
        division_len = len(division_list)
        for i in range(division_len):
            if division_list[i] != "":
                korean_num += (
                    division_list[i] + NUMERICAL_NAME_LIST[division_len - i - 1]
                )

        if korean_num == "":
            korean_num = HAN_NUM_DICT["0"]

        if float_str is not None and float_str:
            korean_num += "쩜 "
            for digit in float_str:
                korean_num += HAN_NUM_DICT[digit]

    if plus:
        korean_num = "플러스 " + korean_num
    elif minus:
        korean_num = "마이너스 " + korean_num

    return korean_num + unit_str
