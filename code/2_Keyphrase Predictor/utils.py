import re

def parse_source(source_str):
    """
    Parses the source string into a dictionary.
    """
    pattern = r'([^:：]+):\s*([^:：]+)(?=\s[^:：]+:|$)'
    matches = re.findall(pattern, source_str)
    return {key.strip(): value.strip().rstrip(',') for key, value in matches}

def combine_source_elements(source_dict, separator=" [SEP] "):
    """
    Combines source elements into a single string and records positions.
    """
    combined = ""
    positions = []
    current_pos = 0
    for key, value in source_dict.items():
        element = f"{key}: {value}"
        if combined:
            combined += separator
            current_pos += len(separator)
        combined += element
        start, end = current_pos, current_pos + len(element)
        positions.append((element, start, end))
        current_pos = end
    return combined, positions
