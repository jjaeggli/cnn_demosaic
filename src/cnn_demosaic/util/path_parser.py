import re
from typing import Iterator

def parse_path_statement(path_statement: str) -> Iterator[str]:
    """
    Parses a path statement to an iterable of path strings.

    A path statement is an expression containing a range, such as:
        '/mount/images/picture[001-100].png'

    Args:
        path_statement: The path statement string to parse.

    Returns:
        An iterator of strings, where each string is a generated path.
    """
    # Find all occurrences of the range pattern
    all_matches = list(re.finditer(r'\[(\d+)-(\d+)\]', path_statement))

    if len(all_matches) > 1:
        raise ValueError("Path statement contains multiple ranges, which is not supported.")

    if not all_matches:
        yield path_statement
        return

    # If there's exactly one match, proceed with parsing it
    match = all_matches[0]
    prefix = path_statement[:match.start()]
    suffix = path_statement[match.end():]
    start_num = int(match.group(1))
    end_num = int(match.group(2))
    padding = len(match.group(1)) # Determine padding from the start number in the bracket

    if start_num > end_num:
        raise ValueError("Start number cannot be greater than end number in path statement range.")

    for i in range(start_num, end_num + 1):
        yield f"{prefix}{str(i).zfill(padding)}{suffix}"
