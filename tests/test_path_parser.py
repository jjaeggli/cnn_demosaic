import pytest
from cnn_demosaic.util.path_parser import parse_path_statement

def test_single_path():
    """Test with a single, simple path."""
    path = "/home/user/image.png"
    expected = ["/home/user/image.png"]
    assert list(parse_path_statement(path)) == expected

def test_path_with_range():
    """Test with a path containing a numerical range."""
    path = "/mount/images/picture[001-003].png"
    expected = [
        "/mount/images/picture001.png",
        "/mount/images/picture002.png",
        "/mount/images/picture003.png",
    ]
    assert list(parse_path_statement(path)) == expected

def test_path_with_different_padding():
    """Test with a path containing a numerical range and different padding."""
    path = "/data/frames/frame[10-12].jpg"
    expected = [
        "/data/frames/frame10.jpg",
        "/data/frames/frame11.jpg",
        "/data/frames/frame12.jpg",
    ]
    assert list(parse_path_statement(path)) == expected

def test_path_with_single_number_in_range():
    """Test with a path where the range specifies a single number."""
    path = "/logs/log[005-005].txt"
    expected = ["/logs/log005.txt"]
    assert list(parse_path_statement(path)) == expected

def test_path_without_range_brackets_but_numbers():
    """Test with a path that has numbers but no range brackets."""
    path = "/archive/document_2023.pdf"
    expected = ["/archive/document_2023.pdf"]
    assert list(parse_path_statement(path)) == expected

def test_path_with_multiple_bracket_sets_raises_error():
    """Test that multiple bracket sets raise a ValueError."""
    path = "/path/to/file[01-02]_[03-04].txt"
    with pytest.raises(ValueError, match="Path statement contains multiple ranges, which is not supported."):
        list(parse_path_statement(path))

def test_invalid_range_start_greater_than_end_raises_error():
    """Test that an invalid range (start > end) raises a ValueError."""
    path = "/images/series[10-01].png"
    with pytest.raises(ValueError):
        list(parse_path_statement(path))

def test_empty_string_input():
    """Test with an empty string input."""
    path = ""
    expected = [""]
    assert list(parse_path_statement(path)) == expected

def test_path_with_non_numeric_characters_in_brackets_are_literal():
    """Test that non-numeric characters in brackets are not parsed as ranges."""
    path = "/data/report[abc-def].csv"
    expected = ["/data/report[abc-def].csv"]
    assert list(parse_path_statement(path)) == expected

def test_path_with_complex_prefix_and_suffix():
    """Test with a more complex prefix and suffix around the range."""
    path = "/var/log/application/debug_[0010-0012]_backup.log"
    expected = [
        "/var/log/application/debug_0010_backup.log",
        "/var/log/application/debug_0011_backup.log",
        "/var/log/application/debug_0012_backup.log",
    ]
    assert list(parse_path_statement(path)) == expected

def test_path_with_no_z_pad():
    """Test that a supplied expression such as [9-10] is parsed as a range."""
    path = "/data/image[9-10].png"
    expected = ["/data/image9.png", "/data/image10.png"]
    assert list(parse_path_statement(path)) == expected
