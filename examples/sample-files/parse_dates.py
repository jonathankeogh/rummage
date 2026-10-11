from datetime import datetime


def parse_date(text):
    """Parse a date written as 2024-03-05 or 5 March 2024."""
    for fmt in ("%Y-%m-%d", "%d %B %Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"unrecognised date: {text}")
