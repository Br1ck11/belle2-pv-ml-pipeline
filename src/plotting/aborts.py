"""Abort events for plots, parsed by dataset.filter_pv.WindowFilter.

WindowFilter defines timestamps (naive = Asia/Tokyo), ring/category
normalization (missing category -> "Unspecified") and event IDs. This module
only adds the catalogue's ``pressure_burst`` annotation and the category
selection used by the plots.
"""

import re

from dataset.filter_pv import WindowFilter


# A location is accepted only when a complete comma-delimited token has the
# SuperKEKB form, e.g. D06_L12 or D07_H04A. Location-looking text inside a
# component description such as GV(D02_L03) is deliberately not extracted.
PRESSURE_LOCATION = re.compile(r"D\d{2}_[HL][0-9A-Z]+")


def pressure_burst_locations(value) -> tuple[str, ...]:
    """Unique location tokens of a pressure_burst string, in first-seen order."""
    if not isinstance(value, str) or not value.strip():
        return ()
    tokens = (token.strip() for token in value.split(","))
    return tuple(dict.fromkeys(token for token in tokens if PRESSURE_LOCATION.fullmatch(token)))


def load_events(aborts_path, start_ns, end_ns, *, categories=()):
    """Events with start_ns <= t < end_ns, optionally restricted to categories.

    ``categories`` matches case-insensitively after WindowFilter's
    normalization; an empty selection keeps every category (including
    "Unspecified"). Returns (events, catalog_sha256). Each event is the
    WindowFilter dictionary plus ``pressure_burst`` and
    ``pressure_burst_locations``.
    """
    selector = WindowFilter("test", 0, 0, aborts_path, exclude_before_ns=0, exclude_after_ns=0,
                            pre_abort_ns=0, post_abort_ns=0)
    parsed = selector.load_events()
    digest = selector.catalog_sha256
    records = selector.read_abort_records()
    if selector.catalog_sha256 != digest:
        raise ValueError("Abort catalogue changed while it was being read")
    wanted = {category.strip().casefold() for category in categories}
    events = []
    for event in parsed:
        if not start_ns <= event["timestamp_ns"] < end_ns:
            continue
        if wanted and event["category"].casefold() not in wanted:
            continue
        burst = records[event["source_index"]].get("pressure_burst")
        events.append({**event, "pressure_burst": burst if isinstance(burst, str) else None,
                       "pressure_burst_locations": list(pressure_burst_locations(burst))})
    return events, digest
