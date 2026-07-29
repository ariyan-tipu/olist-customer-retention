"""Small logging helper so every module logs in one consistent format."""

from __future__ import annotations

import logging
import os
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger writing to stderr, configured once per process.

    The level can be overridden with the OLIST_LOG_LEVEL environment
    variable (for example DEBUG when digging into a Spark plan).
    """
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
        root = logging.getLogger("olist_retention")
        root.addHandler(handler)
        root.setLevel(os.environ.get("OLIST_LOG_LEVEL", "INFO").upper())
        root.propagate = False
        _CONFIGURED = True
    if name.startswith("olist_retention"):
        return logging.getLogger(name)
    return logging.getLogger(f"olist_retention.{name}")
