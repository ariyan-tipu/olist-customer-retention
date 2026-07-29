"""Map Brazilian state codes to the five official regions.

Twenty-seven state dummies would be mostly noise at this sample size, so
models receive the region instead. Portuguese region names are kept, as
they are proper nouns of the data's home country.
"""

from __future__ import annotations

from itertools import chain

from pyspark.sql import Column
from pyspark.sql import functions as F

REGION_BY_STATE: dict[str, str] = {
    "SP": "sudeste",
    "RJ": "sudeste",
    "MG": "sudeste",
    "ES": "sudeste",
    "PR": "sul",
    "SC": "sul",
    "RS": "sul",
    "BA": "nordeste",
    "SE": "nordeste",
    "AL": "nordeste",
    "PE": "nordeste",
    "PB": "nordeste",
    "RN": "nordeste",
    "CE": "nordeste",
    "PI": "nordeste",
    "MA": "nordeste",
    "MT": "centro_oeste",
    "MS": "centro_oeste",
    "GO": "centro_oeste",
    "DF": "centro_oeste",
    "AM": "norte",
    "PA": "norte",
    "RO": "norte",
    "RR": "norte",
    "AC": "norte",
    "AP": "norte",
    "TO": "norte",
}


def region_for(state_column: Column) -> Column:
    """Spark column expression mapping a state code to its region.

    Built per call rather than at import time: PySpark column expressions
    can only be constructed once an active session exists.
    """
    region_map = F.create_map(*[F.lit(x) for x in chain.from_iterable(REGION_BY_STATE.items())])
    return F.coalesce(region_map[state_column], F.lit("desconhecido"))
