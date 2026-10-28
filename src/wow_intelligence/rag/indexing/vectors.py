import math


def normalized_vector(values, dimensions: int | None = None) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 16384:
        raise ValueError("Embedding must contain 1-16384 numeric dimensions.")
    if dimensions is not None and len(values) != dimensions:
        raise ValueError("Embedding dimensions do not match the index.")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("Embedding contains invalid numeric values.")
    norm = math.hypot(*values)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Embedding must have a finite nonzero norm.")
    return tuple(value / norm for value in values)
