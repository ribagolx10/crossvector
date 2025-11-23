"""
Common vector metric constants for all DB adapters.
"""


class VectorMetric:
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


VECTOR_METRIC_MAP = {
    "cosine": VectorMetric.COSINE,
    "dot_product": VectorMetric.DOT_PRODUCT,
    "euclidean": VectorMetric.EUCLIDEAN,
}
