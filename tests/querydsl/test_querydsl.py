"""
Unit tests for crossvector.querydsl (Q object, compilers, normalization).
"""

import pytest

from crossvector.querydsl.compilers.utils import normalize_where_input
from crossvector.querydsl.q import Q


def test_q_basic():
    q = Q(field1="value1", field2__gte=10)
    assert isinstance(q, Q)
    d = q.to_dict()
    assert isinstance(d["field1"], dict)
    assert d["field1"].get("$eq") == "value1"
    # range operator encoding likely nested under field name
    assert isinstance(d.get("field2"), dict)
    assert d["field2"].get("$gte") == 10


def test_q_complex_and_or():
    q = Q(category="tech") & (Q(year__gte=2024) | Q(year__lte=2020))
    d = q.to_dict()
    # structure depends on Q implementation; ensure dict returned
    assert isinstance(d, dict)


def test_q_negation():
    q = ~Q(status="inactive")
    d = q.to_dict()
    assert isinstance(d, dict)


def test_normalize_where_input_with_q():
    q = Q(field1="value1", field2__gte=10)
    result = normalize_where_input(q)
    assert isinstance(result, dict)
    assert isinstance(result["field1"], dict)
    assert result["field1"].get("$eq") == "value1"
    assert isinstance(result.get("field2"), dict)
    assert result["field2"].get("$gte") == 10


def test_normalize_where_input_with_dict():
    d = {"field1": "value1", "field2__gte": 10}
    result = normalize_where_input(d)
    assert isinstance(result, dict)
    assert result["field1"] == "value1"
    assert result["field2__gte"] == 10


def test_normalize_where_input_invalid():
    with pytest.raises(TypeError):
        normalize_where_input(["not", "a", "dict", "or", "Q"])
