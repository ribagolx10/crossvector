"""
Tests for backend compilers: to_where accepts Q and dict, rejects invalid types.
"""

import pytest

from crossvector.querydsl.compilers.astradb import AstraDBWhereCompiler
from crossvector.querydsl.compilers.chroma import ChromaWhereCompiler
from crossvector.querydsl.compilers.milvus import MilvusWhereCompiler
from crossvector.querydsl.compilers.pgvector import PgVectorWhereCompiler
from crossvector.querydsl.compilers.utils import normalize_where_input
from crossvector.querydsl.q import Q

BACKENDS = [
    ("pgvector", PgVectorWhereCompiler()),
    ("chroma", ChromaWhereCompiler()),
    ("milvus", MilvusWhereCompiler()),
    ("astradb", AstraDBWhereCompiler()),
]


@pytest.mark.parametrize("name,compiler", BACKENDS)
def test_compiler_to_where_accepts_q(name, compiler):
    q = Q(category="tech", year__gte=2024)
    where = compiler.to_where(q)
    assert isinstance(where, (dict, str))
    if isinstance(where, dict):
        # Chroma wraps multiple fields in $and, others may use flat dict
        if "$and" in where:
            # Chroma-style: {"$and": [{"category": {...}}, {"year": {...}}]}
            assert isinstance(where["$and"], list)
            # Find category in $and list
            cat_filter = next((f for f in where["$and"] if "category" in f), None)
            assert cat_filter is not None
            cat = cat_filter["category"]
            assert isinstance(cat, dict)
            assert cat.get("$eq", cat.get("==")) == "tech"
        else:
            # Flat dict style
            cat = where.get("category")
            assert isinstance(cat, (dict, str))
            if isinstance(cat, dict):
                assert cat.get("$eq", cat.get("==")) == "tech"
            else:
                assert cat == "tech"
    else:
        assert "category" in where
        assert "tech" in where


@pytest.mark.parametrize("name,compiler", BACKENDS)
def test_compiler_to_where_accepts_dict(name, compiler):
    universal = normalize_where_input(Q(category="tech", year__gte=2024))
    where = compiler.to_where(universal)
    assert isinstance(where, (dict, str))
    if isinstance(where, dict):
        # Chroma wraps multiple fields in $and, others may use flat dict
        if "$and" in where:
            # Chroma-style: {"$and": [{"category": {...}}, {"year": {...}}]}
            assert isinstance(where["$and"], list)
            # Find category in $and list
            cat_filter = next((f for f in where["$and"] if "category" in f), None)
            assert cat_filter is not None
            cat = cat_filter["category"]
            assert isinstance(cat, dict)
            assert cat.get("$eq", cat.get("==")) == "tech"
        else:
            # Flat dict style
            cat = where.get("category")
            assert isinstance(cat, (dict, str))
            if isinstance(cat, dict):
                assert cat.get("$eq", cat.get("==")) == "tech"
            else:
                assert cat == "tech"
    else:
        assert "category" in where
        assert "tech" in where


@pytest.mark.parametrize("name,compiler", BACKENDS)
def test_compiler_to_where_rejects_invalid(name, compiler):
    with pytest.raises(TypeError):
        compiler.to_where(["invalid"])
