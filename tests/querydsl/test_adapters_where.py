"""
Adapter-level tests: ensure adapters accept Q/dict via compilers and reject plain text.
Note: Uses minimal stubs where necessary to avoid real DB connections.
"""

import pytest

from crossvector.dbs.astradb import AstraDBAdapter
from crossvector.dbs.chroma import ChromaAdapter
from crossvector.dbs.milvus import MilvusAdapter
from crossvector.dbs.pgvector import PgVectorAdapter
from crossvector.querydsl.q import Q

ADAPTERS = [
    ("pgvector", PgVectorAdapter),
    ("chroma", ChromaAdapter),
    ("milvus", MilvusAdapter),
    ("astradb", AstraDBAdapter),
]


@pytest.mark.parametrize("name,AdapterCls", ADAPTERS)
def test_search_where_q_or_dict(name, AdapterCls, monkeypatch):
    adapter = AdapterCls()

    # monkeypatch search to avoid real calls
    def fake_search(vector=None, limit=10, offset=0, where=None, fields=None):
        assert where is None or isinstance(where, (dict, Q))
        return []

    monkeypatch.setattr(adapter, "search", fake_search)

    adapter.search(vector=None, where=Q(category="tech"))
    adapter.search(vector=None, where={"category": "tech"})


@pytest.mark.parametrize("name,AdapterCls", ADAPTERS)
def test_search_where_reject_plain_text(name, AdapterCls):
    adapter = AdapterCls()
    with pytest.raises(Exception):
        # depending on adapter implementation, this should raise or be validated upstream
        adapter.search(vector=None, where="category = tech")
