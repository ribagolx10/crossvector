from crossvector.querydsl.q import Q


class TestCommonDSLMock:
    def test_eq_operator(self, mock_engine, sample_docs):
        res = mock_engine.search(where=Q(category="tech"), limit=10)
        assert len(res) == 2

    def test_ne_operator(self, mock_engine, sample_docs):
        res = mock_engine.search(where=Q(category__ne="tech"), limit=10)
        assert all(d.metadata.get("category") != "tech" for d in res)

    def test_gt_operator(self, mock_engine, sample_docs):
        res = mock_engine.search(where=Q(year__gt=2023), limit=10)
        assert all(d.metadata.get("year") > 2023 for d in res)

    def test_gte_operator(self, mock_engine, sample_docs):
        res = mock_engine.search(where=Q(score__gte=90), limit=10)
        assert all(d.metadata.get("score") >= 90 for d in res)

    def test_lt_operator(self, mock_engine, sample_docs):
        res = mock_engine.search(where=Q(year__lt=2024), limit=10)
        assert all(d.metadata.get("year") < 2024 for d in res)

    def test_lte_operator(self, mock_engine, sample_docs):
        res = mock_engine.search(where=Q(score__lte=88), limit=10)
        assert all(d.metadata.get("score") <= 88 for d in res)

    def test_in_operator(self, mock_engine, sample_docs):
        res = mock_engine.search(where=Q(category__in=["tech", "food"]), limit=10)
        assert all(d.metadata.get("category") in {"tech", "food"} for d in res)

    def test_nin_operator(self, mock_engine, sample_docs):
        res = mock_engine.search(where=Q(category__nin=["travel"]), limit=10)
        assert all(d.metadata.get("category") != "travel" for d in res)

    def test_and_combination(self, mock_engine, sample_docs):
        res = mock_engine.search(where=Q(category="tech") & Q(year__gte=2024), limit=10)
        assert len(res) == 2

    def test_or_combination(self, mock_engine, sample_docs):
        res = mock_engine.search(where=Q(category="tech") | Q(category="food"), limit=10)
        assert len(res) >= 4

    def test_complex_combination(self, mock_engine, sample_docs):
        res = mock_engine.search(where=(Q(category="tech") & Q(score__gte=90)) | Q(category="food"), limit=10)
        assert len(res) >= 3

    def test_nested_metadata(self, mock_engine):
        _ = mock_engine.create(
            {"id": "nested1", "text": "nested", "metadata": {"info": {"lang": "en", "tier": "gold"}}}
        )
        res = mock_engine.search(where=Q(info__lang="en"), limit=10)
        assert any(d.id == "nested1" for d in res)

    def test_metadata_only_search(self, mock_engine, sample_docs):
        res = mock_engine.search(where=Q(category="tech"), limit=10)
        assert len(res) == 2

    def test_universal_dict_format(self, mock_engine):
        where = {"category": {"$eq": "tech"}, "year": {"$gte": 2024}}
        res = mock_engine.search(query="AI", where=where, limit=10)
        assert len(res) == 2

    def test_range_query(self, mock_engine, sample_docs):
        res = mock_engine.search(where=Q(score__gte=80) & Q(score__lte=90), limit=10)
        assert len(res) >= 2
