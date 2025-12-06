"""Additional Q class tests to improve coverage."""

from crossvector.querydsl.q import Q


class TestQStringRepresentations:
    """Test __str__ and __repr__ methods."""

    def test_q_str_simple(self):
        """Test __str__ for simple Q node."""
        q = Q(name="test")
        result = str(q)
        assert "name" in result
        assert "$eq" in result

    def test_q_repr_simple(self):
        """Test __repr__ for simple Q node."""
        q = Q(name="test")
        result = repr(q)
        assert "Q:" in result
        assert "name" in result

    def test_q_str_with_and(self):
        """Test __str__ for combined Q nodes."""
        q = Q(name="test") & Q(age__gte=18)
        result = str(q)
        assert "$and" in result
        assert "name" in result

    def test_q_repr_with_or(self):
        """Test __repr__ for OR combined nodes."""
        q = Q(status="active") | Q(status="pending")
        result = repr(q)
        assert "Q:" in result
        assert "$or" in result

    def test_q_str_negated(self):
        """Test __str__ for negated Q node."""
        q = ~Q(deleted="true")
        result = str(q)
        assert "$not" in result
        assert "deleted" in result

    def test_q_repr_complex(self):
        """Test __repr__ for complex nested Q."""
        q = (~Q(archived="true")) & (Q(category="tech") | Q(category="science"))
        result = repr(q)
        assert "Q:" in result
        assert "$not" in result or "$and" in result


class TestQNegation:
    """Test negation operator."""

    def test_negate_simple(self):
        """Test negating a simple Q node."""
        q = Q(active=True)
        negated = ~q
        assert negated.negate is True
        assert q.negate is False  # original unchanged

    def test_double_negate(self):
        """Test double negation."""
        q = Q(status="active")
        double_neg = ~~q
        assert double_neg.negate is False

    def test_negate_preserves_filters(self):
        """Test that negation preserves filters."""
        q = Q(score__gte=80, category="tech")
        negated = ~q
        assert negated.filters == q.filters
        assert negated.negate is True

    def test_negate_with_children(self):
        """Test negating a combined node."""
        q = Q(a=1) & Q(b=2)
        negated = ~q
        assert negated.negate is True
        assert len(negated.children) == 2

    def test_negate_to_dict(self):
        """Test negated node's dict representation."""
        q = ~Q(status="banned")
        result = q.to_dict()
        assert "$not" in result
        assert result["$not"]["status"]["$eq"] == "banned"


class TestQBackendCompilers:
    """Test backend compiler selection and compilation."""

    def test_milvus_backend_selection(self):
        """Test Milvus backend compiler selection."""
        q = Q(age__gte=18)
        compiler = q._get_where_compiler("milvus")
        assert compiler is not None

    def test_chromadb_backend_selection(self):
        """Test Chroma backend compiler selection."""
        q = Q(category="tech")
        compiler = q._get_where_compiler("chromadb")
        assert compiler is not None

    def test_astradb_backend_selection(self):
        """Test AstraDB backend compiler selection."""
        q = Q(id="doc1")
        compiler = q._get_where_compiler("astradb")
        assert compiler is not None

    def test_pgvector_backend_selection(self):
        """Test PGVector backend compiler selection."""
        q = Q(tag__in=["a", "b"])
        compiler = q._get_where_compiler("pgvector")
        assert compiler is not None

    def test_unknown_backend_selection(self):
        """Test unknown backend returns None."""
        q = Q(field="value")
        compiler = q._get_where_compiler("unknown_backend")
        assert compiler is None

    def test_to_where_generic(self):
        """Test to_where with generic backend."""
        q = Q(name="test", score__gte=50)
        result = q.to_where("generic")
        assert isinstance(result, dict)
        assert "name" in result
        assert "score" in result

    def test_to_where_milvus(self):
        """Test to_where with Milvus backend."""
        q = Q(age__lt=30)
        result = q.to_where("milvus")
        # Milvus returns a string expression
        assert isinstance(result, str)

    def test_to_where_chromadb(self):
        """Test to_where with Chroma backend."""
        q = Q(tag="important")
        result = q.to_where("chromadb")
        # Chroma can return dict or string depending on complexity
        assert result is not None

    def test_to_where_astradb(self):
        """Test to_where with AstraDB backend."""
        q = Q(status="active")
        result = q.to_where("astradb")
        assert result is not None

    def test_to_where_pgvector(self):
        """Test to_where with PGVector backend."""
        q = Q(price__gte=100)
        result = q.to_where("pgvector")
        assert result is not None

    def test_to_where_complex_query_milvus(self):
        """Test complex query with Milvus."""
        q = (Q(category="tech") & Q(score__gte=85)) | Q(featured=True)
        result = q.to_where("milvus")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_to_where_with_negation_pgvector(self):
        """Test negated query compilation for PGVector."""
        q = ~Q(archived="true")
        result = q.to_where("pgvector")
        assert result is not None

    def test_to_expr_generic(self):
        """Test to_expr with generic backend."""
        q = Q(field="value")
        result = q.to_expr("generic")
        assert isinstance(result, str)
        assert "field" in result

    def test_to_expr_milvus(self):
        """Test to_expr with Milvus backend."""
        q = Q(level__gte=5)
        result = q.to_expr("milvus")
        assert isinstance(result, str)

    def test_to_expr_chromadb(self):
        """Test to_expr with Chroma backend."""
        q = Q(status__ne="deleted")
        result = q.to_expr("chromadb")
        assert isinstance(result, str)

    def test_to_expr_astradb(self):
        """Test to_expr with AstraDB backend."""
        q = Q(region__in=["US", "EU"])
        result = q.to_expr("astradb")
        assert isinstance(result, str)

    def test_to_expr_pgvector(self):
        """Test to_expr with PGVector backend."""
        q = Q(confidence__lte=0.5)
        result = q.to_expr("pgvector")
        assert isinstance(result, str)

    def test_to_expr_complex(self):
        """Test to_expr with complex nested query."""
        q = (Q(type="A") | Q(type="B")) & Q(status__ne="inactive")
        result = q.to_expr("milvus")
        assert isinstance(result, str)


class TestQLeafToDictEdgeCases:
    """Test edge cases in _leaf_to_dict conversion."""

    def test_implicit_eq_operator(self):
        """Test implicit $eq when no operator specified."""
        q = Q(status="active")
        result = q._leaf_to_dict()
        assert result["status"]["$eq"] == "active"

    def test_double_underscore_field_name(self):
        """Test nested field names with dots converted from underscores."""
        q = Q(user__profile="verified")
        result = q._leaf_to_dict()
        # Should have converted __ to . in field name
        assert "user.profile" in result
        assert result["user.profile"]["$eq"] == "verified"

    def test_triple_underscore_field_and_operator(self):
        """Test triple underscore with operator."""
        q = Q(user__profile__status__eq="active")
        result = q._leaf_to_dict()
        assert "user.profile.status" in result
        assert result["user.profile.status"]["$eq"] == "active"

    def test_invalid_operator_treated_as_field(self):
        """Test invalid operator falls back to field name."""
        q = Q(field__invalid="value")
        result = q._leaf_to_dict()
        # Invalid operator, so whole key becomes field
        assert "field.invalid" in result
        assert result["field.invalid"]["$eq"] == "value"

    def test_all_valid_operators(self):
        """Test all valid operators are converted."""
        operators = ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"]
        for op in operators:
            q = Q(**{f"field__{op}": "value"})
            result = q._leaf_to_dict()
            op_symbol = f"${op}"
            assert result["field"][op_symbol] == "value"

    def test_multiple_filters_same_field_different_ops(self):
        """Test multiple operators on same field."""
        q = Q(age__gte=18, **{"age__lte": 65})
        result = q._leaf_to_dict()
        assert result["age"]["$gte"] == 18
        assert result["age"]["$lte"] == 65

    def test_list_value_with_in_operator(self):
        """Test list values with in operator."""
        q = Q(category__in=["tech", "science", "nature"])
        result = q._leaf_to_dict()
        assert result["category"]["$in"] == ["tech", "science", "nature"]

    def test_none_value(self):
        """Test None as a value."""
        q = Q(field=None)
        result = q._leaf_to_dict()
        assert result["field"]["$eq"] is None

    def test_numeric_values(self):
        """Test numeric values."""
        q = Q(count__gt=0, price__lte=99.99)
        result = q._leaf_to_dict()
        assert result["count"]["$gt"] == 0
        assert result["price"]["$lte"] == 99.99

    def test_boolean_values(self):
        """Test boolean values."""
        q = Q(active=True, archived=False)
        result = q._leaf_to_dict()
        assert result["active"]["$eq"] is True
        assert result["archived"]["$eq"] is False


class TestQComplexCombinations:
    """Test complex nested combinations."""

    def test_and_of_ors(self):
        """Test (Q | Q) & (Q | Q) structure."""
        q = (Q(a=1) | Q(a=2)) & (Q(b="x") | Q(b="y"))
        result = q.to_dict()
        assert result["$and"] is not None
        assert len(result["$and"]) == 2

    def test_or_of_ands(self):
        """Test (Q & Q) | (Q & Q) structure."""
        q = (Q(x=10) & Q(y=20)) | (Q(x=30) & Q(y=40))
        result = q.to_dict()
        assert result["$or"] is not None
        assert len(result["$or"]) == 2

    def test_deep_nesting(self):
        """Test deeply nested structure."""
        q = ((Q(a=1) & Q(b=2)) | (Q(c=3) & Q(d=4))) & Q(e=5)
        result = q.to_dict()
        assert "$and" in result or "$or" in result

    def test_negated_combination(self):
        """Test negating a combined expression."""
        q = ~((Q(status="active") | Q(status="pending")) & Q(priority__gte=5))
        result = q.to_dict()
        assert "$not" in result
        assert "$and" in result["$not"] or "$or" in result["$not"]

    def test_operators_with_combinations(self):
        """Test operators mixed with combinations."""
        q = (Q(age__gte=18) & Q(age__lte=65)) | (Q(special_access=True))
        result = q.to_dict()
        assert "$or" in result

    def test_many_or_conditions(self):
        """Test many OR conditions."""
        q = Q(status="A") | Q(status="B") | Q(status="C") | Q(status="D")
        result = q.to_dict()
        assert "$or" in result

    def test_mixed_operators_and_combinations(self):
        """Test mix of operators and boolean logic."""
        q = (Q(score__gt=50) & Q(score__lt=100)) & (Q(category__in=["X", "Y"]) | ~Q(excluded="true"))
        result = q.to_dict()
        assert isinstance(result, dict)

    def test_negation_cascading(self):
        """Test multiple negations in sequence."""
        q1 = Q(field="value")
        q2 = ~q1
        q3 = ~q2
        q4 = ~q3
        assert q4.negate is True
        assert q3.negate is False
        assert q2.negate is True
        assert q1.negate is False

    def test_combination_of_negated_filters(self):
        """Test combining negated filters."""
        q = ~Q(deleted="true") & ~Q(archived="true")
        result = q.to_dict()
        assert "$and" in result
        assert result["$and"][0]["$not"] is not None

    def test_to_dict_with_empty_filters(self):
        """Test Q node with no filters or children."""
        q = Q()
        result = q.to_dict()
        assert isinstance(result, dict)
