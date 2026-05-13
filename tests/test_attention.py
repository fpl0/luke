"""Tests for luke.attention — persistent foreground commitments (L2)."""

from __future__ import annotations

from typing import Any

from luke import attention


class TestPin:
    """Tests for attention.pin()."""

    def test_pin_inserts_and_returns_id(self, test_db: Any) -> None:
        item_id = attention.pin("100", "track the Fanatics interview prep")
        assert item_id > 0

    def test_pin_persists_row(self, test_db: Any) -> None:
        item_id = attention.pin("100", "watch for Naiara's email", origin="luke")
        rows = attention.list_attention("100")
        assert len(rows) == 1
        assert rows[0]["id"] == item_id
        assert rows[0]["content"] == "watch for Naiara's email"
        assert rows[0]["origin"] == "luke"

    def test_pin_default_origin_is_luke(self, test_db: Any) -> None:
        attention.pin("100", "test item")
        rows = attention.list_attention("100")
        assert rows[0]["origin"] == "luke"

    def test_pin_stores_related_id(self, test_db: Any) -> None:
        attention.pin("100", "item with link", related_id="goal-foo")
        rows = attention.list_attention("100")
        assert rows[0]["related_id"] == "goal-foo"

    def test_pin_returns_distinct_ids(self, test_db: Any) -> None:
        id1 = attention.pin("100", "first")
        id2 = attention.pin("100", "second")
        assert id1 != id2


class TestListAttention:
    """Tests for attention.list_attention()."""

    def test_returns_empty_list_initially(self, test_db: Any) -> None:
        assert attention.list_attention("100") == []

    def test_returns_rows_for_chat(self, test_db: Any) -> None:
        attention.pin("100", "a")
        attention.pin("100", "b")
        rows = attention.list_attention("100")
        assert len(rows) == 2
        contents = {r["content"] for r in rows}
        assert contents == {"a", "b"}

    def test_filters_by_chat(self, test_db: Any) -> None:
        attention.pin("100", "for chat 100")
        attention.pin("200", "for chat 200")
        rows_100 = attention.list_attention("100")
        rows_200 = attention.list_attention("200")
        assert len(rows_100) == 1
        assert len(rows_200) == 1
        assert rows_100[0]["content"] == "for chat 100"
        assert rows_200[0]["content"] == "for chat 200"

    def test_respects_limit(self, test_db: Any) -> None:
        for i in range(5):
            attention.pin("100", f"item-{i}")
        rows = attention.list_attention("100", limit=2)
        assert len(rows) == 2


class TestUnpin:
    """Tests for attention.unpin()."""

    def test_unpin_removes_by_id(self, test_db: Any) -> None:
        item_id = attention.pin("100", "to remove")
        assert attention.unpin("100", item_id) is True
        assert attention.list_attention("100") == []

    def test_unpin_returns_false_for_unknown_id(self, test_db: Any) -> None:
        assert attention.unpin("100", 9999) is False

    def test_unpin_scoped_to_chat(self, test_db: Any) -> None:
        """Unpin must not remove an item from a different chat."""
        item_id = attention.pin("100", "chat 100 item")
        # Wrong chat — should not remove
        assert attention.unpin("200", item_id) is False
        assert len(attention.list_attention("100")) == 1
        # Correct chat — removes
        assert attention.unpin("100", item_id) is True
        assert attention.list_attention("100") == []


class TestBuildAttentionBlock:
    """Tests for attention.build_attention_block()."""

    def test_returns_none_when_empty(self, test_db: Any) -> None:
        assert attention.build_attention_block("100") is None

    def test_formats_items(self, test_db: Any) -> None:
        id1 = attention.pin("100", "track Fanatics prep", origin="luke")
        id2 = attention.pin("100", "watch Naiara's email", origin="user")
        block = attention.build_attention_block("100")
        assert block is not None
        assert "<active-attention>" in block
        assert "</active-attention>" in block
        assert "Things you said matter" in block
        assert "unpin_attention" in block
        assert f"#{id1}" in block
        assert f"#{id2}" in block
        assert "track Fanatics prep" in block
        assert "watch Naiara's email" in block
        assert "luke" in block
        assert "user" in block

    def test_only_chat_items_appear(self, test_db: Any) -> None:
        attention.pin("100", "for 100")
        attention.pin("200", "for 200")
        block = attention.build_attention_block("100")
        assert block is not None
        assert "for 100" in block
        assert "for 200" not in block
