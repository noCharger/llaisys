import pytest
from pydantic import ValidationError
from chatbot.backend.schemas import (
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatCompletionResponseChoice
)
from chatbot.backend.context_manager import ContextManager


class TestSchemas:
    def test_parse_valid_response(self):
        raw = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-3.5-turbo",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        
        resp = ChatCompletionResponse(**raw)
        
        assert resp.id == "chatcmpl-123"
        assert len(resp.choices) == 1
        choice = resp.choices[0]
        assert isinstance(choice, ChatCompletionResponseChoice)
        assert choice.message.content == "Hello"

    def test_validation_missing_fields(self):
        with pytest.raises(ValidationError) as exc:
            ChatCompletionResponse(object="chat.completion")
        
        errors = [e["loc"][0] for e in exc.value.errors()]
        assert {"id", "created", "model", "choices"}.issubset(errors)

    def test_validation_invalid_types(self):
        raw = {
            "id": "test",
            "object": "chat.completion",
            "created": "invalid_int",
            "model": "gpt-4",
            "choices": []
        }
        
        with pytest.raises(ValidationError) as exc:
            ChatCompletionResponse(**raw)
        
        assert "created" in str(exc.value)

    def test_parse_streaming_chunk(self):
        raw = {
            "id": "chunk-1",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}]
        }
        
        chunk = ChatCompletionChunk(**raw)
        assert chunk.choices[0].delta.content == "Hi"


class TestContextManager:
    @pytest.fixture
    def cm(self):
        return ContextManager()

    def test_item_lifecycle(self, cm):
        item = cm.add_item("id1", "content", tags=["tag1"], metadata={"k": "v"})
        assert item.id == "id1"
        assert cm.get_item("id1") == item
        
        assert len(cm.list_items()) == 1
        
        assert cm.remove_item("id1")
        assert cm.get_item("id1") is None
        assert not cm.remove_item("id1")

    def test_tag_filtering(self, cm):
        cm.add_item("1", "A", tags=["t1"])
        cm.add_item("2", "B", tags=["t2"])
        cm.add_item("3", "C", tags=["t1", "t2"])

        t1_items = cm.get_items_by_tag("t1")
        assert {i.id for i in t1_items} == {"1", "3"}

    def test_tag_operations(self, cm):
        item = cm.add_item("1", "content", tags=["t1"])
        
        item.add_tag("t2")
        assert item.has_tag("t2")
        assert item.has_all_tags(["t1", "t2"])
        
        item.remove_tag("t1")
        assert not item.has_tag("t1")
        assert item.has_any_tag(["t2", "t3"])
        assert not item.has_any_tag(["t1", "t3"])

    def test_render_filters(self, cm):
        cm.add_item("sys", "System", tags=["sys"])
        cm.add_item("usr", "User", tags=["usr"])
        cm.add_item("priv", "Private", tags=["private"])

        tpl_filter = "{% for i in context_items | filter_by_tag('sys') %}{{ i.content }}{% endfor %}"
        assert cm.render_template(tpl_filter) == "System"

        tpl_exclude = """
        {%- for i in context_items | exclude_by_tag('private') -%}
        {{ i.content }}|
        {%- endfor -%}
        """
        result = cm.render_template(tpl_exclude)
        assert "System" in result
        assert "User" in result
        assert "Private" not in result

    def test_render_error(self, cm):
        with pytest.raises(Exception):
            cm.render_template("{{ invalid_syntax }")
