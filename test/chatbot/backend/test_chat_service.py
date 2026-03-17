import pytest
from unittest.mock import MagicMock, AsyncMock
from chatbot.backend.app.services.chat_service import ChatService
from chatbot.backend.app.models.chat import ChatRequest, Message

from chatbot.backend.app.services.context_manager import ContextManager

@pytest.mark.anyio
class TestChatService:
    async def test_chat_completion_flow(self):
        mock_scheduler = MagicMock()
        mock_scheduler.submit = AsyncMock()

        mock_session_dao = MagicMock()
        mock_session_dao.create_session = AsyncMock(return_value=MagicMock(id="sess-1", messages=[]))
        mock_session_dao.add_message = AsyncMock()

        mock_context_manager = MagicMock(spec=ContextManager)

        service = ChatService(mock_scheduler, mock_session_dao, mock_context_manager)

        req = ChatRequest(messages=[Message(role="user", content="Hello")])

        async def mock_submit(job):
            job.future.set_result("Hello World")

        mock_scheduler.submit.side_effect = mock_submit

        resp = await service.chat_completion(req, tenant_id="t1")

        assert resp.choices[0]["message"]["content"] == "Hello World"
        mock_session_dao.add_message.assert_called()
