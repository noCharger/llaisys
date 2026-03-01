import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.modules["llaisys"] = MagicMock()
sys.modules["llaisys.models"] = MagicMock()

from server import lifespan, state, app, chat_completions
from schemas import ChatCompletionRequest, ChatMessage
import server_pytorch

class TestServerIsolation(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        state.model = None
        state.tokenizer = None
        state.use_pytorch = False
    
    @patch('server.load_model')
    @patch('server_pytorch.load_model')
    async def test_pytorch_flag_enables_pytorch_path(self, mock_pytorch_load, mock_llaisys_load):
        state.use_pytorch = True
        state.model_path = "dummy_path"
        state.device_name = "cpu"
        state.max_steps = 10
        
        async with lifespan(app):
            pass
            
        mock_pytorch_load.assert_called_once_with("dummy_path", "cpu", 10)
        mock_llaisys_load.assert_not_called()

    @patch('server.load_model')
    @patch('server_pytorch.load_model')
    async def test_default_flag_enables_llaisys_path(self, mock_pytorch_load, mock_llaisys_load):
        state.use_pytorch = False
        state.model_path = "dummy_path"
        
        async with lifespan(app):
            pass
            
        mock_llaisys_load.assert_called_once()
        mock_pytorch_load.assert_not_called()

    @patch('server_pytorch.chat_completions')
    async def test_chat_completions_routing_pytorch(self, mock_pytorch_chat):
        state.use_pytorch = True
        mock_pytorch_chat.return_value = "pytorch_response"
        
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        resp = await chat_completions(req)
        
        mock_pytorch_chat.assert_called_once_with(req)
        self.assertEqual(resp, "pytorch_response")

    # If use_pytorch is False and model not loaded, should raise HTTPException
    async def test_chat_completions_routing_llaisys_error(self):
        state.use_pytorch = False
        state.model = None
        
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        
        from fastapi import HTTPException
        with self.assertRaises(HTTPException):
            await chat_completions(req)

if __name__ == '__main__':
    unittest.main()
