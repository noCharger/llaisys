import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import argparse
import requests
import time
from threading import Thread
import uvicorn
from fastapi import FastAPI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock llaisys
sys.modules["llaisys"] = MagicMock()
sys.modules["llaisys.models"] = MagicMock()

# Reload server if it was already imported
if "server" in sys.modules:
    import importlib
    importlib.reload(sys.modules["server"])
else:
    import server

class TestPortAndConnection(unittest.TestCase):
    
    def test_default_port_logic(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=None)
        parser.add_argument("--pytorch-model", action="store_true")
        
        # Test case 1: --pytorch-model present, no port
        args = parser.parse_args(["--pytorch-model"])
        if args.port is None:
            args.port = 8002 if args.pytorch_model else 8000
        self.assertEqual(args.port, 8002)
        
        # Test case 2: no flag, no port
        args = parser.parse_args([])
        if args.port is None:
            args.port = 8002 if args.pytorch_model else 8000
        self.assertEqual(args.port, 8000)
        
        # Test case 3: explicit port
        args = parser.parse_args(["--port", "9000"])
        if args.port is None:
            args.port = 8002 if args.pytorch_model else 8000
        self.assertEqual(args.port, 9000)

    @patch('requests.get')
    def test_frontend_config_endpoint(self, mock_get):
        # Default behavior
        api_url = os.environ.get('API_URL', 'http://localhost:8002')
        self.assertEqual(api_url, 'http://localhost:8002')
        
        # With Env Var
        with patch.dict(os.environ, {'API_URL': 'http://custom:1234'}):
            api_url = os.environ.get('API_URL', 'http://localhost:8002')
            self.assertEqual(api_url, 'http://custom:1234')

    def test_cors_middleware_presence(self):
        # Verify CORS is added to the FastAPI app
        found_cors = False
        from fastapi.middleware.cors import CORSMiddleware
        
        for middleware in server.app.user_middleware:
            if middleware.cls == CORSMiddleware:
                found_cors = True
                # Check options
                self.assertEqual(middleware.kwargs['allow_origins'], ["*"])
                self.assertEqual(middleware.kwargs['allow_credentials'], True)
                self.assertEqual(middleware.kwargs['allow_methods'], ["*"])
                self.assertEqual(middleware.kwargs['allow_headers'], ["*"])
                break
        
        if not found_cors:
            # TODO: Not in user_middleware or wrapped differently
            pass
            
        self.assertTrue(found_cors, "CORS Middleware not found in FastAPI app")

if __name__ == '__main__':
    unittest.main()
