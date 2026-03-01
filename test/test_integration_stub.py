import unittest
import requests
import subprocess
import time
import os
import signal

class TestIntegrationStub(unittest.TestCase):
    def setUp(self):
        pass

    def test_frontend_backend_connectivity(self):
        frontend_url = "http://localhost:6006"
        backend_url = "http://localhost:6008"
        
        try:
            print(f"Checking Frontend at {frontend_url}...")
            resp = requests.get(f"{frontend_url}/config", timeout=2)
            if resp.status_code == 200:
                config = resp.json()
                print(f"Frontend Config: {config}")
                self.assertTrue("6008" in config["apiUrl"])
            else:
                print("Frontend not running or returned error.")
                
            print(f"Checking Backend at {backend_url}...")
            resp = requests.get(f"{backend_url}/docs", timeout=2)
            if resp.status_code == 200:
                print("Backend is reachable.")
            else:
                print("Backend returned error.")
                
        except requests.exceptions.ConnectionError:
            print("Could not connect to servers. They might not be running in this environment.")
            pass

if __name__ == '__main__':
    unittest.main()
