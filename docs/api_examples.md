# API Interaction Examples

This document provides examples of how to interact with the chat service running at `http://127.0.0.1:6008`.

## 1. Using cURL (Terminal)

### Non-Streaming Request
This waits for the full response before returning.

```bash
curl http://127.0.0.1:6008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2",
    "messages": [
      {"role": "user", "content": "Hello, who are you?"}
    ],
    "stream": false,
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Streaming Request
This receives tokens as they are generated (Server-Sent Events).

```bash
curl -N http://127.0.0.1:6008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2",
    "messages": [
      {"role": "user", "content": "Write a short poem about coding."}
    ],
    "stream": true,
    "max_tokens": 100
  }'
```

## 2. Using Python (`requests`)

Save this as `test_chat.py` and run it with `python test_chat.py`.

```python
import requests
import json

API_URL = "http://127.0.0.1:6008/v1/chat/completions"

def chat_non_streaming():
    print("--- Non-Streaming Request ---")
    payload = {
        "model": "qwen2",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": False,
        "max_tokens": 50
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        print("Response:", result["choices"][0]["message"]["content"])
        if "X-E2E-Latency-Ms" in response.headers:
            print(f"Latency: {response.headers['X-E2E-Latency-Ms']}ms")
    except Exception as e:
        print(f"Error: {e}")

def chat_streaming():
    print("\n--- Streaming Request ---")
    payload = {
        "model": "qwen2",
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        "stream": True,
        "max_tokens": 50
    }
    
    try:
        response = requests.post(API_URL, json=payload, stream=True)
        response.raise_for_status()
        
        print("Response: ", end="", flush=True)
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data_str = line[6:] # Remove 'data: ' prefix
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0]["delta"]
                        if "content" in delta:
                            print(delta["content"], end="", flush=True)
                    except json.JSONDecodeError:
                        pass
        print() # Newline
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    chat_non_streaming()
    chat_streaming()
```

## 3. Using OpenAI Python Client

Since the API is OpenAI-compatible, you can use the official `openai` library.

Install library:
```bash
pip install openai
```

Script (`test_openai.py`):
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:6008/v1",
    api_key="not-needed" # API key is required by client but ignored by server
)

# Streaming
print("--- Streaming ---")
stream = client.chat.completions.create(
    model="qwen2",
    messages=[{"role": "user", "content": "Tell me a joke."}],
    stream=True,
    max_tokens=100
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
print()
```
