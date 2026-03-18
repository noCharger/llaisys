import { useState, useEffect, useRef, useCallback } from 'react';

const useChat = (apiKey) => {
  const [messages, setMessages] = useState([]);
  const [isThinking, setIsThinking] = useState(false);
  const [error, setError] = useState(null);
  const [apiUrl, setApiUrl] = useState('');
  const sessionId = useRef(
    'session-' + Math.random().toString(36).substr(2, 9) + '-' + Date.now()
  );

  const messagesRef = useRef(messages);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const res = await fetch('/config');
        if (res.ok) {
          const config = await res.json();
          if (config.apiUrl) setApiUrl(config.apiUrl);
        }
      } catch (e) {
        console.warn('Using default API URL', e);
      }
    };
    fetchConfig();
  }, []);

  const sendMessage = useCallback(async (text) => {
    if (!text.trim()) return;

    const userMsg = { role: 'user', content: text };

    setMessages(prev => [...prev, userMsg]);
    setIsThinking(true);
    setError(null);

    try {
      const endpoint = apiUrl
        ? `${apiUrl}/v1/chat/completions`
        : '/v1/chat/completions';

      const history = [...messagesRef.current, userMsg];

      const headers = { 'Content-Type': 'application/json' };
      if (apiKey) {
        headers['Authorization'] = `Bearer ${apiKey}`;
      } else {
        headers['x-tenant-id'] = 'dev-tenant';
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model: 'qwen2',
          messages: history,
          stream: true,
          temperature: 0.7,
          session_id: sessionId.current,
          use_template: true
        })
      });

      if (!response.ok) {
        let errorMsg = `${response.status} ${response.statusText}`;
        try {
          const errorJson = await response.json();
          if (errorJson.detail) {
            errorMsg += ` - ${errorJson.detail}`;
          }
        } catch (e) {
          const errorText = await response.text();
          if (errorText) errorMsg += ` - ${errorText}`;
        }
        throw new Error(`API Error: ${errorMsg}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      setIsThinking(false);

      let aiMsg = { role: 'assistant', content: '' };
      setMessages(prev => [...prev, aiMsg]);

      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let deltaContent = '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonStr = line.substring(6).trim();
            if (jsonStr === '[DONE]') break;

            try {
              const data = JSON.parse(jsonStr);
              const delta = data.choices?.[0]?.delta?.content || '';
              if (delta) deltaContent += delta;
            } catch (e) {
              console.warn('Failed to parse SSE JSON:', jsonStr);
            }
          }
        }

        if (deltaContent) {
          aiMsg.content += deltaContent;
          setMessages(prev => {
            const newMsgs = [...prev];
            if (
              newMsgs.length > 0 &&
              newMsgs[newMsgs.length - 1].role === 'assistant'
            ) {
              newMsgs[newMsgs.length - 1] = { ...aiMsg };
            } else {
              newMsgs.push({ ...aiMsg });
            }
            return newMsgs;
          });
        }
      }
    } catch (err) {
      console.error('Connection Error:', err);
      setIsThinking(false);
      setError(err.message);
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: `Connection Error: ${err.message}`,
          isError: true
        }
      ]);
    }
  }, [apiKey, apiUrl]);

  return { messages, sendMessage, isThinking, error, setMessages };
};

export default useChat;