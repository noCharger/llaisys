import { useState, useEffect, useRef } from 'react';

const useChat = () => {
    const [messages, setMessages] = useState([]);
    const [isThinking, setIsThinking] = useState(false);
    const [error, setError] = useState(null);
    const [apiUrl, setApiUrl] = useState(''); // Empty initially, will fetch
    const sessionId = useRef('session-' + Math.random().toString(36).substr(2, 9) + '-' + Date.now());

    useEffect(() => {
        const fetchConfig = async () => {
            try {
                console.log('Fetching configuration from /config...');
                const res = await fetch('/config');
                const config = await res.json();
                console.log('Configuration loaded:', config);
                if (config.apiUrl) setApiUrl(config.apiUrl);
            } catch (e) {
                console.warn('Using default API URL', e);
            }
        };
        fetchConfig();
    }, []);

    const sendMessage = async (text) => {
        if (!text.trim()) return;

        const userMsg = { role: 'user', content: text };
        setMessages(prev => [...prev, userMsg]);
        setIsThinking(true);
        setError(null);

        try {
            const baseUrl = apiUrl || '/api';
            const cleanUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
            const endpoint = `${cleanUrl}/v1/chat/completions`;

            const history = [...messages, userMsg];

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: 'qwen2',
                    messages: history,
                    stream: true,
                    temperature: 0.7,
                    session_id: sessionId.current
                })
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`API Error: ${response.status} ${response.statusText} - ${errorText}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            setIsThinking(false);
            
            let aiMsg = { role: 'assistant', content: '' };
            setMessages(prev => [...prev, aiMsg]);

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const jsonStr = line.substring(6).trim();
                        if (jsonStr === '[DONE]') break;
                        try {
                            const data = JSON.parse(jsonStr);
                            const delta = data.choices[0].delta.content || '';
                            if (delta) {
                                aiMsg.content += delta;
                                setMessages(prev => {
                                    const newMsgs = [...prev];
                                    newMsgs[newMsgs.length - 1] = { ...aiMsg };
                                    return newMsgs;
                                });
                            }
                        } catch (e) {
                            // ignore parse errors
                        }
                    }
                }
            }

        } catch (err) {
            console.error("Connection Error:", err);
            setIsThinking(false);
            setError(err.message);
            setMessages(prev => [...prev, { role: 'assistant', content: `Connection Error: ${err.message}`, isError: true }]);
        }
    };

    return { messages, sendMessage, isThinking, error };
};

export default useChat;
