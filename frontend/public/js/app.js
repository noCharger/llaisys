document.addEventListener('DOMContentLoaded', async () => {
    const chatContainer = document.getElementById('chat-container');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');

    let API_URL = 'http://localhost:8000'; // Default

    // Load Config
    try {
        const res = await fetch('/config');
        const config = await res.json();
        if (config.apiUrl) API_URL = config.apiUrl;
    } catch (e) {
        console.warn('Using default API URL', e);
    }

    // Message History
    const history = [];

    function appendMessage(role, text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role === 'user' ? 'user-message' : 'ai-message'}`;
        msgDiv.textContent = text;
        chatContainer.appendChild(msgDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return msgDiv;
    }

    function createThinkingIndicator() {
        const div = document.createElement('div');
        div.className = 'thinking-indicator';
        div.innerHTML = 'Thinking<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
        chatContainer.appendChild(div);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return div;
    }

    // Generate a unique session ID
    const sessionId = 'session-' + Math.random().toString(36).substr(2, 9) + '-' + Date.now();

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text) return;

        appendMessage('user', text);
        userInput.value = '';
        history.push({ role: 'user', content: text });

        // Show Thinking Indicator
        const thinkingIndicator = createThinkingIndicator();
        let aiMsgDiv = null;
        let aiText = '';

        try {
            const response = await fetch(`${API_URL}/v1/chat/completions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: 'qwen2',
                    messages: history,
                    stream: true,
                    temperature: 0.7,
                    session_id: sessionId
                })
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`API Error: ${response.status} ${response.statusText} - ${errorText}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            // Remove thinking indicator immediately on first successful response
            thinkingIndicator.remove();
            aiMsgDiv = appendMessage('assistant', '');

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
                                aiText += delta;
                                aiMsgDiv.textContent = aiText;
                                chatContainer.scrollTop = chatContainer.scrollHeight;
                            }
                        } catch (e) {
                            // ignore parse errors for partial chunks
                        }
                    }
                }
            }
            
            if (aiText) {
                history.push({ role: 'assistant', content: aiText });
            }

        } catch (err) {
            thinkingIndicator.remove();
            console.error("Connection Error:", err);
            appendMessage('assistant', `Connection Error: Failed to reach backend at ${API_URL}. Please ensure the server is running on port 8002 (or configured port) and bound to 0.0.0.0 or localhost. Details: ${err.message}`);
        }
    });
});
