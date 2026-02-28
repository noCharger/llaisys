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

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text) return;

        // User Message
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
                    temperature: 0.7
                })
            });

            if (!response.ok) throw new Error('API Error');

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const jsonStr = line.substring(6);
                        if (jsonStr === '[DONE]') break;
                        try {
                            const data = JSON.parse(jsonStr);
                            const delta = data.choices[0].delta.content || '';
                            aiText += delta;
                            
                            // Initialize message div on first content
                            if (!aiMsgDiv) {
                                thinkingIndicator.remove();
                                aiMsgDiv = appendMessage('assistant', '');
                            }
                            
                            aiMsgDiv.textContent = aiText;
                            chatContainer.scrollTop = chatContainer.scrollHeight;
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
            appendMessage('assistant', `Error: ${err.message}`);
        }
    });
});
