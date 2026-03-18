import React, { useState } from 'react';
import Navbar from './components/Navbar';
import ChatContainer from './components/ChatContainer';
import InputArea from './components/InputArea';
import AdminPanel from './components/AdminPanel';
import useChat from './hooks/useChat';

function App() {
  const initialApiKey = localStorage.getItem('llaisys_api_key') || '';

  const [apiKeyInput, setApiKeyInput] = useState(initialApiKey);
  const [activeApiKey, setActiveApiKey] = useState(initialApiKey);
  const [isSaved, setIsSaved] = useState(false);

  const { messages, sendMessage, isThinking, error, setMessages } = useChat(activeApiKey);

  const [inputText, setInputText] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [showAdmin, setShowAdmin] = useState(false);

  const handleApiKeyChange = (e) => {
    setApiKeyInput(e.target.value);
    setIsSaved(false);
  };

  const handleSaveApiKey = () => {
    const trimmedKey = apiKeyInput.trim();

    setApiKeyInput(trimmedKey);
    setActiveApiKey(trimmedKey);
    localStorage.setItem('llaisys_api_key', trimmedKey);

    setIsSaved(true);
    setTimeout(() => setIsSaved(false), 3000);
  };

  const handleEdit = (index, content) => {
    setInputText(content);
    setMessages(prev => prev.slice(0, index));
  };

  const handleSendMessage = (text) => {
    sendMessage(text);
    setInputText('');
  };

  return (
    <>
      <Navbar
        onSettingsClick={() => setShowSettings(!showSettings)}
        onAdminClick={() => setShowAdmin(!showAdmin)}
      />

      {showAdmin && <AdminPanel />}

      {showSettings && (
        <div style={{ padding: '15px', background: '#f8f9fa', borderBottom: '1px solid #dee2e6', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <label style={{ fontWeight: 'bold' }}>API Key (Control Plane):</label>
          <input
            type="password"
            value={apiKeyInput}
            onChange={handleApiKeyChange}
            placeholder="Enter sk-... to authenticate via TenantManager"
            style={{ padding: '8px', width: '350px', borderRadius: '4px', border: '1px solid #ced4da' }}
          />
          <button
            onClick={handleSaveApiKey}
            style={{ padding: '8px 15px', background: '#007bff', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
          >
            Save
          </button>
          {isSaved && (
            <span style={{ color: 'green', fontSize: '0.9em' }}>✓ Saved</span>
          )}
          <small style={{ color: '#6c757d', marginLeft: 'auto' }}>
            Required for rate limiting and tenant isolation.
          </small>
        </div>
      )}

      <ChatContainer
        messages={messages}
        isThinking={isThinking}
        error={error}
        onEdit={handleEdit}
      />

      <InputArea
        text={inputText}
        setText={setInputText}
        onSendMessage={handleSendMessage}
        disabled={isThinking}
      />
    </>
  );
}

export default App;