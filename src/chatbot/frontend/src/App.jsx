import React, { useState } from 'react';
import Navbar from './components/Navbar';
import ChatContainer from './components/ChatContainer';
import InputArea from './components/InputArea';
import useChat from './hooks/useChat';

function App() {
  const [apiKey, setApiKey] = useState('');
  const { messages, sendMessage, isThinking, error, setMessages } = useChat(apiKey);
  const [inputText, setInputText] = useState('');
  const [showSettings, setShowSettings] = useState(false);

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
      <Navbar onSettingsClick={() => setShowSettings(!showSettings)} />
      {showSettings && (
        <div style={{ padding: '10px', background: '#f0f0f0', borderBottom: '1px solid #ccc' }}>
          <label style={{ marginRight: '10px' }}>API Key:</label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="sk-..."
            style={{ padding: '5px', width: '300px' }}
          />
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
