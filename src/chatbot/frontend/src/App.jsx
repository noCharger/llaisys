import React, { useState } from 'react';
import Navbar from './components/Navbar';
import ChatContainer from './components/ChatContainer';
import InputArea from './components/InputArea';
import useChat from './hooks/useChat';

function App() {
  const { messages, sendMessage, isThinking, error, setMessages } = useChat();
  const [inputText, setInputText] = useState('');

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
      <Navbar />
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
