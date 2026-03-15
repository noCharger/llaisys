import React from 'react';
import Navbar from './components/Navbar';
import ChatContainer from './components/ChatContainer';
import InputArea from './components/InputArea';
import useChat from './hooks/useChat';

function App() {
  const { messages, sendMessage, isThinking, error } = useChat();

  return (
    <>
      <Navbar />
      <ChatContainer messages={messages} isThinking={isThinking} error={error} />
      <InputArea onSendMessage={sendMessage} disabled={isThinking} />
    </>
  );
}

export default App;
