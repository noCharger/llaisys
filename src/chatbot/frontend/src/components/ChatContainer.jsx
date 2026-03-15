import React, { useEffect, useRef } from 'react';
import Message from './Message';
import ThinkingIndicator from './ThinkingIndicator';

const ChatContainer = ({ messages, isThinking, error, onEdit }) => {
    const endRef = useRef(null);

    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isThinking, error]);

    return (
        <div id="chat-container" className="flex-grow-1 overflow-auto p-3">
            {messages.map((msg, idx) => (
                <Message 
                    key={idx} 
                    index={idx}
                    role={msg.role} 
                    content={msg.content} 
                    onEdit={onEdit}
                />
            ))}
            {isThinking && <ThinkingIndicator />}
            {error && (
                <div className="alert alert-danger mt-3" role="alert">
                    {error}
                </div>
            )}
            <div ref={endRef} />
        </div>
    );
};

export default ChatContainer;
