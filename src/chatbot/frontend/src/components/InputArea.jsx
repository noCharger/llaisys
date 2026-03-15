import React, { useState } from 'react';

const InputArea = ({ onSendMessage, disabled }) => {
    const [text, setText] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (text.trim() && !disabled) {
            onSendMessage(text);
            setText('');
        }
    };

    return (
        <div className="input-area">
            <div className="container">
                <form onSubmit={handleSubmit} className="d-flex gap-2">
                    <input
                        type="text"
                        className="form-control"
                        placeholder="Type your message..."
                        autoComplete="off"
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        disabled={disabled}
                    />
                    <button type="submit" className="btn btn-primary" disabled={disabled}>
                        Send
                    </button>
                </form>
            </div>
        </div>
    );
};

export default InputArea;
