import React, { useState } from 'react';

const InputArea = ({ text, setText, onSendMessage, disabled }) => {
    const [internalText, setInternalText] = useState('');
    
    const isControlled = text !== undefined && setText !== undefined;
    const currentText = isControlled ? text : internalText;
    const handleChange = isControlled ? (e) => setText(e.target.value) : (e) => setInternalText(e.target.value);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (currentText.trim() && !disabled) {
            onSendMessage(currentText);
            if (!isControlled) setInternalText('');
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
                        value={currentText}
                        onChange={handleChange}
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
