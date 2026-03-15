import React from 'react';
import ReactMarkdown from 'react-markdown';
import { parseMessage } from '../utils/messageParser';

const Message = ({ role, content, index, onEdit }) => {
    const safeContent = content || '';

    const renderContent = () => {
        if (role !== 'user' && safeContent) {
            const segments = parseMessage(safeContent);

            return segments.map((segment, idx) => {
                if (segment.type === 'think') {
                    return (
                        <details key={idx} className="think-block mb-2">
                            <summary className="text-muted small cursor-pointer">Thought Process</summary>
                            <div className="text-muted small border-start ps-2 mt-1">
                                {segment.content}
                            </div>
                        </details>
                    );
                }
                
                return (
                    <div key={idx} className="markdown-content">
                        <ReactMarkdown>{segment.content}</ReactMarkdown>
                    </div>
                );
            });
        }
        return <div className="markdown-content"><ReactMarkdown>{safeContent}</ReactMarkdown></div>;
    };

    return (
        <div className={`message ${role === 'user' ? 'user-message' : 'ai-message'} position-relative group mb-3 p-3 rounded`}>
            {role === 'user' && onEdit && (
                <button 
                    className="edit-btn btn btn-sm btn-link p-0 position-absolute d-flex align-items-center justify-content-center"
                    style={{ 
                        top: '5px', 
                        right: '10px', 
                        textDecoration: 'none', 
                        opacity: 1,
                        width: '32px',
                        height: '32px',
                        fontSize: '1.2rem',
                        backgroundColor: '#ffffff',
                        color: '#007bff',
                        borderRadius: '50%',
                        boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
                    }}
                    onClick={() => onEdit(index, safeContent)}
                    title="Edit message"
                    aria-label="Edit message"
                >
                    ✎
                </button>
            )}
            {renderContent()}
        </div>
    );
};

export default Message;
