import React from 'react';

const Navbar = ({ onSettingsClick }) => {
    return (
        <nav className="navbar navbar-dark bg-dark">
            <div className="container-fluid" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1rem' }}>
                <span className="navbar-brand mb-0 h1" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>LLAISYS AI Chat</span>
                <button
                    onClick={onSettingsClick}
                    style={{
                        background: 'transparent',
                        border: '1px solid #ccc',
                        borderRadius: '4px',
                        padding: '5px 10px',
                        cursor: 'pointer',
                        color: 'white'
                    }}
                >
                    Settings
                </button>
            </div>
        </nav>
    );
};

export default Navbar;
