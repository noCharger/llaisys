const express = require('express');
const cors = require('cors');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 6006;
const API_URL = process.env.API_URL || 'http://localhost:6008';

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Proxy API requests (optional, or just use CORS from client)
// For simplicity, we'll let the client talk directly to the backend if CORS is enabled there.
// But a proper frontend server often proxies to avoid CORS issues.

app.get('/config', (req, res) => {    
    let clientApiUrl = API_URL;

    if (API_URL.includes('localhost') || API_URL.includes('127.0.0.1')) {
        const host = req.get('host'); // e.g. 192.168.1.5:3000
        if (host) {
            const hostname = host.split(':')[0];
            try {
                const url = new URL(API_URL);
                url.hostname = hostname;
                clientApiUrl = url.toString();
                // Remove trailing slash if present
                if (clientApiUrl.endsWith('/')) {
                    clientApiUrl = clientApiUrl.slice(0, -1);
                }
            } catch (e) {
                // If API_URL is invalid, keep it as is
            }
        }
    }

    res.json({
        apiUrl: clientApiUrl
    });
});

// SPA Routing: Send index.html for any unknown route
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const server = app.listen(PORT, () => {
    console.log(`Frontend Server running on http://localhost:${PORT}`);
    console.log(`Configured Backend API URL: ${API_URL}`);
});

module.exports = server;
