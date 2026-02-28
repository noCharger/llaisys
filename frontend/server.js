const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const API_URL = process.env.API_URL || 'http://localhost:8001';

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Proxy API requests (optional, or just use CORS from client)
// For simplicity, we'll let the client talk directly to the backend if CORS is enabled there.
// But a proper frontend server often proxies to avoid CORS issues.

app.get('/config', (req, res) => {
    res.json({
        apiUrl: API_URL
    });
});

// SPA Routing: Send index.html for any unknown route
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`Frontend Server running on http://localhost:${PORT}`);
    console.log(`Configured Backend API URL: ${API_URL}`);
});
