const express = require('express');
const cors = require('cors');
const path = require('path');
const helmet = require('helmet');
const fs = require('fs');
const https = require('https');
const http = require('http');
const { execSync } = require('child_process');
const { createProxyMiddleware } = require('http-proxy-middleware');
require('dotenv').config();

const app = express();
const PORT = parseInt(process.env.PORT || 6006, 10);
const HTTPS_PORT = process.env.NODE_ENV === 'test' ? 0 : PORT + 1;
const INTERNAL_API_URL = process.env.INTERNAL_API_URL || 'https://localhost:6009';
const USE_PROXY = process.env.NO_HTTPS === 'true';

const certDir = path.join(__dirname, '..', 'certs');
const keyPath = path.join(certDir, 'key.pem');
const certPath = path.join(certDir, 'cert.pem');

let sslConfigured = false;
try {
    if (process.env.NO_HTTPS !== 'true') {
        if (!fs.existsSync(keyPath) || !fs.existsSync(certPath)) {
            fs.mkdirSync(certDir, { recursive: true });
            execSync(`openssl req -x509 -newkey rsa:4096 -keyout "${keyPath}" -out "${certPath}" -days 365 -nodes -subj "/CN=localhost"`, { stdio: 'pipe' });
        }
        if (fs.existsSync(keyPath) && fs.existsSync(certPath)) {
            sslConfigured = true;
        }
    } else {
        console.log('HTTPS disabled via NO_HTTPS environment variable.');
    }
} catch (err) {
    console.error('Failed to auto-provision certificates. Ensure openssl is installed.', err.message);
}

app.use(helmet({
    hsts: sslConfigured ? { maxAge: 31536000, includeSubDomains: true } : false,
    frameguard: { action: 'deny' },
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            scriptSrc: ["'self'", "'unsafe-inline'"],
            styleSrc: ["'self'", "'unsafe-inline'"],
            connectSrc: ["'self'", USE_PROXY ? "'self'" : INTERNAL_API_URL]
        }
    }
}));

app.use(cors());

app.use((req, res, next) => {
    if (process.env.NODE_ENV !== 'test') {
        console.log(`[Server] Received ${req.method} request for ${req.url}`);
    }
    next();
});

if (USE_PROXY) {
    console.log(`[Server] Enabling API Proxy to ${INTERNAL_API_URL}`);
    
    // Mount proxy at /api/v1 if the frontend calls it there
    app.use('/api/v1', createProxyMiddleware({
        target: INTERNAL_API_URL,
        changeOrigin: true,
        pathRewrite: {
            '^/api/v1': '/v1', // Rewrite /api/v1 -> /v1
        },
        secure: false, 
        ws: true,
        logLevel: 'debug'
    }));

    app.use('/v1', createProxyMiddleware({
        target: INTERNAL_API_URL,
        changeOrigin: true,
        secure: false, 
        ws: true,
        logLevel: 'debug'
    }));
}

app.use(express.json());
app.use(express.static(path.join(__dirname, 'dist')));

app.get('/config', (req, res) => {
    res.json({ apiUrl: USE_PROXY ? '' : INTERNAL_API_URL });
});

app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

let server;
let httpServer;

if (sslConfigured) {
    const httpsOptions = {
        key: fs.readFileSync(keyPath),
        cert: fs.readFileSync(certPath),
        minVersion: 'TLSv1.3'
    };
    
    if (process.env.NODE_ENV !== 'test') {
        server = https.createServer(httpsOptions, app).listen(HTTPS_PORT, () => {
            console.log(`Frontend HTTPS Server running on https://localhost:${HTTPS_PORT} (TLS 1.3)`);
            console.log(`Configured API URL: ${INTERNAL_API_URL}`);
        });

        const httpApp = express();
        httpApp.get('*', (req, res) => {
            const hostWithoutPort = req.headers.host ? req.headers.host.split(':')[0] : 'localhost';
            res.redirect(`https://${hostWithoutPort}:${HTTPS_PORT}${req.url}`);
        });
        
        httpServer = http.createServer(httpApp).listen(PORT, () => {
            console.log(`Frontend HTTP redirect server running on http://localhost:${PORT}`);
        });
    } else {
        server = app;
    }
} else {
    if (process.env.NODE_ENV !== 'test') {
        server = app.listen(PORT, () => {
            console.log(`Frontend Server running on http://localhost:${PORT}`);
            console.log(`Configured API URL: ${INTERNAL_API_URL}`);
        });
    } else {
        server = app;
    }
}

module.exports = server;
