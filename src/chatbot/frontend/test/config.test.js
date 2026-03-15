import request from 'supertest';
import { describe, beforeEach, afterEach, test, expect, vi } from 'vitest';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);

// Tell Node to ignore self-signed certificate errors during tests
process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

describe('Configuration Tests', () => {
    let server;
    let originalEnv;

    beforeEach(() => {
        originalEnv = { ...process.env };
        vi.resetModules();
    });

    afterEach(async () => {
        process.env = originalEnv;
        if (server && typeof server.close === 'function') {
            await new Promise(resolve => server.close(resolve));
        }
    });

    test('should use default API URL when API_URL is not set', async () => {
        delete process.env.INTERNAL_API_URL;
        
        // Clear require cache to force reload
        const serverPath = require.resolve('../server.cjs');
        delete require.cache[serverPath];
        server = require(serverPath);
        
        // server might be https or http depending on cert generation. supertest handles both.
        const res = await request(server).get('/config');
        expect(res.statusCode).toBe(200);
        expect(res.body.apiUrl).toBe('https://localhost:6008'); // default is now https
    });

    test('should use configured API URL when INTERNAL_API_URL is set', async () => {
        process.env.INTERNAL_API_URL = 'https://custom-api:9000';
        
        const serverPath = require.resolve('../server.cjs');
        delete require.cache[serverPath];
        server = require(serverPath);
        
        const res = await request(server).get('/config');
        expect(res.statusCode).toBe(200);
        expect(res.body.apiUrl).toBe('https://custom-api:9000');
    });

    test('should have secure headers applied', async () => {
        const serverPath = require.resolve('../server.cjs');
        delete require.cache[serverPath];
        server = require(serverPath);

        const res = await request(server).get('/');
        // Depending on if the file is served or a 404, headers should still be present
        expect(res.headers['strict-transport-security']).toBeDefined();
        expect(res.headers['x-frame-options']).toBe('DENY');
        expect(res.headers['content-security-policy']).toBeDefined();
    });
});
