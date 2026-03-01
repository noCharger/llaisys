const request = require('supertest');
const path = require('path');

describe('Configuration Tests', () => {
    let server;
    let originalEnv;

    beforeEach(() => {
        originalEnv = { ...process.env };
        jest.resetModules();
    });

    afterEach((done) => {
        process.env = originalEnv;
        if (server) {
            server.close(done);
        } else {
            done();
        }
    });

    test('should use default API URL when API_URL is not set', async () => {
        delete process.env.API_URL;
        server = require('../server');
        
        const res = await request(server).get('/config');
        expect(res.statusCode).toBe(200);
        expect(res.body.apiUrl).toBe('http://localhost:8002');
    });

    test('should use configured API URL when API_URL is set', async () => {
        process.env.API_URL = 'http://custom-api:9000';
        server = require('../server');
        
        const res = await request(server).get('/config');
        expect(res.statusCode).toBe(200);
        expect(res.body.apiUrl).toBe('http://custom-api:9000');
    });
});
