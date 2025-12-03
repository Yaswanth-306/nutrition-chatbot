'''
 * Copyright (C) 2025 Yaswanth Varma Chitraju
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
'''
const http = require('http');
const https = require('https');

const PORT = 3001;
const N8N_WEBHOOK = 'http://localhost:5678/webhook/95bd289b-6522-4c1e-9a2b-68aec385476c';

const server = http.createServer((req, res) => {
    // Enable CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    // Handle preflight
    if (req.method === 'OPTIONS') {
        res.writeHead(200);
        res.end();
        return;
    }

    if (req.method === 'POST') {
        let body = '';
        
        req.on('data', chunk => {
            body += chunk.toString();
        });
        
        req.on('end', () => {
            // Forward to n8n
            const options = {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Content-Length': body.length
                }
            };

            const proxyReq = http.request(N8N_WEBHOOK, options, (proxyRes) => {
                let responseData = '';
                
                proxyRes.on('data', chunk => {
                    responseData += chunk;
                });
                
                proxyRes.on('end', () => {
                    res.writeHead(proxyRes.statusCode, {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    });
                    res.end(responseData);
                });
            });

            proxyReq.on('error', (error) => {
                console.error('Error:', error);
                res.writeHead(500);
                res.end(JSON.stringify({ error: 'Proxy error' }));
            });

            proxyReq.write(body);
            proxyReq.end();
        });
    }
});

server.listen(PORT, () => {
    console.log(`🚀 Proxy server running on http://localhost:${PORT}`);
    console.log(`Forwarding requests to: ${N8N_WEBHOOK}`);
});
