#!/usr/bin/env python3
"""
Custom HTTP Server with API Proxy for ngrok
Serves static files and proxies /api/* requests to backend
Handles CORS and HTTPS correctly for mobile sensor/microphone access
"""

import http.server
import socketserver
import urllib.request
import urllib.error
import json
from pathlib import Path
import mimetypes
import os

# Allow socket reuse to prevent "Address already in use" errors
socketserver.TCPServer.allow_reuse_address = True

class ProxyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Set the directory to serve static files from
        self.static_dir = Path(__file__).parent
        super().__init__(*args, directory=str(self.static_dir), **kwargs)

    def do_GET(self):
        if self.path.startswith('/api/'):
            self.proxy_request()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.startswith('/api/'):
            self.proxy_request()
        else:
            self.send_error(405, "Method Not Allowed")

    def do_OPTIONS(self):
        if self.path.startswith('/api/'):
            self.proxy_request()
        else:
            # Handle CORS preflight for static files
            self.send_cors_headers()
            self.end_headers()

    def proxy_request(self):
        """Proxy API requests to the backend server"""
        backend_url = f'http://localhost:5000{self.path}'

        try:
            # Prepare the request to backend
            req = urllib.request.Request(backend_url, method=self.command)

            # Copy headers from client request (except host)
            for header, value in self.headers.items():
                if header.lower() not in ['host', 'connection']:
                    req.add_header(header, value)

            # Add CORS headers
            req.add_header('Origin', f'http://{self.headers.get("Host", "localhost:8000")}')

            # For POST requests, read and forward the body
            if self.command in ['POST', 'PUT', 'PATCH']:
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length > 0:
                    body = self.rfile.read(content_length)
                    req.data = body

            # Make the request to backend
            with urllib.request.urlopen(req) as response:
                # Send response back to client
                self.send_response(response.status)
                self.send_cors_headers()

                # Copy response headers
                for header, value in response.headers.items():
                    if header.lower() not in ['connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade']:
                        self.send_header(header, value)

                self.end_headers()

                # Send response body
                self.wfile.write(response.read())

        except urllib.error.HTTPError as e:
            # Backend returned an error
            self.send_response(e.code)
            self.send_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_data = {'error': f'Backend error: {e.code} {e.reason}'}
            self.wfile.write(json.dumps(error_data).encode())

        except urllib.error.URLError as e:
            # Could not connect to backend
            self.send_response(503)
            self.send_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_data = {'error': 'Backend server is not available', 'details': str(e)}
            self.wfile.write(json.dumps(error_data).encode())

        except Exception as e:
            # Other errors
            self.send_response(500)
            self.send_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_data = {'error': 'Proxy server error', 'details': str(e)}
            self.wfile.write(json.dumps(error_data).encode())

    def send_cors_headers(self):
        """Send CORS headers for API requests"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
        self.send_header('Access-Control-Expose-Headers', 'Content-Type, Cache-Control, X-Accel-Buffering')

    def end_headers(self):
        """Override to add security headers for mobile sensors"""
        # Allow sensors and microphone on HTTPS (ngrok)
        self.send_header('Permissions-Policy', 'microphone=*, camera=*, accelerometer=*, gyroscope=*')
        self.send_header('Feature-Policy', 'microphone *; camera *; accelerometer *; gyroscope *')
        super().end_headers()

    def log_message(self, format, *args):
        """Override logging to show proxy requests"""
        if self.path.startswith('/api/'):
            print(f"[PROXY] {self.address_string()} - {self.command} {self.path}")
        else:
            print(f"[STATIC] {self.address_string()} - {self.command} {self.path}")

def run_server(port=8000):
    """Run the server on specified port"""
    try:
        with socketserver.TCPServer(("", port), ProxyHTTPRequestHandler) as httpd:
            print(f"🚀 Starting proxy server on port {port}")
            print(f"📁 Static files served from: {Path(__file__).parent}")
            print(f"🔗 API requests proxied to: http://localhost:5000")
            print(f"🌐 Access at: http://localhost:{port}")
            print(f"📡 Ready for ngrok tunneling!")
            print("=" * 50)

            httpd.serve_forever()
    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🛑 Server stopped")

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)