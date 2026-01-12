#!/usr/bin/env python3
"""
Simple HTTP server for the Scan Zone Editor.
Run this script and open http://<jetson-ip>:8080 in your browser.
"""
import http.server
import socketserver
import os

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Allow cross-origin requests
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

if __name__ == '__main__':
    os.chdir(DIRECTORY)

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        ip = os.popen('hostname -I').read().strip().split()[0]
        print(f"\n{'='*60}")
        print(f"Scan Zone Editor Server")
        print(f"{'='*60}")
        print(f"\nServer läuft auf Port {PORT}")
        print(f"\nÖffnen Sie im Browser:")
        print(f"  http://{ip}:{PORT}/zone_editor.html")
        print(f"\nOder lokal:")
        print(f"  http://localhost:{PORT}/zone_editor.html")
        print(f"\nStrg+C zum Beenden")
        print(f"{'='*60}\n")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer beendet.")
