"""Local browser GUI. Run: python gui/app.py"""
import json
import sys
import threading
import webbrowser
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data, kind='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', kind + '; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data if isinstance(data, bytes) else json.dumps(data, allow_nan=False).encode())

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self.reply(200, (ROOT / 'index.html').read_bytes(), 'text/html')
        elif self.path == '/health':
            self.reply(200, {'ok': True})
        else:
            self.reply(404, {'error': 'Not found'})

    def do_POST(self):
        if self.path != '/api/run':
            return self.reply(404, {'error': 'Not found'})
        # Keep this local application restricted to its own browser origin.
        origin = self.headers.get('Origin')
        if origin and origin != f'http://{self.headers.get("Host")}':
            return self.reply(403, {'error': 'Cross-origin requests are not accepted.'})
        if not LOCK.acquire(blocking=False):
            return self.reply(409, {'error': 'An experiment is already running. Please wait.'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length < 1 or length > 10000:
                raise ValueError('Invalid request size.')
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError('Expected an experiment object.')
            from solver import solve
            self.reply(200, solve(data))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.reply(400, {'error': str(exc)})
        except ImportError:
            self.reply(500, {'error': 'Missing Python packages. Run: python -m pip install -r gui/requirements.txt'})
        except Exception as exc:
            self.reply(500, {'error': f'Experiment failed: {type(exc).__name__}: {exc}'})
        finally:
            LOCK.release()

    def log_message(self, fmt, *args):
        pass


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    try:
        server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    except OSError:
        print(f'Port {port} is busy. Try: python gui/app.py {port + 1}')
        raise SystemExit(1)
    address = f'http://127.0.0.1:{port}'
    print(f'Quantum HVAC Lab: {address}\nKeep this terminal open. Press Ctrl+C to stop.', flush=True)
    if '--no-browser' not in sys.argv:
        threading.Timer(0.5, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
