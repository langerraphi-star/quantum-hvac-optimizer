"""Production web entrypoint: gunicorn --config web/gunicorn.conf.py web.app:app"""
import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, jsonify, request, send_file
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix
from gui.solver import solve

ROOT = Path(__file__).resolve().parents[1]
BUILD = 'web-5-layers-v1'
app = Flask(__name__, static_folder=None)
app.config['MAX_CONTENT_LENGTH'] = 10000
# Render terminates TLS at its proxy. Do not trust forwarded host headers.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=0, x_proto=1, x_host=0)
solver_lock = threading.Lock()
cache_lock = threading.Lock()
cache = OrderedDict()


@app.after_request
def headers(response):
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-App-Build'] = BUILD
    return response


@app.get('/')
@app.get('/index.html')
def index():
    return send_file(ROOT / 'web' / 'index.html', mimetype='text/html', conditional=False)


@app.get('/health')
def health():
    return jsonify(ok=True, build=BUILD, max_layers=5)


@app.get('/report.pdf')
def report():
    return send_file(ROOT / 'Quantum HVAC Project Report.pdf', mimetype='application/pdf')


@app.post('/api/run')
def run():
    origin = request.headers.get('Origin')
    if origin and (urlsplit(origin).scheme, urlsplit(origin).netloc) != (request.scheme, request.host):
        return jsonify(error='Please run experiments from this website.'), 403
    if not request.is_json:
        return jsonify(error='Send experiment inputs as JSON.'), 415
    try:
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError('Expected an experiment object.')
        key = json.dumps(data, sort_keys=True, allow_nan=False)
    except (BadRequest, ValueError, TypeError):
        return jsonify(error='Invalid experiment inputs. Please check the form.'), 400
    with cache_lock:
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return jsonify(cached)
    # One compute job per instance keeps other visitors and health checks responsive.
    if not solver_lock.acquire(blocking=False):
        response = jsonify(error='Another visitor is running an experiment. Please try again in a few seconds.')
        response.status_code = 429
        response.headers['Retry-After'] = '5'
        return response
    try:
        result = solve(data)
        result['build'] = BUILD
        with cache_lock:
            cache[key] = result
            cache.move_to_end(key)
            while len(cache) > 16:
                cache.popitem(last=False)
        return jsonify(result)
    except (ValueError, TypeError, OverflowError) as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        logging.exception('QAOA experiment failed')
        return jsonify(error='The experiment could not finish. Please try fewer evaluations or run it again.'), 500
    finally:
        solver_lock.release()


@app.errorhandler(RequestEntityTooLarge)
def too_large(error):
    return jsonify(error='Experiment inputs are too large.'), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify(error='Not found'), 404
