"""A single worker shares the solve lock/cache; threads serve parallel visitors."""
import os

bind = '0.0.0.0:' + os.environ.get('PORT', '8765')
workers = 1
worker_class = 'gthread'
threads = 4
timeout = 180
graceful_timeout = 30
accesslog = '-'
errorlog = '-'
