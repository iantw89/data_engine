"""``python -m data_engine.web [--host H] [--port P]`` starts the review web app."""

import argparse

from .app import serve

parser = argparse.ArgumentParser(description="Data Engine web app")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8610)
arguments = parser.parse_args()
serve(arguments.host, arguments.port)
