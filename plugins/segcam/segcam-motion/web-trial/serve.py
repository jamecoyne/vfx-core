"""Serves the built page on 127.0.0.1:8125 and saves what it posts to report.txt."""

import http.server
import pathlib
import sys


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        pathlib.Path(self.directory, "report.txt").write_bytes(body)
        self.send_response(204)
        self.end_headers()


def main():
    folder = sys.argv[1]
    handler = lambda *args: Handler(*args, directory=folder)
    with http.server.ThreadingHTTPServer(("127.0.0.1", 8125), handler) as server:
        print("serving http://127.0.0.1:8125", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
