# -*- coding: utf-8 -*-
"""Demo server cho Lab 19 — GraphRAG vs Flat RAG.

Chay:
    .\.venv\Scripts\python.exe demo/serve_demo.py
    -> mo http://127.0.0.1:8019

Hai che do:
  * Tab "Ket qua lab": doc outputs/demo_data.json (tinh, khong goi API).
  * Tab "Hoi truc tiep": goi /api/ask -> chay THAT pipeline (Neo4j + Groq) va tra ve
    ca hai cau tra loi kem trace retrieval. Che do nay can .env va graph da duoc nap.

Chi dung thu vien chuan + demo/graphrag_lib.py. Khong bao gio tra secret ve client.
"""
import json
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent.parent
DEMO_DIR = BASE_DIR / "demo"
DEMO_DATA = BASE_DIR / "outputs" / "demo_data.json"
PORT = 8019

_lib = None
_lib_error = None
_lock = threading.Lock()


def get_lib():
    """Import graphrag_lib lazily — demo tinh van chay duoc neu thieu .env/Neo4j."""
    global _lib, _lib_error
    with _lock:
        if _lib is None and _lib_error is None:
            try:
                sys.path.insert(0, str(DEMO_DIR))
                import graphrag_lib
                graphrag_lib.warmup()
                _lib = graphrag_lib
            except Exception as e:
                _lib_error = f"{type(e).__name__}: {e}"
    if _lib is None:
        raise RuntimeError(_lib_error or "khong nap duoc graphrag_lib")
    return _lib


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        raw = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt, *args):
        sys.stderr.write("[demo] " + fmt % args + "\n")

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            f = DEMO_DIR / "index.html"
            if not f.exists():
                return self._send(404, "demo/index.html chua ton tai", "text/plain; charset=utf-8")
            return self._send(200, f.read_bytes(), "text/html; charset=utf-8")

        if path == "/demo_data.json":
            if not DEMO_DATA.exists():
                return self._send(404, json.dumps({"error": "Chua co outputs/demo_data.json — "
                                                            "chay cell 6.1 cua notebook truoc."}))
            return self._send(200, DEMO_DATA.read_bytes())

        if path == "/api/health":
            try:
                lib = get_lib()
                n = lib.run_cypher("MATCH (n:Entity) RETURN count(n) AS n")[0]["n"]
                e = lib.run_cypher("MATCH ()-[r]->() RETURN count(r) AS n")[0]["n"]
                return self._send(200, json.dumps({"live": True, "nodes": n, "edges": e,
                                                   "model": f"{lib.GEN_PROVIDER}/{lib.GEN_MODEL}"}))
            except Exception as ex:
                return self._send(200, json.dumps({"live": False, "reason": str(ex)[:300]}))

        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if urlparse(self.path).path != "/api/ask":
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            question = (payload.get("question") or "").strip()
            if not question:
                return self._send(400, json.dumps({"error": "thieu 'question'"}))
            hops = int(payload.get("max_hops", 2))

            lib = get_lib()
            flat = lib.answer_flat_rag(question)
            graph = lib.answer_graph_rag(question, max_hops=hops)
            body = {
                "question": question,
                "flat": {"answer": flat["answer"], "latency_s": flat["latency_s"],
                         "total_tokens": flat["total_tokens"], "chunks": flat["retrieved"]},
                "graph": {"answer": graph["answer"], "latency_s": graph["latency_s"],
                          "total_tokens": graph["total_tokens"],
                          "lines": graph["graph"]["context"].splitlines()[:60],
                          "diagnostics": graph["graph"]["diagnostics"],
                          "edges": graph["graph"]["edges"][:60],
                          "vector_chunks": graph["vector_docs"]},
            }
            return self._send(200, json.dumps(body, ensure_ascii=False, default=str))
        except Exception:
            return self._send(500, json.dumps({"error": traceback.format_exc()[-800:]}))


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"Demo Lab 19 dang chay tai {url}  (Ctrl+C de dung)")
    print("  - Tab 'Ket qua lab'  : doc outputs/demo_data.json")
    print("  - Tab 'Hoi truc tiep': goi Neo4j + Groq that (can .env)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nDa dung demo server.")


if __name__ == "__main__":
    main()
