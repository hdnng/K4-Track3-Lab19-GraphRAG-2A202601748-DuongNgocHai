# -*- coding: utf-8 -*-
"""Thu vien dung cho demo live: nap lai dung pipeline retrieval cua notebook Lab 19.

Doc secret tu .env (khong bao gio in ra), ket noi Neo4j da duoc notebook nap du lieu,
va dung lai FAISS index tren data/hackernoon_subset.csv.
"""
import os, re, json, time, random, unicodedata
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

load_dotenv(find_dotenv(usecwd=True))

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "hackernoon_subset.csv"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ALLOWED_NODE_TYPES = {"Company", "Person", "Technology"}
SUPER_NODE_DEGREE = 100
SUPER_NODE_EDGE_CAP = 50
GLOBAL_EDGE_CAP = 250
MAX_GRAPH_CONTEXT_CHARS = 14000

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
_groq = Groq(api_key=os.environ["GROQ_API_KEY"])
_driver = GraphDatabase.driver(os.environ["NEO4J_URI"],
                               auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")


def norm_space(x):
    return re.sub(r"\s+", " ", str(x or "")).strip()


def norm_entity(name):
    s = unicodedata.normalize("NFKC", norm_space(name)).lower()
    s = re.sub(r"[^\w\s\-\.]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def run_cypher(query, **params):
    with _driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query, **params)
        rows = [r.data() for r in result]
        result.consume()
    return rows


def parse_json_object(text):
    text = re.sub(r"^```(?:json)?\s*", "", str(text).strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("No JSON object found.")
    return json.loads(text[a:b + 1])


def groq_chat(messages, json_mode=False, max_retries=4):
    last = None
    for attempt in range(max_retries):
        try:
            kwargs = {"model": GROQ_MODEL, "messages": messages, "temperature": 0.0}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = _groq.chat.completions.create(**kwargs)
            usage = {}
            if getattr(resp, "usage", None):
                usage = {"total_tokens": resp.usage.total_tokens}
            return resp.choices[0].message.content, usage
        except Exception as e:
            last = e
            time.sleep(min(10, 2 ** attempt + random.random()))
    raise RuntimeError(last)


def groq_json(system, user):
    text, usage = groq_chat([{"role": "system", "content": system},
                             {"role": "user", "content": user}], json_mode=True)
    return parse_json_object(text), usage


# ---------------------------------------------------------------- Flat RAG
_embedder = None
_flat_index = None
_flat_store = None
_entity_vecs = None
_entity_store = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def warmup():
    """Nap FAISS index + entity matcher (mat ~30-60s lan dau)."""
    global _flat_index, _flat_store, _entity_vecs, _entity_store
    if _flat_index is not None:
        return
    df = pd.read_csv(DATA_PATH)
    vecs = get_embedder().encode(df.text.fillna("").tolist(), batch_size=128,
                                 normalize_embeddings=True,
                                 show_progress_bar=False).astype("float32")
    _flat_index = faiss.IndexFlatIP(vecs.shape[1])
    _flat_index.add(vecs)
    _flat_store = df.reset_index(drop=True)

    nodes = pd.DataFrame(run_cypher(
        "MATCH (n:Entity) RETURN n.id AS id, n.name AS name, n.entity_type AS type"))
    if len(nodes):
        _entity_store = nodes
        _entity_vecs = get_embedder().encode(nodes.name.tolist(), batch_size=128,
                                             normalize_embeddings=True,
                                             show_progress_bar=False).astype("float32")


def retrieve_flat_context(query, k=6):
    warmup()
    qv = get_embedder().encode([query], normalize_embeddings=True,
                               show_progress_bar=False).astype("float32")
    scores, ids = _flat_index.search(qv, min(k, _flat_index.ntotal))
    rows = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        r = _flat_store.iloc[int(idx)]
        rows.append({"score": round(float(score), 3), "chunk_id": r.chunk_id,
                     "published_date": str(r.published_date), "text": r.text})
    context = "\n\n".join(
        "[chunk_id=%s | date=%s | score=%.3f]\n%s" % (r["chunk_id"], r["published_date"],
                                                      r["score"], r["text"])
        for r in rows)
    return context, rows


# ------------------------------------------------------------------ GraphRAG
SEED_SYSTEM = ("Extract useful seed entities for graph retrieval.\n"
               "Allowed types: Company, Person, Technology.\n"
               "Do not answer the question. Return strict JSON only.")


def extract_seeds(query):
    obj, _ = groq_json(SEED_SYSTEM,
                       "Question: %s\nReturn json {\"seeds\":[{\"name\":\"...\","
                       "\"type\":\"Company|Person|Technology|null\"}]}" % query)
    return [{"name": norm_space(x.get("name")),
             "type": x.get("type") if x.get("type") in ALLOWED_NODE_TYPES else None}
            for x in obj.get("seeds", []) if norm_space(x.get("name"))]


SEED_CYPHER = """
MATCH (n:Entity)
WHERE (n.name_norm=$name OR $name IN coalesce(n.aliases_norm,[]))
  AND ($typ IS NULL OR n.entity_type=$typ)
RETURN n.id AS id, n.name AS name, n.entity_type AS type LIMIT 5
"""


def match_seeds(query, fuzzy_threshold=0.66):
    warmup()
    matched = []
    for seed in extract_seeds(query):
        exact = run_cypher(SEED_CYPHER, name=norm_entity(seed["name"]), typ=seed["type"])
        if exact:
            matched += exact
            continue
        if _entity_vecs is None:
            continue
        mask = np.ones(len(_entity_store), dtype=bool)
        if seed["type"]:
            mask = _entity_store.type.eq(seed["type"]).to_numpy()
        idxs = np.flatnonzero(mask)
        if not len(idxs):
            continue
        qv = get_embedder().encode([seed["name"]], normalize_embeddings=True,
                                   show_progress_bar=False).astype("float32")[0]
        sims = _entity_vecs[idxs] @ qv
        j = int(np.argmax(sims))
        if float(sims[j]) >= fuzzy_threshold:
            r = _entity_store.iloc[int(idxs[j])]
            matched.append({"id": r.id, "name": r["name"], "type": r.type})
    return list({x["id"]: x for x in matched}.values())


DEGREE_CYPHER = """
MATCH (n:Entity {id:$id}) OPTIONAL MATCH (n)-[r]-() RETURN count(r) AS degree
"""

EDGES_CYPHER = """
MATCH (n:Entity {id:$id})
MATCH (n)-[r]-(m:Entity)
RETURN startNode(r).id AS source_id, startNode(r).name AS source_name,
       startNode(r).entity_type AS source_type, type(r) AS relation,
       endNode(r).id AS target_id, endNode(r).name AS target_name,
       endNode(r).entity_type AS target_type,
       r.source_chunk_id AS source_chunk_id, r.published_date AS published_date,
       r.evidence AS evidence, m.id AS neighbor_id
ORDER BY coalesce(r.published_date,'') DESC LIMIT $limit
"""


def node_degree(node_id):
    return int(run_cypher(DEGREE_CYPHER, id=node_id)[0]["degree"])


def recent_edges(node_id, limit):
    return run_cypher(EDGES_CYPHER, id=node_id, limit=int(limit))


def textualize(edges):
    edges = sorted(edges, key=lambda e: e.get("published_date") or "", reverse=True)
    lines, used = [], 0
    for e in edges:
        line = ("%s [%s] -%s-> %s [%s] | date=%s | chunk=%s"
                % (e["source_name"], e["source_type"], e["relation"], e["target_name"],
                   e["target_type"], e.get("published_date") or "unknown",
                   e.get("source_chunk_id") or "unknown"))
        if e.get("evidence"):
            line += " | evidence=" + norm_space(e["evidence"])
        if used + len(line) + 1 > MAX_GRAPH_CONTEXT_CHARS:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def retrieve_graph_context(query, max_hops=2, edge_limit=50):
    seeds = match_seeds(query)
    if not seeds:
        return {"context": "", "edges": [],
                "diagnostics": {"reason": "NO_SEED", "matched_seeds": [],
                                "expanded_nodes": 0, "collected_edges": 0,
                                "supernode_events": []}}
    frontier = deque((x["id"], 0) for x in seeds)
    expanded, seen_edges, collected, supernode_events = set(), set(), [], []
    while frontier and len(collected) < GLOBAL_EDGE_CAP:
        node_id, hop = frontier.popleft()
        if node_id in expanded or hop >= max_hops:
            continue
        expanded.add(node_id)
        degree = node_degree(node_id)
        limit = int(edge_limit)
        if degree > SUPER_NODE_DEGREE:
            limit = min(limit, SUPER_NODE_EDGE_CAP)
            supernode_events.append({"node_id": node_id, "degree": degree, "limit": limit})
        for e in recent_edges(node_id, limit):
            key = (e["source_id"], e["relation"], e["target_id"], e["source_chunk_id"])
            if key in seen_edges:
                continue
            seen_edges.add(key)
            collected.append(e)
            if len(collected) >= GLOBAL_EDGE_CAP:
                break
            nb = e.get("neighbor_id")
            if nb and nb not in expanded and hop + 1 < max_hops:
                frontier.append((nb, hop + 1))
    return {"context": textualize(collected), "edges": collected,
            "diagnostics": {"matched_seeds": seeds, "expanded_nodes": len(expanded),
                            "collected_edges": len(collected),
                            "supernode_events": supernode_events}}


ANSWER_SYSTEM = ("Answer only from supplied context.\n"
                 "Be concise but complete. Do not invent facts.\n"
                 "Cite provenance inline as [chunk_id=...] whenever possible.\n"
                 "If evidence is insufficient or conflicting, say so.")


def generate_answer(question, context):
    t0 = time.perf_counter()
    text, usage = groq_chat([{"role": "system", "content": ANSWER_SYSTEM},
                             {"role": "user", "content":
                              "QUESTION:\n%s\n\nCONTEXT:\n%s\n\nANSWER:" % (question, context)}])
    return {"answer": text.strip(), "latency_s": round(time.perf_counter() - t0, 2),
            "total_tokens": usage.get("total_tokens")}


def answer_flat_rag(question, k=6):
    context, retrieved = retrieve_flat_context(question, k=k)
    out = generate_answer(question, context)
    out.update({"context": context, "retrieved": retrieved})
    return out


def answer_graph_rag(question, max_hops=2, edge_limit=50):
    g = retrieve_graph_context(question, max_hops=max_hops, edge_limit=edge_limit)
    vctx, vdocs = retrieve_flat_context(question, k=4)
    context = "=== GRAPH ===\n%s\n\n=== VECTOR ===\n%s" % (g["context"], vctx)
    out = generate_answer(question, context)
    out.update({"context": context, "graph": g, "vector_docs": vdocs})
    return out
