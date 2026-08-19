# Hướng dẫn thực hiện Lab 19 — GraphRAG vs Flat RAG (bản thao tác thực tế)

> Tài liệu này KHÁC với `huong-dan-lab-graphrag-vs-flatrag.md`.
> File kia giải thích *code khung làm gì*. File này là **kế hoạch thi công**: chạy ở đâu, patch gì,
> chạy theo thứ tự nào, và **những chỗ code khung sẽ vỡ** với dữ liệu/môi trường thật của repo này.

---

## 0. TL;DR — 12 việc phải làm

| # | Việc | Mất bao lâu |
|---|------|-------------|
| 1 | Cài 4 package còn thiếu vào `.venv` (`sentence-transformers`, `faiss-cpu`, `networkx`, `ipykernel`) | 5–10' |
| 2 | **Patch A** — `load_dotenv()` + đổi mọi đường dẫn `/content/...` sang thư mục repo | 3' |
| 3 | **Patch B** — thay cell 1.3 bằng bản stream *có lọc* (dataset không có cột `text`!) | 5' |
| 4 | **Patch C** — tạo cột `text = title + description` trước khi gọi `standardize_news()` | 2' |
| 5 | Chạy Phần 1: Neo4j connect → schema → chunking → coreference | 15' |
| 6 | Chạy Phần 2: NER+RE → entity resolution → UNWIND bulk insert → sanity check | 25–35' |
| 7 | Chạy Phần 3: Flat index → entity matcher → thử 1 câu end-to-end | 10' |
| 8 | **Sinh Golden Dataset TỪ GRAPH** bằng Cypher (không dùng 5 câu starter nguyên bản) | 20' |
| 9 | Chạy evaluation + export 2 CSV vào `outputs/` (và copy sang `reports/`) | 15' |
| 10 | Thu bằng chứng rubric: provenance = 0, audit ≥ 10 dòng, super-node cap | 10' |
| 11 | Viết `reports/lab_report.md` + 3 file phụ (xem §10 — đề bài mâu thuẫn) | 30' |
| 12 | Quét secret rồi commit & push | 5' |

---

## 1. Hiện trạng repo (đã kiểm tra)

- Notebook `Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb`: **37 cell, 0 cell có output** → chưa chạy lần nào.
  Code khung đã đầy đủ, phần lớn dòng gọi hàm ở cuối mỗi cell đang bị comment `#`.
- `.env` đã điền đủ 9 biến (Neo4j Aura + Groq `openai/gpt-oss-120b` + judge OpenAI `gpt-4o-mini` + HF token). `.env` đã nằm trong `.gitignore` ✅
- `.venv` (Python 3.11.9) **thiếu**: `sentence-transformers`, `faiss-cpu`, `torch`, `networkx`, `ipykernel`.
  Đã có: `neo4j 6.2.0`, `pandas 3.0.5`, `numpy 2.4.6`, `datasets 5.0.1`, `groq`, `openai`, `tqdm`, `python-dotenv`.
- `outputs/` và `reports/` mới chỉ có `.gitkeep` + `reports/lab_report.md` (template rỗng). **Chưa có thư mục `data/`.**
- `.gitignore` đã sửa (thêm `!outputs/*.csv`) nhưng thiếu newline cuối file, và `data/golden_dataset.csv` vẫn đang **bị ignore** (do rule `golden_dataset.csv`).

---

## 2. ⚠️ PHÁT HIỆN QUAN TRỌNG NHẤT — dataset không khớp code khung

Tôi đã truy vấn thật dataset `HackerNoon/tech-company-news-data-dump` qua HF datasets-server. Kết quả:

**Schema thật:** `companyName`, `companyUrl`, `published_at`, `url`, `title`, `main_image`, `description`
**Tổng số dòng:** 8.079.363

### Hệ quả 1 — cell 1.5 sẽ CRASH ngay

`standardize_news()` gọi `pick_col(raw, ["text","content","article","body","story"])`.
Không cột nào trong số đó tồn tại → **`KeyError: Missing one of columns`**. Đây là lỗi chặn đứng pipeline.
→ Bắt buộc **Patch C** (§4.3).

### Hệ quả 2 — "bài báo" thực chất là snippet ngắn

`description` là đoạn trích ngắn (thường 30–50 từ, nhiều dòng bị cắt bằng `...`), **không phải full article**.
Đo trên 800 dòng mẫu rải đều toàn dataset:

| Chỉ số | Tỷ lệ |
|---|---|
| `description` không rỗng | 72% |
| `description` ≥ 200 ký tự | 30% |
| Có nhắc từ khoá công nghệ (Microsoft/Google/OpenAI/AI/funding/acquisition...) | 13% |
| **Thoả cả hai** | **6%** |

→ Với `CHUNK_WORDS=220`, mỗi bài chỉ ra **đúng 1 chunk**. Coreference gần như không có gì để resolve
(mỗi chunk là 1 snippet độc lập). Đây là **điểm thuyết minh tốt**: multi-hop bắt buộc phải nối
*giữa các bài báo* qua entity chung — đúng chỗ GraphRAG thắng Flat RAG.

### Hệ quả 3 — dữ liệu được sắp xếp theo `companyName`

Kiểm tra offset 0 / 100k / 400k / 1M / 2M / 4M / 6M cho thấy các dòng gom cụm theo công ty,
gần như alphabetical trong từng shard (`01Synergy` → `Agoda` → `Berry Global` → `Dreamweaver` → ...).

→ Nếu chạy nguyên cell 1.3 (`LIMIT_MB=300`, đọc tuần tự từ dòng 0), bạn chỉ lấy được các công ty
bắt đầu bằng `0/1/A/B` — **không có Microsoft, Google, OpenAI, Nvidia** ở cột `companyName`.
Graph sẽ rời rạc, không có super-node, không có đường multi-hop, và Golden Dataset sẽ không trả lời được.

**Cách xử lý (đã tính toán):** không lọc theo `companyName`, mà **lọc theo từ khoá trong `title + description`**.
Vì các bài báo gắn dưới bất kỳ công ty nào cũng thường nhắc đến big tech
(ví dụ bài "Google's Pixel 8 Software Update Plan" nằm dưới `companyName = MobiDev`).
Với hit-rate 6%, quét ~130k–250k dòng thô là đủ 8.000 dòng chất lượng. → **Patch B** (§4.2).

### Hệ quả 4 — dữ liệu gốc có dòng hỏng quoting

Một số dòng bị dồn hết field vào `companyName` (CSV nguồn escape sai). Bộ lọc ở Patch B loại chúng
tự động (vì `description` của các dòng đó rỗng).

> **Ghi cả 4 hệ quả này vào báo cáo.** Mục "Trade-offs & scale 350MB" và "Debugging khó nhất"
> của rubric chấm đúng loại phân tích này. Bạn có bằng chứng số liệu — hãy dùng nó.

---

## 3. Chọn môi trường: Local (khuyến nghị) hay Colab

**Khuyến nghị: chạy local bằng `.venv` đã có**, vì `.env` đã điền sẵn và output cần nằm đúng
`outputs/` trong repo để commit. Colab sẽ phải nhập lại key thủ công + tải file về.

```powershell
# Trong thư mục repo
.\.venv\Scripts\python.exe -m pip install sentence-transformers faiss-cpu networkx ipykernel jupyterlab
.\.venv\Scripts\python.exe -m ipykernel install --user --name lab19-graphrag --display-name "Lab19 GraphRAG"
```

> `sentence-transformers` kéo theo `torch` CPU (~250MB). Đây là phần lâu nhất của bước cài.

Pre-download embedding model (tránh timeout giữa lab):

```powershell
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

Mở notebook:

```powershell
.\.venv\Scripts\jupyter.exe lab Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb
```

**Chọn kernel "Lab19 GraphRAG"** (không phải Python global).

**Bỏ qua cell 1.1** (`%pip install ...`) — đã cài bằng tay rồi, chạy lại chỉ tốn thời gian và có thể phá version.

### Nếu `faiss-cpu` không cài được trên Windows

Thêm cell shim sau cell 1.2 (API tương thích đủ cho lab này):

```python
try:
    import faiss
except ImportError:
    class _FlatIP:
        def __init__(self, dim): self.v = None
        def add(self, x):
            self.v = x if self.v is None else np.vstack([self.v, x])
        @property
        def ntotal(self): return 0 if self.v is None else len(self.v)
        def search(self, q, k):
            s = q @ self.v.T
            idx = np.argsort(-s, axis=1)[:, :k]
            return np.take_along_axis(s, idx, axis=1).astype("float32"), idx
    class faiss:  # noqa
        IndexFlatIP = _FlatIP
    print("Dùng numpy fallback thay FAISS (nhớ ghi vào báo cáo).")
```

---

## 4. Bốn patch BẮT BUỘC trước khi chạy

### 4.1 Patch A — thêm vào CUỐI cell 1.2 (Imports & config)

`get_secret()` chỉ đọc Colab userdata → fallback `os.environ`. Chạy local thì `.env` **không tự nạp**,
và mọi đường dẫn `/content/...` sẽ trỏ vào `D:\content\...` (không tồn tại).

```python
# ============ LOCAL PATCH A ============
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()   # nạp .env TRƯỚC khi dùng lại get_secret()

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data";    DATA_DIR.mkdir(exist_ok=True)
OUT_DIR  = BASE_DIR / "outputs"; OUT_DIR.mkdir(exist_ok=True)
REP_DIR  = BASE_DIR / "reports"; REP_DIR.mkdir(exist_ok=True)

DATA_PATH   = str(DATA_DIR / "hackernoon_subset.csv")
GOLDEN_PATH = str(DATA_DIR / "golden_dataset.csv")
CHECKPOINT  = str(OUT_DIR  / "graphrag_eval_checkpoint.csv")

# nạp lại secrets sau load_dotenv()
NEO4J_URI      = get_secret("NEO4J_URI", "")
NEO4J_USER     = get_secret("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = get_secret("NEO4J_PASSWORD", "")
NEO4J_DATABASE = get_secret("NEO4J_DATABASE", "neo4j")
GROQ_API_KEY   = get_secret("GROQ_API_KEY", "")
GROQ_MODEL     = get_secret("GROQ_MODEL", "")
JUDGE_PROVIDER = get_secret("JUDGE_PROVIDER", "openai").lower()
JUDGE_MODEL    = get_secret("JUDGE_MODEL", "")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")
HF_TOKEN       = get_secret("HF_TOKEN", "")

# Scale guard điều chỉnh cho dataset snippet (1 bài ~ 1 chunk)
LAB_MAX_ARTICLES      = 3000
LAB_MAX_CHUNKS        = 3000
EXTRACTION_MAX_CHUNKS = 400

assert all([NEO4J_URI, NEO4J_PASSWORD, GROQ_API_KEY, GROQ_MODEL, HF_TOKEN]), "Thiếu secret trong .env"
print("Config OK |", GROQ_MODEL, "| judge:", JUDGE_PROVIDER, JUDGE_MODEL)
# KHÔNG print giá trị key ra output cell.
```

> ⚠️ **Không bao giờ `print()` giá trị API key.** Output cell sẽ được commit lên GitHub → rubric trừ **-10 điểm**.

### 4.2 Patch B — THAY TOÀN BỘ cell 1.3 (stream có lọc)

```python
#@title 1.3 — Stream HackerNoon (FILTERED) -> CSV
import csv, os, re
from datasets import load_dataset
from tqdm.auto import tqdm

DATASET_NAME = "HackerNoon/tech-company-news-data-dump"
OUTPUT_CSV   = DATA_PATH

MAX_SCAN_ROWS  = 250_000   # trần số dòng THÔ được quét (an toàn thời gian)
TARGET_KEPT    = 8_000     # số dòng HỮU ÍCH cần giữ
MIN_DESC_CHARS = 200       # snippet phải đủ dài để trích được quan hệ

TECH_KEYWORDS = [
    "microsoft","google","alphabet","openai","meta","facebook","apple","nvidia","amazon","aws",
    "anthropic","tesla","ibm","oracle","samsung","intel","amd","qualcomm","salesforce","adobe",
    "tiktok","bytedance","uber","airbnb","stripe","spacex","chatgpt","gemini","copilot","claude",
    "artificial intelligence","machine learning","generative ai","llm","cloud computing",
    "acquisition","acquired","acquires","funding round","series a","series b","invests","investment",
    "partnership","partners with","launches","startup","ceo","founder","co-founder",
]
KW_RE = re.compile("|".join(re.escape(k) for k in TECH_KEYWORDS), re.I)

def is_useful(row):
    desc = (row.get("description") or "").strip()
    if len(desc) < MIN_DESC_CHARS:
        return False
    return bool(KW_RE.search((row.get("title") or "") + " " + desc))

FIELDS = ["companyName", "published_at", "url", "title", "description"]

ds = load_dataset(DATASET_NAME, split="train", streaming=True, token=HF_TOKEN)
scanned = kept = 0
with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    pbar = tqdm(total=TARGET_KEPT, desc="Kept rows", unit="row")
    for row in ds:
        scanned += 1
        if scanned > MAX_SCAN_ROWS:
            print(f"\n[DỪNG] chạm trần quét {MAX_SCAN_ROWS:,} dòng thô."); break
        if not is_useful(row):
            continue
        w.writerow({k: (row.get(k) or "") for k in FIELDS})
        kept += 1; pbar.update(1)
        if kept >= TARGET_KEPT:
            print(f"\n[DỪNG] đủ {TARGET_KEPT:,} dòng hữu ích."); break
    pbar.close()

print(f"scanned={scanned:,}  kept={kept:,}  keep_rate={kept/max(scanned,1):.1%}  "
      f"size={os.path.getsize(OUTPUT_CSV)/1e6:.1f} MB -> {OUTPUT_CSV}")
```

**Phải thấy:** `keep_rate` khoảng 4–8%, `kept` = 8.000, file ~4–6 MB.
Nếu `keep_rate` < 1% → dataset đổi schema, kiểm tra lại tên cột.

> **Ghi vào báo cáo (mục scale/trade-off):** đây là *topical sampling có chủ đích*, đánh đổi
> tính đại diện thống kê để lấy mật độ entity — điều kiện cần để đồ thị đủ dày cho multi-hop.

### 4.3 Patch C — sửa phần gọi hàm ở CUỐI cell 1.5

Thay 4 dòng comment cuối cell bằng:

```python
raw_df = load_news(DATA_PATH)
print("Cột thật:", raw_df.columns.tolist(), "| rows:", len(raw_df))

# Dataset KHÔNG có cột 'text' -> ghép title + description
raw_df["text"] = (
    raw_df["title"].fillna("").astype(str).str.strip() + ". " +
    raw_df["description"].fillna("").astype(str).str.strip()
).str.strip()
raw_df = raw_df[raw_df["text"].str.len() >= 120].reset_index(drop=True)

news_df   = standardize_news(raw_df)     # giờ pick_col tìm thấy 'text', 'title', 'published_at'
chunks_df = build_chunks(news_df)
print("articles:", len(news_df), "| chunks:", len(chunks_df))
print("Chunk/article ratio:", round(len(chunks_df)/max(len(news_df),1), 2))
print("published_date rỗng:", float((chunks_df.published_date.fillna("") == "").mean()))
display(chunks_df.head(3))
```

**Phải thấy:**

- `Exact dedup: X -> Y` với Y < X (chứng minh dedup chạy — rubric 1.1),
- ratio ≈ 1.0 (mỗi bài 1 chunk, đúng như phân tích §2),
- `chunks_df` đủ cột `chunk_id, article_id, title, published_date, text`,
- **`published_date rỗng` ≈ 0.0**.

Tại sao dòng cuối quan trọng: nếu `published_date` rỗng thì mọi edge sẽ có `published_date=""`.
Cypher check của lab chỉ bắt `IS NULL` nên vẫn "pass", nhưng `ORDER BY published_date DESC`
của super-node mitigation trở nên vô nghĩa → mất điểm mục 2.1 khi giám khảo soi.

### 4.4 Patch D — chọn subset extraction trải đều (cuối cell 1.7)

`chunks_df.head(400)` = 400 dòng đầu = các công ty đầu bảng chữ cái. Đổi sang lấy mẫu ngẫu nhiên:

```python
extraction_source = chunks_df.sample(
    min(EXTRACTION_MAX_CHUNKS, len(chunks_df)), random_state=SEED
).reset_index(drop=True).copy()

coref_df = run_coref(extraction_source, batch_size=5)
extraction_source = extraction_source.merge(coref_df, on="chunk_id", how="left")

# Spot-check bắt buộc (vật liệu cho câu thuyết minh #1)
mask = extraction_source.unresolved_mentions.map(lambda x: bool(x) and len(x) > 0)
print("Chunk có unresolved_mentions:", int(mask.sum()), "/", len(extraction_source))
display(extraction_source.loc[mask, ["chunk_id", "unresolved_mentions"]].head(10))

# Tìm 1 ví dụ coref ĐÃ đổi text (để trích dẫn cụ thể trong báo cáo)
changed = extraction_source[extraction_source.text != extraction_source.resolved_text]
print("Chunk bị coref sửa:", len(changed))
if len(changed):
    r = changed.iloc[0]
    print("chunk_id:", r.chunk_id); print("BEFORE:", r.text[:300]); print("AFTER :", r.resolved_text[:300])
```

⚠️ Nếu thấy nhiều `COREF_BATCH_FAILED` → model Groq trả JSON hỏng hoặc bị rate-limit. Xem §5.

---

## 5. Ghi chú vận hành LLM (Groq `openai/gpt-oss-120b`)

- **Ngân sách gọi:** coref 400/5 = 80 call + extraction 400/4 = 100 call + eval ~4 call/câu.
  Với snippet ngắn (~100 token/chunk), mỗi call ~1.1k token → tổng ~200k token cho toàn pipeline.
- **Nếu bị 429 / chạy chậm:** tăng `batch_size` extraction lên **6–8** (text ngắn nên vẫn an toàn),
  hoặc hạ `EXTRACTION_MAX_CHUNKS` xuống 250 cho lần chạy đầu — chạy thông suốt trước, tăng lại sau.
- **Nếu `response_format={"type":"json_object"}` lỗi** với `gpt-oss-120b`:
  đổi `.env` sang `GROQ_MODEL=llama-3.3-70b-versatile` (model README khuyến nghị) rồi restart kernel.
- `groq_chat` đã có retry exponential backoff 4 lần — đừng bọc thêm retry ở ngoài.

---

## 6. Chạy pipeline theo thứ tự + điểm kiểm chứng

Bỏ comment các dòng gọi hàm ở cuối mỗi cell. Sau **mỗi** bước, kiểm tra đúng điều kiện dưới đây
rồi mới đi tiếp — sai ở bước sớm sẽ hỏng toàn bộ phần sau.

| Cell | Bỏ comment / chạy | Phải thấy gì mới đi tiếp |
|------|-------------------|--------------------------|
| 1.4 | `connect_neo4j()` → `setup_graph_schema()` | `Neo4j connected.` + `Schema ready.` |
| 1.5 | Patch C | `Exact dedup: X -> Y`, chunks > 1000, `published_date rỗng ≈ 0` |
| 1.6 | chạy cả cell | không lỗi (chỉ định nghĩa hàm) |
| 1.7 | Patch D | có `resolved_text`; xem được vài `unresolved_mentions` |
| 2.1 | `raw_triples_df, extraction_errors_df = run_extraction(extraction_source, batch_size=6)` | `len(raw_triples_df)` **≥ 300**; `extraction_errors_df` gần rỗng |
| 2.2 | `entity_map, entity_resolution_audit_df = build_resolution_map(raw_triples_df)` → `triples_df = canonicalize_triples(raw_triples_df, entity_map)` | audit **≥ 10 dòng** (chưa đủ → §8.2) |
| 2.3 | `nodes_df = build_nodes(triples_df)` → `bulk_insert_nodes(nodes_df)` → `bulk_insert_edges(triples_df)` | không lỗi Cypher |
| 2.4 | `graph_counts, top_degree_df = graph_checks()` | `invalid_provenance_edges: 0` + bảng top-15 degree |
| 3.1 | `build_flat_index(chunks_df)` | `Flat vectors: ~3000` |
| 3.2 | `build_entity_matcher(nodes_df)` | không lỗi |
| 3.3 | chạy cả cell | chỉ định nghĩa hàm |
| 3.4 | **smoke test 1 câu** (dưới đây) | cả 2 hàm trả câu trả lời có `[chunk_id=...]` |

**Smoke test trước khi chạy evaluation hàng loạt** (tiết kiệm rất nhiều thời gian và token):

```python
q = "What did Microsoft announce recently?"   # đổi theo entity có thật trong graph của bạn
g = retrieve_graph_context(q, max_hops=2, edge_limit=50, return_debug=True)
print("seeds:", g["diagnostics"]["matched_seeds"])
print("edges:", g["diagnostics"]["collected_edges"], "| supernode:", g["diagnostics"]["supernode_events"])
print(g["context"][:800])

print(answer_flat_rag(q)["answer"][:600])
print(answer_graph_rag(q)["answer"][:600])
```

Nếu `matched_seeds` rỗng → seed không khớp node nào. Xem tên node thật:

```python
display(pd.DataFrame(run_cypher(
  "MATCH (n:Entity) RETURN n.name AS name, n.entity_type AS type ORDER BY name LIMIT 60")))
```

rồi viết lại câu hỏi bằng đúng tên đó, hoặc hạ `fuzzy_threshold` từ 0.66 → 0.55 (và ghi lý do vào báo cáo).

---

## 7. Golden Dataset — sinh TỪ graph, đừng dùng 5 câu starter nguyên bản

5 câu starter (G01–G05) được viết trước, **không** dựa trên subset bạn nạp.
G01 hỏi "CEO of Hugging Face in 2023" — gần như chắc chắn không có trong 400 chunk đã trích xuất
→ cả Flat lẫn Graph đều trả lời sai → benchmark vô nghĩa, và cả 20 điểm mục 3 bị ảnh hưởng.

`huong-dan-lab-graphrag-vs-flatrag.md` §4.1 **cho phép sửa câu hỏi** miễn giữ đúng nhóm
(`factoid` / `multi-hop` / `cross-doc`) và giải thích lý do trong thuyết minh. Hãy làm vậy.

### 7.1 Cypher để tìm câu hỏi & reference_answer có thật

**Nhóm `factoid`** — quan hệ đơn, confidence cao, có evidence:

```python
display(pd.DataFrame(run_cypher("""
MATCH (a:Entity)-[r]->(b:Entity)
WHERE r.evidence IS NOT NULL AND size(r.evidence) > 40
RETURN a.name AS src, type(r) AS rel, b.name AS dst,
       r.published_date AS date, r.source_chunk_id AS chunk,
       r.confidence AS conf, r.evidence AS ev
ORDER BY r.confidence DESC LIMIT 25
""")))
```

→ Câu hỏi: *"Which company did X acquire, and when?"* · `reference_answer` lấy từ `dst` + `date`.

**Nhóm `multi-hop`** — đường 2 chặng, **hai chặng đến từ hai chunk khác nhau**
(điều kiện then chốt: Flat RAG chỉ retrieve top-k chunk rời rạc nên rất dễ trượt):

```python
display(pd.DataFrame(run_cypher("""
MATCH (a:Entity)-[r1]->(m:Entity)-[r2]->(c:Entity)
WHERE a <> c AND r1.source_chunk_id <> r2.source_chunk_id
RETURN a.name AS a, type(r1) AS rel1, m.name AS mid, type(r2) AS rel2, c.name AS c,
       r1.source_chunk_id AS chunk1, r2.source_chunk_id AS chunk2,
       r1.published_date AS d1, r2.published_date AS d2
LIMIT 30
""")))
```

→ Câu hỏi: *"Which technology is developed by the company that X invested in?"*

**Nhóm `cross-doc`** — cùng cặp entity được chứng thực bởi ≥ 2 chunk:

```python
display(pd.DataFrame(run_cypher("""
MATCH (a:Entity)-[r]->(b:Entity)
WITH a, b, type(r) AS rel,
     collect(DISTINCT r.source_chunk_id) AS chunks,
     collect(DISTINCT r.published_date)  AS dates
WHERE size(chunks) >= 2
RETURN a.name AS a, rel, b.name AS b, size(chunks) AS n_chunks, dates
ORDER BY n_chunks DESC LIMIT 25
""")))
```

→ Câu hỏi: *"Summarize how the relationship between A and B evolved across articles."*

### 7.2 Ghi Golden Dataset ra file

Tối thiểu 5 câu, **phủ đủ 3 nhóm** (gợi ý: 2 factoid + 2 multi-hop + 2 cross-doc = 6 câu):

```python
golden_df = pd.DataFrame([
    {"id":"G01","group":"factoid","question":"...","reference_answer":"...",
     "reference_evidence":"chunk=<chunk_id>; edge=<A -REL-> B @date>"},
    # ... 5 dòng nữa
])
golden_df.to_csv(GOLDEN_PATH, index=False)
validate_golden(golden_df, require_answers=True)   # phải in "Golden Dataset valid."
display(golden_df)
```

`reference_evidence` **luôn ghi kèm `chunk_id` + edge nguồn** — đây chính là vật liệu cho
mục "Phân tích 2 ca lỗi" trong báo cáo, và là bằng chứng bạn không bịa gold answer.

---

## 8. Bằng chứng bắt buộc cho rubric Failure Modes (20 điểm)

### 8.1 Provenance = 0 (6 điểm)

`graph_checks()` đã `assert invalid == 0`. **Giữ nguyên output cell này trong notebook nộp.**
Bổ sung kiểm tra chuỗi rỗng (Cypher check gốc chỉ bắt NULL):

```python
print(run_cypher("""
MATCH ()-[r]->()
RETURN count(r) AS total,
       sum(CASE WHEN r.source_chunk_id IS NULL OR r.source_chunk_id = '' THEN 1 ELSE 0 END) AS bad_chunk,
       sum(CASE WHEN r.published_date  IS NULL OR r.published_date  = '' THEN 1 ELSE 0 END) AS bad_date
"""))
```

Cả `bad_chunk` và `bad_date` phải = 0.

### 8.2 Entity Resolution audit ≥ 10 dòng (6 điểm)

Nếu `entity_resolution_audit_df` quá ít dòng, chạy **thí nghiệm độ nhạy threshold** —
vừa tăng số dòng audit, vừa trả lời sẵn câu thuyết minh #2:

```python
sweep = []
for th in [0.80, 0.85, 0.90, 0.95]:
    _m, _a = build_resolution_map(raw_triples_df, threshold=th)
    vc = _a.decision.value_counts().to_dict()
    sweep.append({"threshold": th, **vc, "audit_rows": len(_a)})
display(pd.DataFrame(sweep))
```

Chọn threshold cuối cùng và giải thích bằng chính bảng này
("0.85 sinh N cặp merge sai kiểu X nên tôi giữ 0.90").

Lấy bằng chứng "similarity cao nhưng KHÔNG merge":

```python
show_resolution_audit(entity_resolution_audit_df)
display(entity_resolution_audit_df.query("decision=='REJECT_GUARD' and similarity > 0.85")
        .sort_values("similarity", ascending=False).head(10))
```

Chép **1 cặp cụ thể** vào báo cáo (ví dụ `Apple` vs `Apple Watch`, `Sam Altman` vs `Steve Altman`).

### 8.3 Super-node cap (8 điểm) — ⚠️ cái bẫy lớn nhất của bài lab

Với 400 chunk, **max degree rất có thể < 100** → `test_supernode_policy()` chạy xong mà
`assert` không bao giờ thực thi → **không có bằng chứng nào**, mất điểm dù code hoàn toàn đúng.

Kiểm tra trước:

```python
print("Max degree thực tế:", int(top_degree_df.degree.max()))
```

**Nếu max degree > 100:** chạy `test_supernode_policy()` bình thường, giữ output. Xong.

**Nếu max degree ≤ 100 — chọn 1 trong 2 (nên làm cả 2):**

*(a) Chứng minh policy bằng ngưỡng hạ tạm thời — rẻ, luôn chạy được:*

```python
top = top_degree_df.iloc[0]
_old = SUPER_NODE_DEGREE
SUPER_NODE_DEGREE = int(top.degree) - 1     # ép node đầu bảng thành "super-node"
try:
    g = retrieve_graph_context(f"Tell me everything about {top['name']}",
                               max_hops=2, edge_limit=200, return_debug=True)
    ev = g["diagnostics"]["supernode_events"]
    print("degree thật:", int(top.degree), "| ngưỡng test:", SUPER_NODE_DEGREE)
    print("supernode_events:", ev)
    print("edges thu được:", g["diagnostics"]["collected_edges"])
    assert ev and all(e["limit"] <= SUPER_NODE_EDGE_CAP for e in ev), "Cap KHÔNG được áp dụng!"
    print("Super-node cap hoạt động: mọi event đều bị chặn ở", SUPER_NODE_EDGE_CAP)
finally:
    SUPER_NODE_DEGREE = _old
```

Ghi rõ trong báo cáo: *"graph ở scale lab chỉ đạt max degree = D < 100, nên tôi hạ ngưỡng
xuống D−1 để kiểm chứng cơ chế; policy production vẫn giữ 100."* — đây là câu trả lời **đúng**,
không phải lách luật.

*(b) Tạo super-node thật:* tăng `EXTRACTION_MAX_CHUNKS` lên 800–1000 rồi chạy lại 2.1 → 2.4.
Tốn thêm ~15–20' gọi LLM. Nếu còn thời gian thì nên làm, vì có super-node thật thì câu thuyết minh
#3 và #5 (rủi ro của việc "ưu tiên 50 cạnh mới nhất") mới có số liệu thật để bàn.

---

## 9. Evaluation & export

```python
validate_golden(golden_df, require_answers=True)
eval_results_df = run_evaluation(golden_df)      # checkpoint tự lưu sau MỖI câu
display(eval_results_df)

comparison_df = comparison_table(eval_results_df)
display(comparison_df)

eval_results_df.to_csv(OUT_DIR / "graphrag_eval_results.csv", index=False)
comparison_df.to_csv(OUT_DIR / "graphrag_vs_flatrag_summary.csv", index=False)

# RUBRIC mục 3.3 nhắc thư mục reports/ trong khi README nhắc outputs/ -> copy cả hai, chi phí bằng 0
eval_results_df.to_csv(REP_DIR / "graphrag_eval_results.csv", index=False)
comparison_df.to_csv(REP_DIR / "graphrag_vs_flatrag_summary.csv", index=False)
print("Exported 2 CSV x 2 nơi")
```

**Lưu bằng chứng ca lỗi ngay lúc chạy** (không thì phải chạy lại, tốn token):

```python
d = eval_results_df.copy()
for m in ["comprehensiveness", "faithfulness", "multi_hop_reasoning"]:
    d[f"delta_{m}"] = d[f"graph_{m}"] - d[f"flat_{m}"]
d["delta_total"] = d[[c for c in d.columns if c.startswith("delta_")]].sum(axis=1)
display(d[["id","group","delta_total","flat_judge_rationale","graph_judge_rationale"]]
        .sort_values("delta_total"))
```

- Dòng `delta_total` **cao nhất** → ca "Flat RAG thất bại, GraphRAG thành công" (mục 4.1 báo cáo).
- Dòng **thấp nhất** → ca "GraphRAG thất bại" (mục 4.2 báo cáo).

Với mỗi ca, in `graph_debug["diagnostics"]` của câu đó để truy nguyên nhân gốc
(seed không match? edge thiếu do extraction sót? bị super-node cap cắt mất?).

---

## 10. Báo cáo — ⚠️ đề bài mâu thuẫn, làm cả hai cho an toàn

| Nguồn | Yêu cầu file báo cáo |
|---|---|
| `README.md` + `ASSIGNMENT.md` (phần cuối) | **1 file duy nhất**: `reports/lab_report.md` |
| `RUBRIC.md` (mục 4.1–4.3) + `ASSIGNMENT.md` (phần trên) | **3 file**: `technical_defense.md`, `failure_analysis.md`, `reflection_[HọTên].md` |

**Cách xử lý:** viết đầy đủ `reports/lab_report.md` (template đã có sẵn — nó bao trọn cả 3 nội dung),
rồi tách/copy ra 3 file kia. Rubric phạt **-5 điểm/file thiếu**, còn thừa file thì không bị phạt.

```
reports/lab_report.md                  <- bản đầy đủ (dùng template có sẵn)
reports/technical_defense.md           <- Phần 1, mục 1-5 (10 câu thuyết minh)
reports/failure_analysis.md            <- Phần 1, mục 4 (2 ca lỗi + root cause)
reports/reflection_DuongNgocHai.md     <- Phần 2 (mapping bài giảng + action plan)
```

### Bảng tra: mỗi câu thuyết minh lấy số liệu ở đâu

| Câu hỏi | Nguồn số liệu trong notebook |
|---|---|
| 1. Coreference sai | `unresolved_mentions` + cặp `text` vs `resolved_text` in ra ở Patch D |
| 2. Threshold + Lexical Guard | bảng sweep §8.2 + dòng `REJECT_GUARD` có similarity > 0.85 |
| 3. Top 3 super-node | `top_degree_df.head(3)` |
| 4. Bảng benchmark + 2 ca lỗi | `comparison_df` + bảng `delta_total` §9 + `graph_debug["diagnostics"]` |
| 5. Trade-off / Agent / scale 350MB | `flat_latency_s` vs `graph_latency_s`, `*_total_tokens` trong `eval_results_df` |
| Mapping bài giảng | tên hàm: `resolve_coref_batch`, `build_resolution_map`, `UF`, `bulk_insert_nodes/edges`, `retrieve_graph_context`, `judge_answer` |

**Vật liệu sẵn cho câu "scale 350MB — bottleneck đầu tiên":** bạn có con số thật —
8.079.363 dòng, keep-rate ~6%, ~200k token LLM cho 400 chunk.
→ Ngoại suy: full dump ≈ 485k chunk hữu ích ≈ 121k call extraction ≈ **~250M token**
→ bottleneck #1 là **số lượt gọi LLM cho extraction**, không phải Neo4j write throughput,
cũng không phải chi phí embedding (MiniLM chạy local, ~485k vector chỉ vài phút GPU).
Giải pháp: async worker queue + batch lớn hơn, model nhỏ cho pass NER đầu tiên, lọc bằng
rule/NER cục bộ trước khi gọi LLM, và HNSW + blocking cho entity resolution thay vì FlatIP O(N²).

**Câu "AI Coding Agent đề xuất gì mà bạn từ chối":** dùng ngay Challenge A —
từ chối pairwise cosine O(N²) trên toàn dataset cho near-dedup, chọn MinHash/LSH.
Hoặc: từ chối đề xuất bỏ `merge_guard` để "tăng recall entity resolution",
vì false merge tạo false edge lan sang mọi câu trả lời multi-hop về sau.

---

## 11. Bonus — chọn cái rẻ nhất trên mỗi điểm

| Bonus | Điểm | Chi phí thực tế | Đánh giá |
|---|---|---|---|
| **C. Self-Correction** | +5 | ~5' (cell 35 đã viết sẵn) | **Làm cái này trước.** Chỉ cần chạy `self_correcting_context()` cho 2–3 câu multi-hop và ghi lại `route` (hop2 / hop3 / hop3+vector) |
| **B. Community Detection** | +5 | ~10' (cell 34 đã viết sẵn) + LLM summarize | Chạy `build_communities()` rồi thêm 1 truy vấn "global" |
| **A. Near-Dedup MinHash** | +3 | ~25' (phải tự viết) | Làm sau cùng nếu còn giờ |

Rubric yêu cầu **"có định lượng trước/sau"** — với Bonus C, bảng này là đủ:

| Question ID | route | edges trước | edges sau | judge score trước → sau |
|---|---|---|---|---|

---

## 12. Trước khi commit — quét secret & sửa .gitignore

```bash
# 1. Không được có key nào trong file sẽ commit
git grep -nE "gsk_[A-Za-z0-9]|sk-proj-|sk-[A-Za-z0-9]{20}|hf_[A-Za-z0-9]{20}|neo4j\+s://[a-z0-9]{6}" \
  -- '*.ipynb' '*.md' '*.py'

# 2. .env KHÔNG được nằm trong danh sách sẽ commit
git status --porcelain | grep -i "\.env$"    # phải rỗng

# 3. Soi output notebook: traceback lỗi Neo4j có thể in ra URI/host
```

Sửa `.gitignore` (đang thiếu newline cuối, và đang chặn golden dataset):

```gitignore
!outputs/*.csv
!data/golden_dataset.csv
```

Commit:

```bash
git add -A
git commit -m "Lab19: hoan thanh pipeline GraphRAG vs FlatRAG + bao cao"
git push origin main
```

---

## 13. Checklist nộp bài (bám rubric 100đ)

- [ ] Notebook chạy `Restart & Run All` không crash, **giữ nguyên output mọi cell**
- [ ] `Exact dedup: X -> Y` hiển thị (1.1 — 8đ)
- [ ] `raw_triples_df` ≥ 300 dòng, có `evidence` + `confidence` (1.2 — 10đ)
- [ ] Output `bulk_insert_*` chạy xong + constraint/index đã tạo (1.3 — 10đ)
- [ ] `answer_flat_rag()` và `answer_graph_rag()` có output thật (1.4 — 12đ)
- [ ] Bằng chứng super-node cap (§8.3) — kể cả khi max degree < 100 (2.1 — 8đ)
- [ ] `invalid_provenance_edges == 0` + check chuỗi rỗng (2.2 — 6đ)
- [ ] `entity_resolution_audit_df` ≥ 10 dòng, đủ 3 loại decision (2.3 — 6đ)
- [ ] Golden ≥ 5 câu, đủ 3 nhóm, **reference_answer thật** lấy từ graph (3.1 — 6đ)
- [ ] Judge chạy hết, có `rationale` (3.2 — 8đ)
- [ ] `outputs/graphrag_eval_results.csv` + `outputs/graphrag_vs_flatrag_summary.csv` (3.3 — 6đ)
- [ ] `reports/lab_report.md` đầy đủ 2 phần + 3 file phụ (4.1–4.3 — 20đ)
- [ ] Bonus C có bảng định lượng trước/sau (+5đ)
- [ ] Không có API key trong bất kỳ file nào được commit (tránh -10đ)
