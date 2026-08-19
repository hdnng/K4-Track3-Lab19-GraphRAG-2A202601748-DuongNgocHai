# Huong dan chi tiet Lab: GraphRAG vs Flat RAG - Production-Grade Lab

> Track 3 - Lab 19 | Thoi luong: 120 phut | Muc do: Trung cap

> Nguon: VLearn Codelabs (codelabs.vlearn.dev) - Tai lieu nay la ban tong hop va huong dan thao tac, khong sao chep toan bo code trong notebook goc. Hoc vien can mo notebook Colab that su cua lab de lay code khung day du.

---

## LUU Y QUAN TRONG - DOC KY TRUOC KHI BAT DAU

1. **Muc tieu cot loi**: Xay pipeline Hybrid GraphRAG hoan chinh tren Neo4j (chunk -> coreference resolution -> NER/RE -> entity resolution -> bulk insert bang UNWIND -> graph traversal co kiem soat super-node -> hybrid retrieval), roi so sanh dinh luong voi Flat RAG (vector-only) bang Golden Dataset va LLM-as-a-Judge (do quality, latency, token usage).

2. **Notebook da co code khung san**. Nhiem vu chinh KHONG phai viet lai toan bo pipeline, ma la: chay code, chinh prompt/threshold/retrieval policy cho phu hop, va thuyet minh (giai thich) ly do lua chon cua minh. Dung bo qua phan thuyet minh vi no chiem 20% diem.

3. **Chuan bi Secrets TRUOC khi vao lab** (khai bao trong Colab Secrets, KHONG hard-code API key vao notebook):
   - `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (va tuy chon `NEO4J_DATABASE`)
   - `GROQ_API_KEY`, `GROQ_MODEL`
   - `HF_TOKEN` de stream dataset tu Hugging Face
   - `JUDGE_PROVIDER` (openai hoac groq), `JUDGE_MODEL`, va `OPENAI_API_KEY` neu dung OpenAI lam judge

4. **Dataset dung trong lab**: `HackerNoon/tech-company-news-data-dump` tren Hugging Face. Neu dataset yeu cau gated access, phai vao trang dataset tren Hugging Face bam Agree/Request access TRUOC khi chay cell stream du lieu, neu khong se bi loi khi load.

5. **Scale guard - RAT QUAN TRONG**: Dataset goc ~350MB, KHONG duoc gui toan bo qua LLM trong 2 gio lab. Mac dinh dung subset voi cac gioi han:
   - `LAB_MAX_ARTICLES = 1500` (so bai bao toi da sau dedup)
   - `LAB_MAX_CHUNKS = 3000` (so chunk toi da dua vao Flat RAG index)
   - `EXTRACTION_MAX_CHUNKS = 400` (so chunk toi da dua qua LLM de trich xuat triple)
   - `LIMIT_MB = 300` khi stream dataset tu Hugging Face (co the tang sau buoi hoc)
   Kien truc phai the hien kha nang scale duoc, con volume trong gio lab chi de chung minh pipeline chay dung.

6. **3 failure-mode trong tam cua lab** (bat buoc xu ly it nhat 2/3 de dat rubric 30%):
   - **Coreference sai** -> tao ra false edge (canh sai trong graph). Nguyen tac: chi resolve dai tu khi antecedent ro rang trong CUNG mot chunk, khong bao gio bay dat (invent) fact, giu nguyen so lieu/ngay/ticker/ten san pham, neu con nghi ngo (ambiguous) thi giu nguyen va log vao `unresolved_mentions`.
   - **Entity Resolution sai** -> merge nham hai entity khac nhau (vi du hai cong ty ten giong nhau nhung khac nhau) hoac khong merge duoc cac bien the cua cung 1 entity (viet tat, ticker, alias). Can co "audit table" ghi lai moi quyet dinh merge/reject va similarity score.
   - **Super-node** -> mot entity co qua nhieu canh (degree lon, vi du cong ty lon nhu Microsoft/Google) se lam "no" context khi traversal. Can co nguong `SUPER_NODE_DEGREE` va gioi han `SUPER_NODE_EDGE_CAP` khi expand.

7. **Rubric cham diem** (bam sat khi lam bai):
   - 30% Chay duoc code: graph nap thanh cong, schema dung, xuat duoc bang so sanh.
   - 30% Failure modes: xu ly duoc it nhat 2/3 van de Super-node, Entity Resolution, Coreference.
   - 20% Evaluation: chay het Golden Dataset, phan tich hop ly.
   - 20% Thuyet minh: giai thich duoc kien truc va cach kiem soat AI Coding Agent (neu dung).

8. **Golden Dataset**: Notebook chi co san 5 cau hoi mau (starter), trong do CHI CO 1 cau (G01) co san reference_answer. 4 cau con lai (G02-G05) phai duoc hoc vien tu dien `reference_answer` thuc te dua tren du lieu da nap vao Neo4j/Flat index TRUOC khi chay evaluation cuoi cung - day la buoc de bo qua trong khi lam bai, nen lam som.

9. **Neu dung AI Coding Agent** (Cursor/Copilot/ChatGPT...) de ho tro code trong 2 "Challenge" (Near Dedup va Entity Resolution guard) hoac phan Bonus, phai neu ro trong bao cao: threshold da chon, false positive gap phai, cach audit cac cap bi merge, va nhung de xuat cua Agent ma ban KHONG dung kem ly do - day la mot muc trong phan thuyet minh 20%.

10. **Nop bai**: Chuan bi san link GitHub/Drive/LMS chua notebook da chay xong (co output cell, khong can chay lai tu dau khi nop) va bang so sanh CSV.

---

## Tong quan bai Lab

Hoc vien xay dung mot pipeline Hybrid GraphRAG hoan chinh tren Neo4j, xu ly cac bai toan thuc te (coreference, entity resolution, super-node), roi so sanh dinh luong voi Flat RAG bang Golden Dataset va LLM-as-a-Judge.

Kien thuc/ky nang can co truoc khi lam lab: Python va pandas o muc co ban; khai niem RAG, embedding va vector search; Cypher/Neo4j o muc co ban (khong bat buoc vi da co code khung); cach goi LLM API (Groq/OpenAI) va xu ly JSON output.

### Timeline de xuat (tong 120 phut)

| Thoi gian | Noi dung |
|---|---|
| 00-15 phut | Setup & Preprocessing: cai moi truong, load du lieu, dedup, chunk, coreference resolution |
| 15-45 phut | Triple Extraction & Neo4j Bulk Insert: NER/RE, entity resolution bang vector similarity, bulk insert bang UNWIND |
| 45-75 phut | Flat RAG & Hybrid GraphRAG: xay Flat RAG baseline, graph traversal co super-node mitigation, hybrid retrieval |
| 75-105 phut | Golden Dataset & LLM-as-a-Judge: tao Golden Dataset, cham diem bang LLM-as-a-Judge, xuat bang so sanh hai kien truc |
| 105-120 phut | Failure-mode Checks & Submission: kiem tra failure mode, lam bonus (tuy chon), export ket qua, thuyet minh ky thuat |

### Ket thuc bai lab, ban se co

- Xay dung duoc pipeline Hybrid GraphRAG end-to-end tren Neo4j bang bulk insert UNWIND.
- Xu ly duoc Coreference Resolution, Entity Resolution va Super-node Mitigation trong thuc te.
- So sanh dinh luong Flat RAG va GraphRAG bang Golden Dataset + LLM-as-a-Judge.
- Giai thich duoc kien truc, trade-off latency/token va failure modes cua he thong RAG.

---

## PHAN 1 - SETUP & PREPROCESSING (0-15 phut)

### Buoc 1.0 - Tao Secrets tren Google Colab

Vao bieu tuong chia khoa (Secrets) ben trai Colab, tao cac secret sau: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `GROQ_API_KEY`, `GROQ_MODEL`, `HF_TOKEN`, `JUDGE_PROVIDER`, `JUDGE_MODEL`, `OPENAI_API_KEY` (neu judge la OpenAI). Nho bat toggle "Notebook access" cho tung secret. Tuyet doi khong dan API key truc tiep vao code cell.

### Buoc 1.1 - Install thu vien

Chay cell pip install cac package: neo4j, pandas, numpy, pyarrow, sentence-transformers, faiss-cpu, groq, openai, tqdm, networkx, spacy, datasets, langchain-community, llama-index. Doi cell chay xong hoan toan (co the mat 1-2 phut) truoc khi qua buoc sau.

### Buoc 1.2 - Import va khai bao config

Chay cell import (os, re, json, pandas, numpy, neo4j driver, sentence-transformers, faiss...) va set SEED=42 de ket qua reproducible. Ham `get_secret()` se doc secret tu Colab userdata truoc, neu khong co thi fallback sang bien moi truong - kiem tra ham nay chay khong loi tuc la secrets da duoc doc dung.

Kiem tra lai cac hang so scale guard da duoc set dung: `LAB_MAX_ARTICLES=1500`, `LAB_MAX_CHUNKS=3000`, `EXTRACTION_MAX_CHUNKS=400`, `CHUNK_WORDS=220`, `CHUNK_OVERLAP_WORDS=40`.

### Buoc 1.3 - Download dataset HackerNoon bang Hugging Face Streaming

Cell nay stream truc tiep dataset `HackerNoon/tech-company-news-data-dump` va ghi dan ra file CSV `/content/hackernoon_subset.csv`, khong tai toan bo dataset vao RAM.

- Neu dataset yeu cau gated access: mo trang dataset tren Hugging Face, dang nhap va bam Agree/Request access TRUOC, neu khong cell se loi authorization.
- Hai co che dung: `LIMIT_ROWS` (so dong toi da) va `LIMIT_MB` (dung luong file toi da, mac dinh 300MB). Bien `PRIORITIZE_MB=True` nghia la uu tien dung theo dung luong MB; `False` nghia la progress bar theo so dong nhung van giu hard-stop LIMIT_ROWS.
- Chay cell va cho den khi thay dong "Hoan thanh" voi so rows va dung luong file. Sau khi chay xong, bien `DATA_PATH` da tu dong tro ve file CSV vua tao, cell loader ke tiep chay truc tiep duoc.

Neu gap loi: kiem tra lai (1) HF_TOKEN da dung chua, (2) da Agree/Request access tren Hugging Face chua, (3) ket noi mang cua Colab.

### Buoc 1.4 - Ket noi Neo4j va tao schema

Goi ham `connect_neo4j()` de ket noi driver Neo4j (dung Aura hoac instance tu host) - se in ra "Neo4j connected" neu thanh cong. Sau do goi `setup_graph_schema()` de tao constraint unique cho `Entity.id` va index cho `name_norm` cua Entity/Company/Person/Technology. Hai lenh nay dang bi comment (#) trong code khung - can bo comment va chay truoc khi qua Phan 2.

### Buoc 1.5 - Load du lieu, exact dedup va chunking

- `load_news(DATA_PATH)`: doc file CSV/JSON/Parquet thanh DataFrame.
- `standardize_news(raw_df)`: tu dong nhan dien cot text/title/date/id (ho tro nhieu ten cot khac nhau), chuan hoa khoang trang, loai bai qua ngan (<80 ky tu), tinh hash `dedup_key` de loai bai trung lap tuyet doi (exact dedup), va lay mau (sample) toi da `LAB_MAX_ARTICLES` bai theo SEED co dinh de reproducible.
- `build_chunks(news_df)`: cat moi bai bao thanh cac chunk ~220 tu, overlap 40 tu (sliding window), dat `chunk_id` dang `{article_id}::c{index}`, va dung lai khi vuot `LAB_MAX_CHUNKS`.
- Bo comment 4 dong cuoi cell nay (`raw_df = load_news(...)`, `news_df = standardize_news(...)`, `chunks_df = build_chunks(...)`, `display(chunks_df.head())`) va chay de tao `chunks_df` - day la dau vao cho tat ca cac buoc sau.

**Challenge A (AI Coding Agent - tuy chon nhung nen lam de cong diem thuyet minh)**: Exact hash dedup chi bat duoc ban sao chinh xac, KHONG bat duoc bai bi "repost"/gan trung (near-duplicate). Hay dung AI Agent thiet ke them MinHash/LSH, SimHash, hoac embedding+ANN de phat hien near-duplicate. Khong duoc chap nhan cach lam pairwise cosine O(N^2) tren toan dataset (qua cham). Trong bao cao can neu: threshold da chon, ty le false positive quan sat duoc, va cach ban audit (kiem tra lai) cac cap bi merge.

### Buoc 1.6 - LLM wrapper co retry va JSON parsing

Cell nay tao Groq client va 2 ham quan trong dung xuyen suot lab: `parse_json_object(text)` (bam ra JSON object tu output LLM, tu dong bo code-fence markdown) va `groq_chat(...)`/`groq_json(...)` (goi Groq API voi retry co exponential backoff toi da 4 lan, temperature=0.0 de on dinh, ho tro `json_mode=True`). Khong can sua gi o buoc nay, chi can chay de cac buoc sau su dung.

### Buoc 1.7 - Coreference Resolution

Day la buoc "chuan hoa" van ban truoc khi trich xuat, cuc ky quan trong vi coreference sai se tao ra canh (edge) sai trong graph sau nay.

Nguyen tac bat buoc cua prompt coreference (da duoc thiet ke san trong `COREF_SYSTEM`): chi resolve dai tu/tham chieu khi antecedent duoc ho tro ro rang trong CUNG mot chunk; khong duoc bay dat (invent) fact moi; phai giu nguyen ngay thang, so lieu, ma ticker va ten san pham; neu con nghi ngo (ambiguous) thi giu nguyen text goc va log vao mang `unresolved_mentions`.

- Ham `resolve_coref_batch(batch_df)` gui tung batch nho (mac dinh 5 chunk/batch) qua LLM va tra ve `resolved_text` cung `unresolved_mentions`.
- Ham `run_coref(chunks_subset, batch_size=5)` lap qua toan bo subset, neu 1 batch loi thi fallback giu nguyen text goc va gan flag `COREF_BATCH_FAILED` de khong lam sap ca pipeline.
- Bo comment 3 dong cuoi: lay `extraction_source = chunks_df.head(EXTRACTION_MAX_CHUNKS)`, chay `coref_df = run_coref(extraction_source)`, roi merge ket qua vao `extraction_source`. Day la du lieu dau vao cho buoc trich xuat triple o Phan 2.
- **Nen lam**: sau khi chay xong, xem qua vai dong `unresolved_mentions` khac rong de hieu cac truong hop LLM tu choi resolve - se can khi tra loi cau hoi thuyet minh o Phan 5.

---

## PHAN 2 - TRIPLE EXTRACTION & NEO4J BULK INSERT (15-45 phut)

### Graph schema can tuan thu

- Node types (co label goc `Entity`): `Company`, `Person`, `Technology`.
- Relation types duoc phep (allowlist): `ACQUIRED`, `DEVELOPED`, `INVESTED_IN`, `FOUNDED`, `WORKED_AT`, `PARTNERED_WITH`, `USES`, `LEADS`.
- Moi edge bat buoc phai co `source_chunk_id` va `published_date` (provenance - de biet thong tin lay tu dau); khuyen nghi them `evidence` (cau trich dan) va `confidence` (do tin cay).
- Relation type sinh ra tu LLM phai duoc loc qua allowlist truoc khi ghep vao cau lenh Cypher, tuyet doi khong noi truc tiep string tu LLM vao Cypher de tranh injection va relation "la".

### Buoc 2.1 - NER + RE Extraction (trich xuat thuc the va quan he)

- `EXTRACT_SYSTEM` yeu cau LLM chi trich xuat quan he thuoc allowlist, uu tien precision hon recall (chi lay quan he chac chan, bo qua neu khong ro), va moi quan he phai co doan evidence ngan kem theo.
- Ham `extract_batch(batch_df)` gui batch chunk (dung `resolved_text` tu buoc coref neu co, fallback text goc) qua LLM, yeu cau tra ve JSON co cau truc `items -> relations` gom source/source_type/relation/target/target_type/evidence/confidence.
- Ham `run_extraction(source_df, batch_size=4)` lap qua toan bo extraction_source, loc bo cac quan he khong hop le (source/target rong, type khong thuoc allowlist, relation khong thuoc allowlist), gom thanh `raw_triples_df`; batch loi duoc ghi vao `errors_df` de audit.
- Bo comment cuoi cell, chay `raw_triples_df, extraction_errors_df = run_extraction(extraction_source)` va xem `raw_triples_df.head()`. Neu `extraction_errors_df` co nhieu dong, can xem lai batch_size hoac rate limit cua Groq.

### Buoc 2.2 - Entity Resolution bang Vector Similarity

Muc tieu: gop cac cach viet khac nhau cua CUNG mot thuc the (vi du "MSFT", "Microsoft Corp", "Microsoft Corporation" -> "Microsoft") nhung KHONG gop nham hai thuc the khac nhau. Pipeline gom 4 buoc:

1. **Manual aliases**: bang `MANUAL_ALIASES` cho cac ticker/ten rat pho bien (msft, googl, aapl...) - map thang ve ten chuan, uu tien cao nhat.
2. **Embedding ANN candidate**: dung `sentence-transformers/all-MiniLM-L6-v2` encode ten entity, dung FAISS IndexFlatIP tim top-k lang gieng gan nhat theo cosine similarity, mac dinh `threshold=0.90`.
3. **Lexical guard** (`merge_guard`): sau khi vuot threshold embedding, con phai qua kiem tra lexical (bo suffix Inc/Corp/Ltd... roi so sanh chuoi bang SequenceMatcher ratio >= 0.72) moi duoc merge thuc su - day la lop bao ve giam false merge; cap nao vuot threshold embedding nhung khong qua guard se bi danh dau `REJECT_GUARD`.
4. **Union-Find (UF)** de gop nhom cac entity da duoc phep merge, chon ten canonical la ten xuat hien nhieu nhat (hoac ngan/gon nhat khi hoa).
5. Xuat ra `entity_resolution_audit_df` ghi lai moi cap (left, right, similarity, decision: MERGE_MANUAL/MERGE_VECTOR/REJECT_GUARD) - bang nay la bang chung bat buoc phai nop.

- Bo comment chay: `entity_map, entity_resolution_audit_df = build_resolution_map(raw_triples_df)` roi `triples_df = canonicalize_triples(raw_triples_df, entity_map)`. Xem `entity_resolution_audit_df.head(20)` de kiem tra cac quyet dinh merge co hop ly khong.
- **Nen lam**: thu doi `threshold` (vi du 0.85 hoac 0.95) va quan sat so luong MERGE_VECTOR/REJECT_GUARD thay doi ra sao - day chinh la du lieu can cho phan thuyet minh "entity threshold bao nhieu, vi sao".

**Challenge B (AI Coding Agent - tuy chon)**: Cai tien lexical guard de xu ly tot hon cac truong hop: ticker (vi du TSLA vs Tesla), suffix cong ty (Inc./Corp./Ltd.), san pham co chua ten cong ty (vi du "Google Search" khong duoc merge voi "Google" o muc entity Company neu that su la 2 loai khac nhau), va nguoi trung ho/ten gan giong nhau (khong duoc merge nham 2 nguoi khac nhau).

### Buoc 2.3 - Bang Node va Bulk Insert bang UNWIND

- `build_nodes(triples_df)`: gom tat ca source/target thanh mot bang node duy nhat, group theo (id, name, name_norm, type), gop het cac bien the (alias) tung xuat hien thanh danh sach `aliases`/`aliases_norm` luu tren node - giup sau nay match seed entity nhanh hon.
- `bulk_insert_nodes(nodes_df, batch_size=1000)`: VOI TUNG LOAI NODE (Company/Person/Technology), dung `UNWIND $rows AS row MERGE (n:Entity {id: row.id}) SET n:Type, ...` de insert theo lo (batch 1000 dong/lan) - đây chính là yêu cầu bat buoc "bulk insert bang UNWIND, khong insert tung row" cua lab.
- `bulk_insert_edges(triples_df, batch_size=1000)`: kiem tra truoc rang du lieu co du cot `source_chunk_id` va `published_date` (provenance bat buoc), sau do voi tung loai relation, dung `UNWIND $rows AS row MATCH (s)...MATCH (t)...MERGE (s)-[r:REL {source_chunk_id: row.source_chunk_id}]->(t) SET r.published_date=..., r.evidence=..., r.confidence=...` theo batch.
- Bo comment 3 dong cuoi, chay theo dung thu tu: `nodes_df = build_nodes(triples_df)` -> `bulk_insert_nodes(nodes_df)` -> `bulk_insert_edges(triples_df)`.

### Buoc 2.4 - Sanity checks (kiem tra suc khoe du lieu)

Ham `graph_checks()` chay 3 kiem tra: (1) dem so canh bi thieu provenance (`source_chunk_id` hoac `published_date` null) - PHAI BANG 0, co `assert invalid == 0` de chan lab neu sai; (2) tong so node va edge trong graph; (3) top 15 node co degree (so canh) cao nhat - day chinh la danh sach ung vien "super-node" se dung o Phan 3 va Phan 5.

Bo comment chay `graph_counts, top_degree_df = graph_checks()` va luu lai bang `top_degree_df` de doi chieu voi buoc kiem tra super-node sau nay.

---

## PHAN 3 - FLAT RAG & HYBRID GRAPHRAG (45-75 phut)

Nguyen tac so sanh cong bang: ca hai kien truc dung CUNG mot embedding model va CUNG mot generator LLM, chi khac nhau ve CACH RETRIEVE context - de phan tich tap trung vao kien truc retrieval, khong bi nhieu boi model khac nhau.

### Buoc 3.1 - Xay Flat RAG baseline

- `build_flat_index(chunks_df)`: encode toan bo `chunks_df.text` bang sentence-transformers (normalize_embeddings=True), dua vao FAISS `IndexFlatIP` (inner product = cosine tren vector da normalize).
- `retrieve_flat_context(query, k=6)`: encode cau hoi, tim k chunk gan nhat, tra ve context da ghep chuoi kem `chunk_id`, `published_date`, `score` (de trace duoc provenance) va DataFrame ket qua.
- Bo comment chay `build_flat_index(chunks_df)`.

### Luong xu ly cua Graph Retrieval (tong quan truoc khi doc code)

1. LLM trich xuat "seed entities" tu cau hoi.
2. Tim seed do trong Neo4j (khop chinh xac ten/alias truoc), neu khong co thi fallback bang embedding fuzzy match.
3. BFS (duyet theo chieu rong) tu cac seed, toi da `max_hops` buoc.
4. Neu mot node co degree > 100 (super-node) thi chi lay toi da 50 canh MOI NHAT cua node do, khong lay het.
5. Co gioi han tong so canh toan cuc (global edge cap) de tranh "no" context (context explosion).
6. Chuyen subgraph thu duoc thanh text co kem provenance (ngay, chunk_id, evidence) de dua vao prompt.

### Buoc 3.2 - Seed matching (khop thuc the hat giong)

- `extract_seeds(query)`: goi LLM (`SEED_SYSTEM` yeu cau KHONG tra loi cau hoi, chi trich seed entity thuoc 3 loai Company/Person/Technology) de lay danh sach ten thuc the tiem nang tu cau hoi nguoi dung.
- `build_entity_matcher(nodes_df)`: encode truoc ten tat ca node trong graph thanh vector, luu vao `entity_match_vectors`/`entity_match_store` de tra cuu nhanh (chay 1 lan sau khi co `nodes_df`).
- `match_seeds(query, fuzzy_threshold=0.66)`: voi moi seed, thu tim EXACT match trong Neo4j qua `name_norm`/`aliases_norm` truoc; neu khong co ket qua moi fallback sang so sanh embedding (cosine) voi nguong 0.66 va lay node co similarity cao nhat.
- Bo comment chay `build_entity_matcher(nodes_df)` sau khi da co `nodes_df` tu Phan 2.

### Buoc 3.3 - Graph traversal + Super-node mitigation

Cac tham so kiem soat (co the tinh chinh va giai thich trong bao cao):
- `SUPER_NODE_DEGREE = 100` - nguong de coi mot node la "super-node".
- `SUPER_NODE_EDGE_CAP = 50` - so canh toi da duoc lay tu mot super-node.
- `GLOBAL_EDGE_CAP = 250` - tong so canh toi da thu thap cho toan bo qua trinh traversal cua 1 cau hoi.
- `MAX_GRAPH_CONTEXT_CHARS = 14000` - do dai toi da (ky tu) cua context dang text dua vao prompt.

- `node_degree(node_id)`: dem so canh (ke ca 2 chieu) cua mot node bang Cypher.
- `recent_edges(node_id, limit)`: lay toi da `limit` canh, SAP XEP THEO NGAY MOI NHAT TRUOC (ORDER BY published_date DESC) - day la 1 policy can duoc giai thich/danh gia trong bao cao (uu tien tin moi co the dung hoac sai tuy loai cau hoi).
- `textualize(edges)`: chuyen list canh thanh cac dong text dang "A [Type] -RELATION-> B [Type] | date=... | chunk=... | evidence=...", cat bot khi vuot `MAX_GRAPH_CONTEXT_CHARS`.
- `retrieve_graph_context(query, max_hops=2, edge_limit=50, return_debug=False)`: dung hang doi (deque) de BFS tu cac seed da match; voi moi node duoc "expand" thi kiem tra degree, neu vuot `SUPER_NODE_DEGREE` thi gioi han lai edge_limit ve toi da `SUPER_NODE_EDGE_CAP` (va ghi lai vao `supernode_events` de audit); dung lai khi dat `GLOBAL_EDGE_CAP` hoac het frontier. Ket qua tra ve gom context da textualize, DataFrame edges, va diagnostics (seed da match, so node da expand, so canh thu duoc, danh sach supernode_events).

### Buoc 3.4 - Sinh cau tra loi: Flat vs Hybrid GraphRAG

- `ANSWER_SYSTEM` yeu cau LLM chi tra loi tu context duoc cung cap, khong bay dat, phai trich dan provenance dang `[chunk_id=...]`, va phai noi ro neu bang chung khong du/mau thuan.
- `generate_answer(question, context)`: goi LLM sinh cau tra loi, do luon `latency_s` (thoi gian) va `total_tokens` (usage) - hai chi so nay se dung cho bang so sanh Phan 4.
- `answer_flat_rag(question)`: retrieve 6 chunk gan nhat bang Flat RAG roi sinh cau tra loi.
- `answer_graph_rag(question)`: day chinh la kien truc **Hybrid** - lay ca graph context (`retrieve_graph_context`, max_hops=2) VA vector context (4 chunk qua `retrieve_flat_context`), ghep 2 phan context lai ("=== GRAPH ===" va "=== VECTOR ===") roi moi sinh cau tra loi. Day la ly do lab goi la "Hybrid GraphRAG" chu khong phai graph-only.

**Nen lam ngay**: Thu chay `answer_flat_rag()` va `answer_graph_rag()` voi vai cau hoi mau de kiem tra pipeline chay end-to-end thong suot truoc khi qua Phan 4 (chay hang loat qua Golden Dataset se ton thoi gian/token, nen debug tung cau truoc).

---

## PHAN 4 - GOLDEN DATASET & LLM-AS-A-JUDGE (75-105 phut)

### Buoc 4.1 - Chuan bi Golden Dataset

Schema Golden Dataset gom: `id`, `group` (loai cau hoi: factoid/multi-hop/cross-doc), `question`, `reference_answer`, va tuy chon `reference_evidence`.

Notebook co san 5 cau hoi starter:
- G01 (factoid) - DA CO reference_answer mau.
- G02 (multi-hop) - "Startup nao duoc thanh lap boi cuu nhan vien Microsoft va sau do nhan von dau tu tu Google?" - CHUA co reference_answer, phai dien.
- G03 (cross-doc) - So sanh huong dau tu AI cua Meta va Apple trong 2023 tu nhieu bai bao - CHUA co reference_answer, phai dien.
- G04 (multi-hop) - Tim cong ty duoc dau tu boi mot cong ty cong nghe lon va cung phat trien mot cong nghe AI cu the - CHUA co reference_answer, phai dien.
- G05 (cross-doc) - Tim mot cong nghe lien quan cung mot cong ty qua it nhat 2 chunk khac nhau, tom tat thay doi theo thoi gian - CHUA co reference_answer, phai dien.

**Buoc bat buoc, KHONG duoc bo qua**: Voi 4 cau G02-G05, ban phai tu tra cuu trong chinh du lieu `chunks_df`/graph da nap (dung Cypher hoac tim kiem text) de dien `reference_answer` THAT (dung voi du lieu thuc te da load, khong bay dat), roi luu ra file `golden_dataset.csv` (duong dan `GOLDEN_PATH`). Ham `validate_golden(df, require_answers=True)` se raise loi va in ra danh sach cau con thieu reference_answer neu ban chua dien du - hay chay ham nay som de biet minh con thieu cau nao.

**Goi y**: Neu cau hoi mau (vi du G02) khong tim thay du lieu phu hop trong subset da lay mau ngau nhien (vi du khong co cong ty nao thoa dieu kien "tung lam Microsoft roi duoc Google dau tu"), ban duoc phep sua lai noi dung cau hoi cho phu hop voi du lieu thuc te dang co, miem la van giu dung LOAI cau hoi (factoid/multi-hop/cross-doc) va giai thich ly do thay doi trong phan thuyet minh.

### Buoc 4.2 - LLM-as-a-Judge

- `JUDGE_SYSTEM`: yeu cau judge cham nghiem khac tren thang 1-5 cho 3 tieu chi: `comprehensiveness` (do day du), `faithfulness` (do trung thuc voi context duoc cung cap - khong bay dat), `multi_hop_reasoning` (do chinh xac cua suy luan nhieu buoc), dung reference_answer lam moc dung/sai.
- `judge_json(system, user)`: linh hoat chon provider judge qua `JUDGE_PROVIDER` - neu la "groq" thi dung lai ham groq_json, neu la "openai" thi goi OpenAI Chat Completions voi `response_format=json_object`.
- `judge_answer(question, reference, answer, context)`: dong goi prompt judge day du (cau hoi, reference, candidate answer, candidate context toi da 18000 ky tu), ep diem ve khoang [1,5], tra ve dict 3 diem so + `rationale` (giai trinh 2-5 cau).
**Luu y**: judge nen dung MODEL KHAC hoac it nhat prompt doc lap voi model sinh cau tra loi de giam thien vi tu-danh-gia; neu dung OpenAI lam judge se ton chi phi rieng, kiem tra `OPENAI_API_KEY` truoc.

### Buoc 4.3 - Chay Evaluation + checkpoint

Ham `run_evaluation(golden_df)` lap qua tung cau trong Golden Dataset, chay CA `answer_flat_rag` VA `answer_graph_rag`, cham diem CA HAI bang judge, roi gop tat ca vao 1 dong ket qua (bao gom 3 diem judge, latency, token, so luong supernode_events, rationale) va LUU CHECKPOINT ra CSV sau MOI LAN CHAY XONG (khong phai cho het moi luu 1 lan) - de tranh mat toan bo ket qua neu Colab bi disconnect giua luc chay.

Truoc khi chay full: goi `validate_golden(golden_df, require_answers=True)` de chan lai neu con cau thieu reference_answer. Sau do moi bo comment chay `eval_results_df = run_evaluation(golden_df)`. Vi mot cau chay ca 2 kien truc + 2 lan judge (4 loi goi LLM/cau), voi 5 cau se ton khoang 20 loi goi LLM - can du kien thoi gian cho.

### Buoc 4.4 - Bang so sanh + export

Ham `comparison_table(eval_df)` group theo `group` (loai cau hoi), tinh trung binh 5 metric (Comprehensiveness, Faithfulness, Multi-hop reasoning, Latency, Token usage) cho ca Flat RAG va GraphRAG, VA TU DONG SINH NHAN XET: voi Latency/Token thi Flat RAG re/nhanh hon la binh thuong; voi 3 metric chat luong, neu GraphRAG hon >= 0.75 diem thi ghi nhan "cai thien ro", neu kem hon <= -0.5 diem thi ghi "Flat RAG tot hon, graph extraction/retrieval co the gay mat thong tin hoac nhieu", con lai la "hai phuong phap gan nhau".

Bo comment chay `comparison_df = comparison_table(eval_results_df)`, roi export 2 file bat buoc: `eval_results_df.to_csv("/content/graphrag_eval_results.csv")` va `comparison_df.to_csv("/content/graphrag_vs_flatrag_summary.csv")`. Day la 2 file can nop kem notebook.

---

## PHAN 5 - FAILURE-MODE CHECKS & SUBMISSION (105-120 phut)

Phan nay yeu cau chung minh bang code + so lieu (khong chi noi mieng) 4 dieu: edge khong thieu provenance, entity resolution co audit, super-node degree > 100 chi expand toi da 50 canh, va co bang comparison.

### Buoc 5.1 - Kiem tra Super-node policy + Entity audit

- `test_supernode_policy()`: tim node co degree cao nhat trong toan graph, goi `recent_edges` voi limit tuong ung (50 neu degree > `SUPER_NODE_DEGREE`, nguoc lai 1000), roi `assert` so canh lay ve <= 50 khi node do la super-node. Chay ham nay va CHUP LAI/GHI LAI ket qua in ra (ten node, degree, so canh fetch duoc) - day la bang chung nop bai.
- `show_resolution_audit(entity_resolution_audit_df)`: hien 30 dong co similarity cao nhat, va rieng 20 dong bi `REJECT_GUARD` co similarity cao nhat (nhung cap "nhin giong nhau ma khong duoc merge") - day la bang chung cho thay lexical guard hoat dong dung, rat quan trong de tra loi cau hoi "candidate nao similarity cao nhung khong nen merge".

### Buoc 5.2 - Thuyet minh ky thuat (BAT BUOC, chiem 20% diem)

Tu viet cau tra loi (bang van xuoi, co du lieu/so lieu minh chung, KHONG tra loi chung) cho 10 cau hoi sau va dua vao notebook hoac bao cao nop bai:

1. Coreference resolution sai trong tinh huong nao (vi du cu the tu `unresolved_mentions` hoac loi phat hien duoc)?
2. Entity resolution threshold ban chon la bao nhieu (0.90 mac dinh hay da doi), vi sao?
3. Co candidate entity nao similarity cao nhung KHONG nen merge (lay tu bang `REJECT_GUARD`)? Vi sao khong nen merge?
4. Top 3 super-node trong graph cua ban la gi, degree bao nhieu (lay tu `top_degree_df`)?
5. Chinh sach uu tien lay canh (edge) moi nhat (`ORDER BY published_date DESC`) co the dung hoac sai trong truong hop nao?
6. Flat RAG thang (diem cao hon) o nhom cau hoi nao (factoid/multi-hop/cross-doc)?
7. GraphRAG thang o nhom cau hoi nao?
8. Trade-off giua latency va token usage giua 2 kien truc la gi (lay so lieu tu comparison_df)?
9. Neu co dung AI Coding Agent: Agent da de xuat gi ma ban KHONG dung, vi sao?
10. Voi quy mo du lieu day du 350MB (khong con scale guard), bottleneck (nut co chai) dau tien se la gi (goi y: so luot goi LLM cho extraction, chi phi embedding, hay Neo4j write throughput)?

---

## PHAN BONUS (tuy chon, lam sau khi da hoan thanh Phan 1-5)

### Bonus A - Low-level / High-level retrieval

Tao thêm tang "high-level": community/topic report (tom tat theo cum entity) ben canh cac "local entity" hien co, roi dung mot query router de chon tang retrieval phu hop (cau hoi cu the -> local; cau hoi tong quan/xu huong -> high-level).

### Bonus B - Global Search qua Community Reports (fallback bang NetworkX)

Neu Neo4j instance khong co Graph Data Science (GDS) library phu hop, dung fallback: export toan bo edges ra pandas, dung NetworkX `greedy_modularity_communities` de phat hien cum (community detection), viet nguoc `community_id` vao Neo4j bang UNWIND theo batch (ham `build_communities(limit_edges=20000)`), roi dung LLM tom tat noi dung tung community, cuoi cung cho phep query "global" tren cac ban tom tat community nay thay vi tren tung node rieng le.

### Bonus C - Self-Correction Graph Retrieval

Y tuong: sau khi lay context o hop 2, dung mot LLM kiem tra (`context_sufficient`) xem context da du de tra loi chua; neu thieu thi tang len hop 3; neu van thieu thi fallback sang vector search bo sung. Bat buoc phai co dieu kien dung (stop condition) de tranh vong lap vo han - trong code khung, toi da chi den hop 3 roi bat buoc dung lai va bo sung vector.

---

## CHECKLIST NOP BAI (rieng cho ban tich khi hoan thanh)

- [ ] Neo4j da connect thanh cong (`connect_neo4j()` khong loi)
- [ ] Da chay dedup/chunking (`chunks_df` co du lieu)
- [ ] Da spot-check (kiem tra ngau nhien) ket qua coreference resolution
- [ ] Da co bang audit entity resolution (`entity_resolution_audit_df`)
- [ ] Da bulk insert node + edge bang UNWIND (khong insert tung dong)
- [ ] `graph_checks()` cho ket qua 0 canh thieu provenance
- [ ] Flat RAG da chay va tra loi duoc
- [ ] Hybrid GraphRAG da chay va tra loi duoc
- [ ] Da chay `test_supernode_policy()` va co bang chung cap 50 canh
- [ ] Golden Dataset da co du 5 reference_answer THAT (khong con "TO_BE_FILLED_FROM_DATASET")
- [ ] Da chay evaluation het toan bo Golden Dataset (co checkpoint CSV)
- [ ] Da export `graphrag_eval_results.csv` va `graphrag_vs_flatrag_summary.csv`
- [ ] Da viet day du phan thuyet minh ky thuat (10 cau hoi Phan 5.2)
- [ ] Neu lam Bonus: co so lieu dinh luong truoc/sau ro rang
- [ ] Da chuan bi link GitHub/Drive/LMS de nop bai

---

## RUBRIC CHAM DIEM (tom tat)

| Tieu chi | Trong so | Yeu cau |
|---|---|---|
| Chay duoc code | 30% | Graph nap thanh cong, schema dung, xuat duoc bang so sanh |
| Xu ly Failure modes | 30% | Xu ly duoc it nhat 2/3 van de: Super-node, Entity Resolution, Coreference |
| Evaluation | 20% | Chay het Golden Dataset, phan tich hop ly |
| Thuyet minh | 20% | Giai thich duoc kien truc va cach kiem soat AI Coding Agent |

---

## Ghi chu cuoi

Tai lieu nay la ban tong hop huong dan thao tac dua tren noi dung cong khai cua trang codelab (VLearn), nham giup ban di dung tung buoc trong Colab. Toan bo code chi tiet, day du va co the copy-paste truc tiep van nam trong notebook Colab goc cua khoa hoc - hay mo notebook do song song voi file huong dan nay khi thuc hanh. Ban nen doc lai muc "LUU Y QUAN TRONG" o dau file truoc khi bat dau tung phan.
