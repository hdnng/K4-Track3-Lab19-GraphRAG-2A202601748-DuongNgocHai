# Hướng dẫn chi tiết Lab: GraphRAG vs Flat RAG - Production-Grade Lab

> Track 3 - Lab 19 | Thời lượng: 120 phút | Mức độ: Trung cấp

> Nguồn: VLearn Codelabs (codelabs.vlearn.dev) - Tài liệu này là bản tổng hợp và hướng dẫn thao tác, không sao chép toàn bộ code trong notebook gốc. Học viên cần mở notebook Colab thật sự của lab để lấy code khung đầy đủ.

---

## LƯU Ý QUAN TRỌNG - ĐỌC KỸ TRƯỚC KHI BẮT ĐẦU

1. **Mục tiêu cốt lõi**: Xây pipeline Hybrid GraphRAG hoàn chỉnh trên Neo4j (chunk -> coreference resolution -> NER/RE -> entity resolution -> bulk insert bằng UNWIND -> graph traversal có kiểm soát super-node -> hybrid retrieval), rồi so sánh định lượng với Flat RAG (vector-only) bằng Golden Dataset và LLM-as-a-Judge (đo quality, latency, token usage).

2. **Notebook đã có code khung sẵn**. Nhiệm vụ chính KHÔNG phải viết lại toàn bộ pipeline, mà là: chạy code, chỉnh prompt/threshold/retrieval policy cho phù hợp, và thuyết minh (giải thích) lý do lựa chọn của mình. Đừng bỏ qua phần thuyết minh vì nó chiếm 20% điểm.

3. **Chuẩn bị Secrets TRƯỚC khi vào lab** (khai báo trong Colab Secrets, KHÔNG hard-code API key vào notebook):
   - `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (và tùy chọn `NEO4J_DATABASE`)
   - `GROQ_API_KEY`, `GROQ_MODEL`
   - `HF_TOKEN` để stream dataset từ Hugging Face
   - `JUDGE_PROVIDER` (openai hoặc groq), `JUDGE_MODEL`, và `OPENAI_API_KEY` nếu dùng OpenAI làm judge

4. **Dataset dùng trong lab**: `HackerNoon/tech-company-news-data-dump` trên Hugging Face. Nếu dataset yêu cầu gated access, phải vào trang dataset trên Hugging Face bấm Agree/Request access TRƯỚC khi chạy cell stream dữ liệu, nếu không sẽ bị lỗi khi load.

5. **Scale guard - RẤT QUAN TRỌNG**: Dataset gốc ~350MB, KHÔNG được gửi toàn bộ qua LLM trong 2 giờ lab. Mặc định dùng subset với các giới hạn:
   - `LAB_MAX_ARTICLES = 1500` (số bài báo tối đa sau dedup)
   - `LAB_MAX_CHUNKS = 3000` (số chunk tối đa đưa vào Flat RAG index)
   - `EXTRACTION_MAX_CHUNKS = 400` (số chunk tối đa đưa qua LLM để trích xuất triple)
   - `LIMIT_MB = 300` khi stream dataset từ Hugging Face (có thể tăng sau buổi học)
   Kiến trúc phải thể hiện khả năng scale được, còn volume trong giờ lab chỉ để chứng minh pipeline chạy đúng.

6. **3 failure-mode trọng tâm của lab** (bắt buộc xử lý ít nhất 2/3 để đạt rubric 30%):
   - **Coreference sai** -> tạo ra false edge (cạnh sai trong graph). Nguyên tắc: chỉ resolve đại từ khi antecedent rõ ràng trong CÙNG một chunk, không bao giờ bịa đặt (invent) fact, giữ nguyên số liệu/ngày/ticker/tên sản phẩm, nếu còn nghi ngờ (ambiguous) thì giữ nguyên và log vào `unresolved_mentions`.
   - **Entity Resolution sai** -> merge nhầm hai entity khác nhau (ví dụ hai công ty tên giống nhau nhưng khác nhau) hoặc không merge được các biến thể của cùng 1 entity (viết tắt, ticker, alias). Cần có "audit table" ghi lại mọi quyết định merge/reject và similarity score.
   - **Super-node** -> một entity có quá nhiều cạnh (degree lớn, ví dụ công ty lớn như Microsoft/Google) sẽ làm "nổ" context khi traversal. Cần có ngưỡng `SUPER_NODE_DEGREE` và giới hạn `SUPER_NODE_EDGE_CAP` khi expand.

7. **Rubric chấm điểm** (bám sát khi làm bài):
   - 30% Chạy được code: graph nạp thành công, schema đúng, xuất được bảng so sánh.
   - 30% Failure modes: xử lý được ít nhất 2/3 vấn đề Super-node, Entity Resolution, Coreference.
   - 20% Evaluation: chạy hết Golden Dataset, phân tích hợp lý.
   - 20% Thuyết minh: giải thích được kiến trúc và cách kiểm soát AI Coding Agent (nếu dùng).

8. **Golden Dataset**: Notebook chỉ có sẵn 5 câu hỏi mẫu (starter), trong đó CHỈ CÓ 1 câu (G01) có sẵn reference_answer. 4 câu còn lại (G02-G05) phải được học viên tự điền `reference_answer` thực tế dựa trên dữ liệu đã nạp vào Neo4j/Flat index TRƯỚC khi chạy evaluation cuối cùng - đây là bước dễ bỏ qua trong khi làm bài, nên làm sớm.

9. **Nếu dùng AI Coding Agent** (Cursor/Copilot/ChatGPT...) để hỗ trợ code trong 2 "Challenge" (Near Dedup và Entity Resolution guard) hoặc phần Bonus, phải nêu rõ trong báo cáo: threshold đã chọn, false positive gặp phải, cách audit các cặp bị merge, và những đề xuất của Agent mà bạn KHÔNG dùng kèm lý do - đây là một mục trong phần thuyết minh 20%.

10. **Nộp bài**: Chuẩn bị sẵn link GitHub/Drive/LMS chứa notebook đã chạy xong (có output cell, không cần chạy lại từ đầu khi nộp) và bảng so sánh CSV.

---

## Tổng quan bài Lab

Học viên xây dựng một pipeline Hybrid GraphRAG hoàn chỉnh trên Neo4j, xử lý các bài toán thực tế (coreference, entity resolution, super-node), rồi so sánh định lượng với Flat RAG bằng Golden Dataset và LLM-as-a-Judge.

Kiến thức/kỹ năng cần có trước khi làm lab: Python và pandas ở mức cơ bản; khái niệm RAG, embedding và vector search; Cypher/Neo4j ở mức cơ bản (không bắt buộc vì đã có code khung); cách gọi LLM API (Groq/OpenAI) và xử lý JSON output.

### Timeline đề xuất (tổng 120 phút)

| Thời gian | Nội dung |
|---|---|
| 00-15 phút | Setup & Preprocessing: cài môi trường, load dữ liệu, dedup, chunk, coreference resolution |
| 15-45 phút | Triple Extraction & Neo4j Bulk Insert: NER/RE, entity resolution bằng vector similarity, bulk insert bằng UNWIND |
| 45-75 phút | Flat RAG & Hybrid GraphRAG: xây Flat RAG baseline, graph traversal có super-node mitigation, hybrid retrieval |
| 75-105 phút | Golden Dataset & LLM-as-a-Judge: tạo Golden Dataset, chấm điểm bằng LLM-as-a-Judge, xuất bảng so sánh hai kiến trúc |
| 105-120 phút | Failure-mode Checks & Submission: kiểm tra failure mode, làm bonus (tùy chọn), export kết quả, thuyết minh kỹ thuật |

### Kết thúc bài lab, bạn sẽ có

- Xây dựng được pipeline Hybrid GraphRAG end-to-end trên Neo4j bằng bulk insert UNWIND.
- Xử lý được Coreference Resolution, Entity Resolution và Super-node Mitigation trong thực tế.
- So sánh định lượng Flat RAG và GraphRAG bằng Golden Dataset + LLM-as-a-Judge.
- Giải thích được kiến trúc, trade-off latency/token và failure modes của hệ thống RAG.

---

## PHẦN 1 - SETUP & PREPROCESSING (0-15 phút)

### Bước 1.0 - Tạo Secrets trên Google Colab

Vào biểu tượng chìa khóa (Secrets) bên trái Colab, tạo các secret sau: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `GROQ_API_KEY`, `GROQ_MODEL`, `HF_TOKEN`, `JUDGE_PROVIDER`, `JUDGE_MODEL`, `OPENAI_API_KEY` (nếu judge là OpenAI). Nhớ bật toggle "Notebook access" cho từng secret. Tuyệt đối không dán API key trực tiếp vào code cell.

### Bước 1.1 - Install thư viện

Chạy cell pip install các package: neo4j, pandas, numpy, pyarrow, sentence-transformers, faiss-cpu, groq, openai, tqdm, networkx, spacy, datasets, langchain-community, llama-index. Đợi cell chạy xong hoàn toàn (có thể mất 1-2 phút) trước khi qua bước sau.

### Bước 1.2 - Import và khai báo config

Chạy cell import (os, re, json, pandas, numpy, neo4j driver, sentence-transformers, faiss...) và set SEED=42 để kết quả reproducible. Hàm `get_secret()` sẽ đọc secret từ Colab userdata trước, nếu không có thì fallback sang biến môi trường - kiểm tra hàm này chạy không lỗi tức là secrets đã được đọc đúng.

Kiểm tra lại các hằng số scale guard đã được set đúng: `LAB_MAX_ARTICLES=1500`, `LAB_MAX_CHUNKS=3000`, `EXTRACTION_MAX_CHUNKS=400`, `CHUNK_WORDS=220`, `CHUNK_OVERLAP_WORDS=40`.

### Bước 1.3 - Download dataset HackerNoon bằng Hugging Face Streaming

Cell này stream trực tiếp dataset `HackerNoon/tech-company-news-data-dump` và ghi dần ra file CSV `/content/hackernoon_subset.csv`, không tải toàn bộ dataset vào RAM.

- Nếu dataset yêu cầu gated access: mở trang dataset trên Hugging Face, đăng nhập và bấm Agree/Request access TRƯỚC, nếu không cell sẽ lỗi authorization.
- Hai cơ chế dừng: `LIMIT_ROWS` (số dòng tối đa) và `LIMIT_MB` (dung lượng file tối đa, mặc định 300MB). Biến `PRIORITIZE_MB=True` nghĩa là ưu tiên dừng theo dung lượng MB; `False` nghĩa là progress bar theo số dòng nhưng vẫn giữ hard-stop LIMIT_ROWS.
- Chạy cell và chờ đến khi thấy dòng "Hoàn thành" với số rows và dung lượng file. Sau khi chạy xong, biến `DATA_PATH` đã tự động trỏ về file CSV vừa tạo, cell loader kế tiếp chạy trực tiếp được.

Nếu gặp lỗi: kiểm tra lại (1) HF_TOKEN đã đúng chưa, (2) đã Agree/Request access trên Hugging Face chưa, (3) kết nối mạng của Colab.

### Bước 1.4 - Kết nối Neo4j và tạo schema

Gọi hàm `connect_neo4j()` để kết nối driver Neo4j (dùng Aura hoặc instance tự host) - sẽ in ra "Neo4j connected" nếu thành công. Sau đó gọi `setup_graph_schema()` để tạo constraint unique cho `Entity.id` và index cho `name_norm` của Entity/Company/Person/Technology. Hai lệnh này đang bị comment (#) trong code khung - cần bỏ comment và chạy trước khi qua Phần 2.

### Bước 1.5 - Load dữ liệu, exact dedup và chunking

- `load_news(DATA_PATH)`: đọc file CSV/JSON/Parquet thành DataFrame.
- `standardize_news(raw_df)`: tự động nhận diện cột text/title/date/id (hỗ trợ nhiều tên cột khác nhau), chuẩn hóa khoảng trắng, loại bài quá ngắn (<80 ký tự), tính hash `dedup_key` để loại bài trùng lặp tuyệt đối (exact dedup), và lấy mẫu (sample) tối đa `LAB_MAX_ARTICLES` bài theo SEED cố định để reproducible.
- `build_chunks(news_df)`: cắt mỗi bài báo thành các chunk ~220 từ, overlap 40 từ (sliding window), đặt `chunk_id` dạng `{article_id}::c{index}`, và dừng lại khi vượt `LAB_MAX_CHUNKS`.
- Bỏ comment 4 dòng cuối cell này (`raw_df = load_news(...)`, `news_df = standardize_news(...)`, `chunks_df = build_chunks(...)`, `display(chunks_df.head())`) và chạy để tạo `chunks_df` - đây là đầu vào cho tất cả các bước sau.

**Challenge A (AI Coding Agent - tùy chọn nhưng nên làm để cộng điểm thuyết minh)**: Exact hash dedup chỉ bắt được bản sao chính xác, KHÔNG bắt được bài bị "repost"/gần trùng (near-duplicate). Hãy dùng AI Agent thiết kế thêm MinHash/LSH, SimHash, hoặc embedding+ANN để phát hiện near-duplicate. Không được chấp nhận cách làm pairwise cosine O(N^2) trên toàn dataset (quá chậm). Trong báo cáo cần nêu: threshold đã chọn, tỷ lệ false positive quan sát được, và cách bạn audit (kiểm tra lại) các cặp bị merge.

### Bước 1.6 - LLM wrapper có retry và JSON parsing

Cell này tạo Groq client và 2 hàm quan trọng dùng xuyên suốt lab: `parse_json_object(text)` (bóc tách JSON object từ output LLM, tự động bỏ code-fence markdown) và `groq_chat(...)`/`groq_json(...)` (gọi Groq API với retry có exponential backoff tối đa 4 lần, temperature=0.0 để ổn định, hỗ trợ `json_mode=True`). Không cần sửa gì ở bước này, chỉ cần chạy để các bước sau sử dụng.

### Bước 1.7 - Coreference Resolution

Đây là bước "chuẩn hóa" văn bản trước khi trích xuất, cực kỳ quan trọng vì coreference sai sẽ tạo ra cạnh (edge) sai trong graph sau này.

Nguyên tắc bắt buộc của prompt coreference (đã được thiết kế sẵn trong `COREF_SYSTEM`): chỉ resolve đại từ/tham chiếu khi antecedent được hỗ trợ rõ ràng trong CÙNG một chunk; không được bịa đặt (invent) fact mới; phải giữ nguyên ngày tháng, số liệu, mã ticker và tên sản phẩm; nếu còn nghi ngờ (ambiguous) thì giữ nguyên text gốc và log vào mảng `unresolved_mentions`.

- Hàm `resolve_coref_batch(batch_df)` gửi từng batch nhỏ (mặc định 5 chunk/batch) qua LLM và trả về `resolved_text` cùng `unresolved_mentions`.
- Hàm `run_coref(chunks_subset, batch_size=5)` lặp qua toàn bộ subset, nếu 1 batch lỗi thì fallback giữ nguyên text gốc và gắn flag `COREF_BATCH_FAILED` để không làm sập cả pipeline.
- Bỏ comment 3 dòng cuối: lấy `extraction_source = chunks_df.head(EXTRACTION_MAX_CHUNKS)`, chạy `coref_df = run_coref(extraction_source)`, rồi merge kết quả vào `extraction_source`. Đây là dữ liệu đầu vào cho bước trích xuất triple ở Phần 2.
- **Nên làm**: sau khi chạy xong, xem qua vài dòng `unresolved_mentions` khác rỗng để hiểu các trường hợp LLM từ chối resolve - sẽ cần khi trả lời câu hỏi thuyết minh ở Phần 5.

---

## PHẦN 2 - TRIPLE EXTRACTION & NEO4J BULK INSERT (15-45 phút)

### Graph schema cần tuân thủ

- Node types (có label gốc `Entity`): `Company`, `Person`, `Technology`.
- Relation types được phép (allowlist): `ACQUIRED`, `DEVELOPED`, `INVESTED_IN`, `FOUNDED`, `WORKED_AT`, `PARTNERED_WITH`, `USES`, `LEADS`.
- Mọi edge bắt buộc phải có `source_chunk_id` và `published_date` (provenance - để biết thông tin lấy từ đâu); khuyến nghị thêm `evidence` (câu trích dẫn) và `confidence` (độ tin cậy).
- Relation type sinh ra từ LLM phải được lọc qua allowlist trước khi ghép vào câu lệnh Cypher, tuyệt đối không nối trực tiếp string từ LLM vào Cypher để tránh injection và relation "lạ".

### Bước 2.1 - NER + RE Extraction (trích xuất thực thể và quan hệ)

- `EXTRACT_SYSTEM` yêu cầu LLM chỉ trích xuất quan hệ thuộc allowlist, ưu tiên precision hơn recall (chỉ lấy quan hệ chắc chắn, bỏ qua nếu không rõ), và mỗi quan hệ phải có đoạn evidence ngắn kèm theo.
- Hàm `extract_batch(batch_df)` gửi batch chunk (dùng `resolved_text` từ bước coref nếu có, fallback text gốc) qua LLM, yêu cầu trả về JSON có cấu trúc `items -> relations` gồm source/source_type/relation/target/target_type/evidence/confidence.
- Hàm `run_extraction(source_df, batch_size=4)` lặp qua toàn bộ extraction_source, lọc bỏ các quan hệ không hợp lệ (source/target rỗng, type không thuộc allowlist, relation không thuộc allowlist), gom thành `raw_triples_df`; batch lỗi được ghi vào `errors_df` để audit.
- Bỏ comment cuối cell, chạy `raw_triples_df, extraction_errors_df = run_extraction(extraction_source)` và xem `raw_triples_df.head()`. Nếu `extraction_errors_df` có nhiều dòng, cần xem lại batch_size hoặc rate limit của Groq.

### Bước 2.2 - Entity Resolution bằng Vector Similarity

Mục tiêu: gộp các cách viết khác nhau của CÙNG một thực thể (ví dụ "MSFT", "Microsoft Corp", "Microsoft Corporation" -> "Microsoft") nhưng KHÔNG gộp nhầm hai thực thể khác nhau. Pipeline gồm 4 bước:

1. **Manual aliases**: bảng `MANUAL_ALIASES` cho các ticker/tên rất phổ biến (msft, googl, aapl...) - map thẳng về tên chuẩn, ưu tiên cao nhất.
2. **Embedding ANN candidate**: dùng `sentence-transformers/all-MiniLM-L6-v2` encode tên entity, dùng FAISS IndexFlatIP tìm top-k láng giềng gần nhất theo cosine similarity, mặc định `threshold=0.90`.
3. **Lexical guard** (`merge_guard`): sau khi vượt threshold embedding, còn phải qua kiểm tra lexical (bỏ suffix Inc/Corp/Ltd... rồi so sánh chuỗi bằng SequenceMatcher ratio >= 0.72) mới được merge thực sự - đây là lớp bảo vệ giảm false merge; cặp nào vượt threshold embedding nhưng không qua guard sẽ bị đánh dấu `REJECT_GUARD`.
4. **Union-Find (UF)** để gộp nhóm các entity đã được phép merge, chọn tên canonical là tên xuất hiện nhiều nhất (hoặc ngắn/gọn nhất khi hòa).
5. Xuất ra `entity_resolution_audit_df` ghi lại mọi cặp (left, right, similarity, decision: MERGE_MANUAL/MERGE_VECTOR/REJECT_GUARD) - bảng này là bằng chứng bắt buộc phải nộp.

- Bỏ comment chạy: `entity_map, entity_resolution_audit_df = build_resolution_map(raw_triples_df)` rồi `triples_df = canonicalize_triples(raw_triples_df, entity_map)`. Xem `entity_resolution_audit_df.head(20)` để kiểm tra các quyết định merge có hợp lý không.
- **Nên làm**: thử đổi `threshold` (ví dụ 0.85 hoặc 0.95) và quan sát số lượng MERGE_VECTOR/REJECT_GUARD thay đổi ra sao - đây chính là dữ liệu cần cho phần thuyết minh "entity threshold bao nhiêu, vì sao".

**Challenge B (AI Coding Agent - tùy chọn)**: Cải tiến lexical guard để xử lý tốt hơn các trường hợp: ticker (ví dụ TSLA vs Tesla), suffix công ty (Inc./Corp./Ltd.), sản phẩm có chứa tên công ty (ví dụ "Google Search" không được merge với "Google" ở mức entity Company nếu thật sự là 2 loại khác nhau), và người trùng họ/tên gần giống nhau (không được merge nhầm 2 người khác nhau).

### Bước 2.3 - Bảng Node và Bulk Insert bằng UNWIND

- `build_nodes(triples_df)`: gom tất cả source/target thành một bảng node duy nhất, group theo (id, name, name_norm, type), gộp hết các biến thể (alias) từng xuất hiện thành danh sách `aliases`/`aliases_norm` lưu trên node - giúp sau này match seed entity nhanh hơn.
- `bulk_insert_nodes(nodes_df, batch_size=1000)`: VỚI TỪNG LOẠI NODE (Company/Person/Technology), dùng `UNWIND $rows AS row MERGE (n:Entity {id: row.id}) SET n:Type, ...` để insert theo lô (batch 1000 dòng/lần) - đây chính là yêu cầu bắt buộc "bulk insert bằng UNWIND, không insert từng row" của lab.
- `bulk_insert_edges(triples_df, batch_size=1000)`: kiểm tra trước rằng dữ liệu có đủ cột `source_chunk_id` và `published_date` (provenance bắt buộc), sau đó với từng loại relation, dùng `UNWIND $rows AS row MATCH (s)...MATCH (t)...MERGE (s)-[r:REL {source_chunk_id: row.source_chunk_id}]->(t) SET r.published_date=..., r.evidence=..., r.confidence=...` theo batch.
- Bỏ comment 3 dòng cuối, chạy theo đúng thứ tự: `nodes_df = build_nodes(triples_df)` -> `bulk_insert_nodes(nodes_df)` -> `bulk_insert_edges(triples_df)`.

### Bước 2.4 - Sanity checks (kiểm tra sức khỏe dữ liệu)

Hàm `graph_checks()` chạy 3 kiểm tra: (1) đếm số cạnh bị thiếu provenance (`source_chunk_id` hoặc `published_date` null) - PHẢI BẰNG 0, có `assert invalid == 0` để chặn lab nếu sai; (2) tổng số node và edge trong graph; (3) top 15 node có degree (số cạnh) cao nhất - đây chính là danh sách ứng viên "super-node" sẽ dùng ở Phần 3 và Phần 5.

Bỏ comment chạy `graph_counts, top_degree_df = graph_checks()` và lưu lại bảng `top_degree_df` để đối chiếu với bước kiểm tra super-node sau này.

---

## PHẦN 3 - FLAT RAG & HYBRID GRAPHRAG (45-75 phút)

Nguyên tắc so sánh công bằng: cả hai kiến trúc dùng CÙNG một embedding model và CÙNG một generator LLM, chỉ khác nhau về CÁCH RETRIEVE context - để phân tích tập trung vào kiến trúc retrieval, không bị nhiễu bởi model khác nhau.

### Bước 3.1 - Xây Flat RAG baseline

- `build_flat_index(chunks_df)`: encode toàn bộ `chunks_df.text` bằng sentence-transformers (normalize_embeddings=True), đưa vào FAISS `IndexFlatIP` (inner product = cosine trên vector đã normalize).
- `retrieve_flat_context(query, k=6)`: encode câu hỏi, tìm k chunk gần nhất, trả về context đã ghép chuỗi kèm `chunk_id`, `published_date`, `score` (để trace được provenance) và DataFrame kết quả.
- Bỏ comment chạy `build_flat_index(chunks_df)`.

### Luồng xử lý của Graph Retrieval (tổng quan trước khi đọc code)

1. LLM trích xuất "seed entities" từ câu hỏi.
2. Tìm seed đó trong Neo4j (khớp chính xác tên/alias trước), nếu không có thì fallback bằng embedding fuzzy match.
3. BFS (duyệt theo chiều rộng) từ các seed, tối đa `max_hops` bước.
4. Nếu một node có degree > 100 (super-node) thì chỉ lấy tối đa 50 cạnh MỚI NHẤT của node đó, không lấy hết.
5. Có giới hạn tổng số cạnh toàn cục (global edge cap) để tránh "nổ" context (context explosion).
6. Chuyển subgraph thu được thành text có kèm provenance (ngày, chunk_id, evidence) để đưa vào prompt.

### Bước 3.2 - Seed matching (khớp thực thể hạt giống)

- `extract_seeds(query)`: gọi LLM (`SEED_SYSTEM` yêu cầu KHÔNG trả lời câu hỏi, chỉ trích seed entity thuộc 3 loại Company/Person/Technology) để lấy danh sách tên thực thể tiềm năng từ câu hỏi người dùng.
- `build_entity_matcher(nodes_df)`: encode trước tên tất cả node trong graph thành vector, lưu vào `entity_match_vectors`/`entity_match_store` để tra cứu nhanh (chạy 1 lần sau khi có `nodes_df`).
- `match_seeds(query, fuzzy_threshold=0.66)`: với mỗi seed, thử tìm EXACT match trong Neo4j qua `name_norm`/`aliases_norm` trước; nếu không có kết quả mới fallback sang so sánh embedding (cosine) với ngưỡng 0.66 và lấy node có similarity cao nhất.
- Bỏ comment chạy `build_entity_matcher(nodes_df)` sau khi đã có `nodes_df` từ Phần 2.

### Bước 3.3 - Graph traversal + Super-node mitigation

Các tham số kiểm soát (có thể tinh chỉnh và giải thích trong báo cáo):
- `SUPER_NODE_DEGREE = 100` - ngưỡng để coi một node là "super-node".
- `SUPER_NODE_EDGE_CAP = 50` - số cạnh tối đa được lấy từ một super-node.
- `GLOBAL_EDGE_CAP = 250` - tổng số cạnh tối đa thu thập cho toàn bộ quá trình traversal của 1 câu hỏi.
- `MAX_GRAPH_CONTEXT_CHARS = 14000` - độ dài tối đa (ký tự) của context dạng text đưa vào prompt.

- `node_degree(node_id)`: đếm số cạnh (kể cả 2 chiều) của một node bằng Cypher.
- `recent_edges(node_id, limit)`: lấy tối đa `limit` cạnh, SẮP XẾP THEO NGÀY MỚI NHẤT TRƯỚC (ORDER BY published_date DESC) - đây là 1 policy cần được giải thích/đánh giá trong báo cáo (ưu tiên tin mới có thể đúng hoặc sai tùy loại câu hỏi).
- `textualize(edges)`: chuyển list cạnh thành các dòng text dạng "A [Type] -RELATION-> B [Type] | date=... | chunk=... | evidence=...", cắt bớt khi vượt `MAX_GRAPH_CONTEXT_CHARS`.
- `retrieve_graph_context(query, max_hops=2, edge_limit=50, return_debug=False)`: dùng hàng đợi (deque) để BFS từ các seed đã match; với mỗi node được "expand" thì kiểm tra degree, nếu vượt `SUPER_NODE_DEGREE` thì giới hạn lại edge_limit về tối đa `SUPER_NODE_EDGE_CAP` (và ghi lại vào `supernode_events` để audit); dừng lại khi đạt `GLOBAL_EDGE_CAP` hoặc hết frontier. Kết quả trả về gồm context đã textualize, DataFrame edges, và diagnostics (seed đã match, số node đã expand, số cạnh thu được, danh sách supernode_events).

### Bước 3.4 - Sinh câu trả lời: Flat vs Hybrid GraphRAG

- `ANSWER_SYSTEM` yêu cầu LLM chỉ trả lời từ context được cung cấp, không bịa đặt, phải trích dẫn provenance dạng `[chunk_id=...]`, và phải nói rõ nếu bằng chứng không đủ/mâu thuẫn.
- `generate_answer(question, context)`: gọi LLM sinh câu trả lời, đo luôn `latency_s` (thời gian) và `total_tokens` (usage) - hai chỉ số này sẽ dùng cho bảng so sánh Phần 4.
- `answer_flat_rag(question)`: retrieve 6 chunk gần nhất bằng Flat RAG rồi sinh câu trả lời.
- `answer_graph_rag(question)`: đây chính là kiến trúc **Hybrid** - lấy cả graph context (`retrieve_graph_context`, max_hops=2) VÀ vector context (4 chunk qua `retrieve_flat_context`), ghép 2 phần context lại ("=== GRAPH ===" và "=== VECTOR ===") rồi mới sinh câu trả lời. Đây là lý do lab gọi là "Hybrid GraphRAG" chứ không phải graph-only.

**Nên làm ngay**: Thử chạy `answer_flat_rag()` và `answer_graph_rag()` với vài câu hỏi mẫu để kiểm tra pipeline chạy end-to-end thông suốt trước khi qua Phần 4 (chạy hàng loạt qua Golden Dataset sẽ tốn thời gian/token, nên debug từng câu trước).

---

## PHẦN 4 - GOLDEN DATASET & LLM-AS-A-JUDGE (75-105 phút)

### Bước 4.1 - Chuẩn bị Golden Dataset

Schema Golden Dataset gồm: `id`, `group` (loại câu hỏi: factoid/multi-hop/cross-doc), `question`, `reference_answer`, và tùy chọn `reference_evidence`.

Notebook có sẵn 5 câu hỏi starter:
- G01 (factoid) - ĐÃ CÓ reference_answer mẫu.
- G02 (multi-hop) - "Startup nào được thành lập bởi cựu nhân viên Microsoft và sau đó nhận vốn đầu tư từ Google?" - CHƯA có reference_answer, phải điền.
- G03 (cross-doc) - So sánh hướng đầu tư AI của Meta và Apple trong 2023 từ nhiều bài báo - CHƯA có reference_answer, phải điền.
- G04 (multi-hop) - Tìm công ty được đầu tư bởi một công ty công nghệ lớn và cũng phát triển một công nghệ AI cụ thể - CHƯA có reference_answer, phải điền.
- G05 (cross-doc) - Tìm một công nghệ liên quan cùng một công ty qua ít nhất 2 chunk khác nhau, tóm tắt thay đổi theo thời gian - CHƯA có reference_answer, phải điền.

**Bước bắt buộc, KHÔNG được bỏ qua**: Với 4 câu G02-G05, bạn phải tự tra cứu trong chính dữ liệu `chunks_df`/graph đã nạp (dùng Cypher hoặc tìm kiếm text) để điền `reference_answer` THẬT (đúng với dữ liệu thực tế đã load, không bịa đặt), rồi lưu ra file `golden_dataset.csv` (đường dẫn `GOLDEN_PATH`). Hàm `validate_golden(df, require_answers=True)` sẽ raise lỗi và in ra danh sách câu còn thiếu reference_answer nếu bạn chưa điền đủ - hãy chạy hàm này sớm để biết mình còn thiếu câu nào.

**Gợi ý**: Nếu câu hỏi mẫu (ví dụ G02) không tìm thấy dữ liệu phù hợp trong subset đã lấy mẫu ngẫu nhiên (ví dụ không có công ty nào thỏa điều kiện "từng làm Microsoft rồi được Google đầu tư"), bạn được phép sửa lại nội dung câu hỏi cho phù hợp với dữ liệu thực tế đang có, miễn là vẫn giữ đúng LOẠI câu hỏi (factoid/multi-hop/cross-doc) và giải thích lý do thay đổi trong phần thuyết minh.

### Bước 4.2 - LLM-as-a-Judge

- `JUDGE_SYSTEM`: yêu cầu judge chấm nghiêm khắc trên thang 1-5 cho 3 tiêu chí: `comprehensiveness` (độ đầy đủ), `faithfulness` (độ trung thực với context được cung cấp - không bịa đặt), `multi_hop_reasoning` (độ chính xác của suy luận nhiều bước), dùng reference_answer làm mốc đúng/sai.
- `judge_json(system, user)`: linh hoạt chọn provider judge qua `JUDGE_PROVIDER` - nếu là "groq" thì dùng lại hàm groq_json, nếu là "openai" thì gọi OpenAI Chat Completions với `response_format=json_object`.
- `judge_answer(question, reference, answer, context)`: đóng gói prompt judge đầy đủ (câu hỏi, reference, candidate answer, candidate context tối đa 18000 ký tự), ép điểm về khoảng [1,5], trả về dict 3 điểm số + `rationale` (giải trình 2-5 câu).
**Lưu ý**: judge nên dùng MODEL KHÁC hoặc ít nhất prompt độc lập với model sinh câu trả lời để giảm thiên vị tự-đánh-giá; nếu dùng OpenAI làm judge sẽ tốn chi phí riêng, kiểm tra `OPENAI_API_KEY` trước.

### Bước 4.3 - Chạy Evaluation + checkpoint

Hàm `run_evaluation(golden_df)` lặp qua từng câu trong Golden Dataset, chạy CẢ `answer_flat_rag` VÀ `answer_graph_rag`, chấm điểm CẢ HAI bằng judge, rồi gộp tất cả vào 1 dòng kết quả (bao gồm 3 điểm judge, latency, token, số lượng supernode_events, rationale) và LƯU CHECKPOINT ra CSV sau MỖI LẦN CHẠY XONG (không phải chờ hết mới lưu 1 lần) - để tránh mất toàn bộ kết quả nếu Colab bị disconnect giữa lúc chạy.

Trước khi chạy full: gọi `validate_golden(golden_df, require_answers=True)` để chặn lại nếu còn câu thiếu reference_answer. Sau đó mới bỏ comment chạy `eval_results_df = run_evaluation(golden_df)`. Vì một câu chạy cả 2 kiến trúc + 2 lần judge (4 lời gọi LLM/câu), với 5 câu sẽ tốn khoảng 20 lời gọi LLM - cần dự kiến thời gian chờ.

### Bước 4.4 - Bảng so sánh + export

Hàm `comparison_table(eval_df)` group theo `group` (loại câu hỏi), tính trung bình 5 metric (Comprehensiveness, Faithfulness, Multi-hop reasoning, Latency, Token usage) cho cả Flat RAG và GraphRAG, VÀ TỰ ĐỘNG SINH NHẬN XÉT: với Latency/Token thì Flat RAG rẻ/nhanh hơn là bình thường; với 3 metric chất lượng, nếu GraphRAG hơn >= 0.75 điểm thì ghi nhận "cải thiện rõ", nếu kém hơn <= -0.5 điểm thì ghi "Flat RAG tốt hơn, graph extraction/retrieval có thể gây mất thông tin hoặc nhiễu", còn lại là "hai phương pháp gần nhau".

Bỏ comment chạy `comparison_df = comparison_table(eval_results_df)`, rồi export 2 file bắt buộc: `eval_results_df.to_csv("/content/graphrag_eval_results.csv")` và `comparison_df.to_csv("/content/graphrag_vs_flatrag_summary.csv")`. Đây là 2 file cần nộp kèm notebook.

---

## PHẦN 5 - FAILURE-MODE CHECKS & SUBMISSION (105-120 phút)

Phần này yêu cầu chứng minh bằng code + số liệu (không chỉ nói miệng) 4 điều: edge không thiếu provenance, entity resolution có audit, super-node degree > 100 chỉ expand tối đa 50 cạnh, và có bảng comparison.

### Bước 5.1 - Kiểm tra Super-node policy + Entity audit

- `test_supernode_policy()`: tìm node có degree cao nhất trong toàn graph, gọi `recent_edges` với limit tương ứng (50 nếu degree > `SUPER_NODE_DEGREE`, ngược lại 1000), rồi `assert` số cạnh lấy về <= 50 khi node đó là super-node. Chạy hàm này và CHỤP LẠI/GHI LẠI kết quả in ra (tên node, degree, số cạnh fetch được) - đây là bằng chứng nộp bài.
- `show_resolution_audit(entity_resolution_audit_df)`: hiện 30 dòng có similarity cao nhất, và riêng 20 dòng bị `REJECT_GUARD` có similarity cao nhất (những cặp "nhìn giống nhau mà không được merge") - đây là bằng chứng cho thấy lexical guard hoạt động đúng, rất quan trọng để trả lời câu hỏi "candidate nào similarity cao nhưng không nên merge".

### Bước 5.2 - Thuyết minh kỹ thuật (BẮT BUỘC, chiếm 20% điểm)

Tự viết câu trả lời (bằng văn xuôi, có dữ liệu/số liệu minh chứng, KHÔNG trả lời chung chung) cho 10 câu hỏi sau và đưa vào notebook hoặc báo cáo nộp bài:

1. Coreference resolution sai trong tình huống nào (ví dụ cụ thể từ `unresolved_mentions` hoặc lỗi phát hiện được)?
2. Entity resolution threshold bạn chọn là bao nhiêu (0.90 mặc định hay đã đổi), vì sao?
3. Có candidate entity nào similarity cao nhưng KHÔNG nên merge (lấy từ bảng `REJECT_GUARD`)? Vì sao không nên merge?
4. Top 3 super-node trong graph của bạn là gì, degree bao nhiêu (lấy từ `top_degree_df`)?
5. Chính sách ưu tiên lấy cạnh (edge) mới nhất (`ORDER BY published_date DESC`) có thể đúng hoặc sai trong trường hợp nào?
6. Flat RAG thắng (điểm cao hơn) ở nhóm câu hỏi nào (factoid/multi-hop/cross-doc)?
7. GraphRAG thắng ở nhóm câu hỏi nào?
8. Trade-off giữa latency và token usage giữa 2 kiến trúc là gì (lấy số liệu từ comparison_df)?
9. Nếu có dùng AI Coding Agent: Agent đã đề xuất gì mà bạn KHÔNG dùng, vì sao?
10. Với quy mô dữ liệu đầy đủ 350MB (không còn scale guard), bottleneck (nút cổ chai) đầu tiên sẽ là gì (gợi ý: số lượt gọi LLM cho extraction, chi phí embedding, hay Neo4j write throughput)?

---

## PHẦN BONUS (tùy chọn, làm sau khi đã hoàn thành Phần 1-5)

### Bonus A - Low-level / High-level retrieval

Tạo thêm tầng "high-level": community/topic report (tóm tắt theo cụm entity) bên cạnh các "local entity" hiện có, rồi dùng một query router để chọn tầng retrieval phù hợp (câu hỏi cụ thể -> local; câu hỏi tổng quan/xu hướng -> high-level).

### Bonus B - Global Search qua Community Reports (fallback bằng NetworkX)

Nếu Neo4j instance không có Graph Data Science (GDS) library phù hợp, dùng fallback: export toàn bộ edges ra pandas, dùng NetworkX `greedy_modularity_communities` để phát hiện cụm (community detection), viết ngược `community_id` vào Neo4j bằng UNWIND theo batch (hàm `build_communities(limit_edges=20000)`), rồi dùng LLM tóm tắt nội dung từng community, cuối cùng cho phép query "global" trên các bản tóm tắt community này thay vì trên từng node riêng lẻ.

### Bonus C - Self-Correction Graph Retrieval

Ý tưởng: sau khi lấy context ở hop 2, dùng một LLM kiểm tra (`context_sufficient`) xem context đã đủ để trả lời chưa; nếu thiếu thì tăng lên hop 3; nếu vẫn thiếu thì fallback sang vector search bổ sung. Bắt buộc phải có điều kiện dừng (stop condition) để tránh vòng lặp vô hạn - trong code khung, tối đa chỉ đến hop 3 rồi bắt buộc dừng lại và bổ sung vector.

---

## CHECKLIST NỘP BÀI (riêng cho bạn tích khi hoàn thành)

- [ ] Neo4j đã connect thành công (`connect_neo4j()` không lỗi)
- [ ] Đã chạy dedup/chunking (`chunks_df` có dữ liệu)
- [ ] Đã spot-check (kiểm tra ngẫu nhiên) kết quả coreference resolution
- [ ] Đã có bảng audit entity resolution (`entity_resolution_audit_df`)
- [ ] Đã bulk insert node + edge bằng UNWIND (không insert từng dòng)
- [ ] `graph_checks()` cho kết quả 0 cạnh thiếu provenance
- [ ] Flat RAG đã chạy và trả lời được
- [ ] Hybrid GraphRAG đã chạy và trả lời được
- [ ] Đã chạy `test_supernode_policy()` và có bằng chứng cap 50 cạnh
- [ ] Golden Dataset đã có đủ 5 reference_answer THẬT (không còn "TO_BE_FILLED_FROM_DATASET")
- [ ] Đã chạy evaluation hết toàn bộ Golden Dataset (có checkpoint CSV)
- [ ] Đã export `graphrag_eval_results.csv` và `graphrag_vs_flatrag_summary.csv`
- [ ] Đã viết đầy đủ phần thuyết minh kỹ thuật (10 câu hỏi Phần 5.2)
- [ ] Nếu làm Bonus: có số liệu định lượng trước/sau rõ ràng
- [ ] Đã chuẩn bị link GitHub/Drive/LMS để nộp bài

---

## RUBRIC CHẤM ĐIỂM (tóm tắt)

| Tiêu chí | Trọng số | Yêu cầu |
|---|---|---|
| Chạy được code | 30% | Graph nạp thành công, schema đúng, xuất được bảng so sánh |
| Xử lý Failure modes | 30% | Xử lý được ít nhất 2/3 vấn đề: Super-node, Entity Resolution, Coreference |
| Evaluation | 20% | Chạy hết Golden Dataset, phân tích hợp lý |
| Thuyết minh | 20% | Giải thích được kiến trúc và cách kiểm soát AI Coding Agent |

---

## Ghi chú cuối

Tài liệu này là bản tổng hợp hướng dẫn thao tác dựa trên nội dung công khai của trang codelab (VLearn), nhằm giúp bạn đi đúng từng bước trong Colab. Toàn bộ code chi tiết, đầy đủ và có thể copy-paste trực tiếp vẫn nằm trong notebook Colab gốc của khóa học - hãy mở notebook đó song song với file hướng dẫn này khi thực hành. Bạn nên đọc lại mục "LƯU Ý QUAN TRỌNG" ở đầu file trước khi bắt đầu từng phần.
