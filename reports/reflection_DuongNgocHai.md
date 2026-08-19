# Suy ngẫm Cá nhân & Kế hoạch Đồ án — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Dương Ngọc Hải — 2A202601748 · AICB-K34 Track 3 · 19/08/2026

---

## 1. Mapping bài giảng vào code

| Khái niệm trong bài giảng | Module | Hàm / khối code | Quan sát thực tế & đánh giá |
|---|---|---|---|
| **Conservative Coreference** | M1 | `needs_coref()`, `resolve_coref_batch()`, `run_coref()` | Rule-gate cắt 43% lượt gọi LLM (197/400 cần, 171 bỏ qua). Nhưng 71/114 thay đổi chỉ là viết lại dấu câu → prompt nên bắt trả **span** thay vì full text |
| **Chunking & Exact dedup** | M1 | `chunk_text()`, `standardize_news()` | 2.687 → 2.114 bài. Snippet ngắn nên tỉ lệ 1 chunk/bài — mọi multi-hop buộc phải nối **giữa** các tài liệu, đúng chỗ Flat RAG yếu nhất |
| **Schema & Allowlist Guard** | M2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS`, `parse_extraction_items()` | Chặn được rác, nhưng cũng **làm mất thông tin** — nguyên nhân gốc của ca lỗi `G5000-28` |
| **Edge Provenance** | M2 | `MERGE (s)-[r {source_chunk_id}]->(t)` | Đặt `source_chunk_id` trong `MERGE` (không phải `SET`) để hai bài báo tạo hai cạnh — giữ tín hiệu cross-doc. Kết quả: `invalid_provenance_edges = 0/271` |
| **Bulk Cypher Ingestion** | M2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | `UNWIND` batch 1.000: 393 node + 271 cạnh trong **1,6 giây** — chứng minh Neo4j không phải nút thắt khi scale |
| **Entity Resolution & Union-Find** | M3 | `build_resolution_map()`, `merge_guard()`, `UF` | 23 dòng audit. Phát hiện guard 0.72 **quá lỏng cho `Person`** (`Sam Altman`/`Steve Altman` ratio 0.727) |
| **Super-node Degree Cap** | M4 | `retrieve_graph_context()`, `recent_edges()` | Max degree thật = 21 < 100 → phải hạ ngưỡng tạm thời mới kiểm chứng được cơ chế |
| **Seed Extraction & Fuzzy Match** | M4 | `extract_seeds()`, `match_seeds()` | 3 tầng exact → lexical → vector, mỗi seed ghi lại route. Nhờ đó chẩn đoán được ca `G5000-30` là lỗi **seed nghèo** |
| **Hybrid Context** | M4 | `answer_graph_rag()`, `textualize()` | `=== GRAPH ===` + `=== VECTOR ===`; nhánh vector cứu được các câu đồ thị không phủ tới |
| **LLM-as-a-Judge** | M5 | `judge_answer()`, `run_evaluation()` | Bổ sung `evidence_recall()` làm thước đo khách quan song song với điểm chủ quan |
| **Community Detection** | Bonus B | `build_communities()`, `community_report()` | 145 community; 5 cụm lớn nhất sinh được report cho câu hỏi vĩ mô |
| **Self-Correction** | Bonus C | `context_sufficient()`, `self_correcting_context()` | 3 câu khó nhất: tổng điểm judge **+3** |
| **Near-Dedup** | Bonus A | `minhash_signature()`, LSH banding | So **1.128 cặp** thay vì 2.232.241 cặp O(N²) |

---

## 2. Quá trình debugging & bài học

### Lỗi khó nhất: trần token/ngày của Groq cạn giữa chừng

**Triệu chứng gây hiểu nhầm:** coref chạy 43 giây/batch trong khi extraction chỉ 17 giây/batch. Phản xạ đầu tiên
của tôi là “prompt coref quá dài” — và nếu đi theo hướng đó thì sẽ tối ưu nhầm chỗ hoàn toàn.

**Điều thực sự xảy ra:** Groq free tier có **hai trần khác nhau** — 8.000 token/phút (rolling) và
**200.000 token/ngày, tính riêng cho từng model**. Hạn mức của `gpt-oss-120b` đã cạn ở mức 199.770/200.000, và
mỗi lượt gọi phải chờ backoff nên *trông giống như* prompt chậm.

**Chuỗi xử lý:**
1. Đọc header `x-ratelimit-*` thay vì đoán → xác nhận trần theo từng model, không phải theo tài khoản.
2. **Giảm nhu cầu:** `reasoning_effort="low"`, rule-gate cho coref (−43% lượt gọi), batch 4 cho extraction.
3. **Định tuyến lại theo vai trò:** NER+RE giữ trên Groq (bước dựng đồ thị, theo đúng tinh thần đề bài),
   coref/seed/generation/judge sang OpenAI. Vì cả hai kiến trúc dùng **chung một generator**, phép so sánh
   vẫn công bằng.
4. **Chống mất mát:** cache đĩa cho coref/triples + checkpoint tăng dần **sau mỗi batch**, checkpoint đánh giá
   theo từng câu hỏi.

**Kết quả đo được:** lần chạy đầu 28 phút; các lần chạy lại sau đó chỉ **3 phút**. Và khi notebook gặp lỗi ghi file
trên Windows (`OSError: Errno 22`, do IDE giữ file) thì **không mất một kết quả nào** — chỉ cần chạy lại.

**Bài học lớn nhất của bài lab này:** với pipeline LLM dài, **checkpoint và idempotency quan trọng ngang thuật toán**.
Chi phí khoảng 30 dòng code, nhưng nó biến mọi vòng lặp sửa lỗi từ “chạy lại nửa tiếng” thành “chạy lại ba phút” —
tức là biến việc thử nghiệm từ đắt đỏ thành rẻ, và đó mới là thứ quyết định chất lượng cuối cùng.

### Lỗi đáng nhớ thứ hai: tin vào schema mà model hứa

Model nhỏ (`gpt-oss-20b`) đôi lúc trả `items` là **list chuỗi** thay vì list object → `AttributeError` giết cả batch,
mất toàn bộ công của 4 chunk. Sửa bằng `parse_extraction_items()` phòng thủ: phần tử sai schema chỉ bị bỏ qua và đếm
vào `schema_violations`, kèm cơ chế **hạ batch xuống từng chunk** khi cả lô hỏng.

Nguyên tắc rút ra: *khi LLM là một thành phần trong đường ống, hãy đối xử với output của nó như dữ liệu từ mạng —
luôn giả định nó có thể sai định dạng, và đừng bao giờ để một phần tử hỏng làm chết cả lô.*

### Điều tôi làm khác so với gợi ý mặc định

Bài lab gợi ý `chunks_df.head(400)` để chọn subset trích xuất. Tôi khảo sát dataset trước và thấy dữ liệu được
**sắp xếp theo `companyName`** — 400 dòng đầu chỉ có công ty vần 0/1/A/B, không có Microsoft/Google/OpenAI, nên đồ thị
sẽ rời rạc và không có đường multi-hop nào. Tôi thay bằng **entity-anchored stratified sampling** ba tầng:
bắt buộc phủ 28/28 chunk bằng chứng vàng → ưu tiên chunk giàu thực thể → phần còn lại lấy ngẫu nhiên có seed cố định.
Không có bước khảo sát dữ liệu này thì toàn bộ benchmark đã trở nên vô nghĩa.

---

## 3. Kế hoạch áp dụng vào đồ án thực tế (Action Plan)

**Tên đồ án:** Hệ hỏi–đáp trên kho tài liệu nội bộ (hợp đồng, biên bản, quyết định) của một tổ chức.

### Bài toán của tôi có cần GraphRAG không?

**Cần, nhưng ở dạng Hybrid có định tuyến — không phải GraphRAG thuần.** Lý do dựa trên số liệu của chính lab này:

| Nhóm câu hỏi | Tỉ lệ ước tính trong đồ án | Multi-hop reasoning (Flat → Graph) | Kết luận |
|---|---|---|---|
| Tra cứu điều khoản (factoid) | ~70% | 5.00 → 5.00 (**hoà**) | Flat RAG là đủ; GraphRAG chỉ tốn thêm 30% token |
| Truy vết quan hệ nhiều chặng | ~20% | 3.58 → **4.17** | GraphRAG thắng rõ |
| So sánh/tổng hợp nhiều văn bản | ~10% | 2.64 → **3.91** | GraphRAG thắng đậm nhất |

→ Thiết kế: **router phân loại câu hỏi trước**, factoid đi nhánh Flat, còn lại đi nhánh Graph.
Đây là kết luận tôi sẽ không rút ra được nếu chỉ nhìn con số tổng hợp — phải tách theo nhóm mới thấy.

### Cấu trúc Node & Relation dự kiến

- **Nodes:** `Organization`, `Person`, `Document`, `Contract`, `Department`, `Project`
- **Relations:** `SIGNED_BY`, `APPROVED_BY`, `SUPERSEDES`, `REFERENCES`, `BELONGS_TO`, `SUPPLIES_TO`
- **Provenance bắt buộc trên mọi cạnh:** `doc_id`, `page`, `clause_id`, `effective_date`, `evidence`, `confidence`

Điểm khác biệt quan trọng so với lab: văn bản pháp lý phải trích dẫn được tới **điều khoản**, không chỉ tới tài liệu.
Bài học từ `MERGE (s)-[r {source_chunk_id}]->(t)` áp dụng trực tiếp — khoá `MERGE` sẽ là `clause_id`, để hai điều khoản
khác nhau cùng nói về một quan hệ vẫn là hai cạnh riêng biệt.

Ngoài ra `SUPERSEDES` là quan hệ mà lab không có nhưng đồ án bắt buộc phải có: văn bản pháp lý có hiệu lực theo thời
gian, một phụ lục có thể thay thế điều khoản cũ. Truy vấn phải trả lời được “điều khoản nào **đang** có hiệu lực”,
chứ không phải “có những điều khoản nào”.

### Chiến lược Entity Resolution

Bài học trực tiếp từ cặp `Sam Altman` / `Steve Altman`: **khi thực thể là người hoặc pháp nhân, false merge có hậu
quả pháp lý.** Vì vậy:

1. **Định danh cứng làm khoá chính:** tổ chức có mã số thuế, nhân sự có mã nhân viên → gộp theo mã, không theo tên.
2. **Vector chỉ để gợi ý, không tự động gộp:** cặp có similarity cao được đưa vào **hàng đợi kiểm duyệt của người**,
   giống bảng audit của lab nhưng có thêm bước phê duyệt.
3. **Lexical-first cho tên tổ chức:** khớp tuyệt đối sau khi chuẩn hoá hậu tố (`Công ty TNHH`, `Cổ phần`, `JSC`)
   thì gộp ngay — sửa đúng lỗi #6 mà tôi phát hiện trong lab (`Synopsys` vs `Synopsys Inc.` bị gộp hụt).

### Chiến lược Super-node

Node kiểu “Phòng Hành chính” hay “Ban Giám đốc” sẽ nối tới hàng nghìn văn bản — vượt xa mức degree 21 của lab.
Chính sách “50 cạnh mới nhất” của lab là **chưa đủ**, vì nó cắt theo node mà không biết câu hỏi đang hỏi gì.
Kế hoạch:

1. **Cắt theo loại quan hệ suy ra từ câu hỏi:** câu hỏi về ký kết → chỉ mở cạnh `SIGNED_BY`/`APPROVED_BY`,
   không mở `REFERENCES`.
2. **`effective_date DESC` làm tiêu chí phụ**, cộng lọc theo trạng thái hiệu lực (loại cạnh đã bị `SUPERSEDES`).
3. **Trần cứng theo hop** giữ nguyên như lab (đã chứng minh hoạt động: 52 cạnh ≤ 250, context 11.484 ≤ 14.000 ký tự).
4. **Phân vùng theo phòng ban/dự án** (áp dụng community detection của Bonus B) để truy vấn không quét toàn cục.

### Điều tôi sẽ làm khác ngay từ đầu

- **Xây bộ golden dataset trước khi xây hệ thống.** Trong lab, việc golden set gắn với chỉ số dòng gốc cho phép tính
  Evidence Recall — và chính chỉ số đó mới giúp tôi chẩn đoán được ca `G5000-28` là lỗi *mô hình hoá* chứ không phải
  lỗi *truy hồi*. Không có nó, tôi đã đi sửa nhầm bước retrieval.
- **Thiết kế schema với biên độ dự phòng.** Ca `G5000-28` cho thấy allowlist quá hẹp sinh ra cạnh *đúng cú pháp
  nhưng sai ngữ nghĩa* — loại lỗi khó phát hiện hơn hẳn lỗi crash. Tôi sẽ bắt đầu bằng một vòng khảo sát thủ công
  ~50 tài liệu để liệt kê các quan hệ thực sự tồn tại, thay vì tự nghĩ ra danh sách.
- **Đo chi phí ngay từ ngày đầu.** Bảng `llm_cost_summary()` (số lượt gọi, token, retry theo từng tag) là thứ rẻ nhất
  để thêm và hữu ích nhất khi cần trả lời câu hỏi “scale lên thì hỏng ở đâu”.

---

## 🎯 Tự đánh giá

| Tiêu chí | Điểm (1–5) | Ghi chú |
|---|---|---|
| Mức độ hiểu bài giảng GraphRAG | 5 | Giải thích được vì sao lợi ích tập trung ở cross-doc (+1.27) và vì sao factoid hoà |
| Khả năng kiểm soát AI Coding Agent | 5 | Từ chối 4 đề xuất có hại, mỗi lần đều kèm bằng chứng số liệu |
| Chất lượng đồ thị tri thức | 4 | Provenance 100%, audit minh bạch; trừ điểm vì allowlist quá hẹp gây mất thông tin |
| Khả năng phân tích và debug | 5 | Truy được nguyên nhân gốc của cả rate-limit lẫn ca lỗi GraphRAG nhờ Evidence Recall |
