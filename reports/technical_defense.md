# Thuyết minh Kỹ thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Dương Ngọc Hải — 2A202601748 · AICB-K34 Track 3 · 19/08/2026
**Bản đầy đủ (kèm reflection):** [`lab_report.md`](lab_report.md)

Mọi số liệu dưới đây lấy từ lần chạy thật trong `Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb`
và các tệp trong `outputs/`.

---

## 1. Coreference Resolution phân giải sai ở đâu, hậu quả gì?

**Ca cụ thể — `row00219::c0000`:** đoạn trích lời lãnh đạo dùng ngôi thứ nhất (`we are`, `our platform`).
Model thay bằng **`the company is`, `its platform`** — tức là thay một tham chiếu mơ hồ bằng một tham chiếu
mơ hồ khác, thay vì phân giải về tên công ty thật.

**Hậu quả với KG:** nếu bước NER+RE tin chuỗi này, nó có thể sinh node tên `The company` — một thực thể rác
nối nhiều bài báo không liên quan, tạo **super-node giả**. Kiểm tra graph thực tế: không có node dạng này
(chỉ có tên riêng hợp lệ `The Container Store`), nhờ hai lớp chặn — allowlist kiểu node và yêu cầu `evidence`
cho mọi quan hệ.

**Ca hỏng thứ hai — `row00391::c0000`:** model **xoá mất một ký tự** khi viết lại. Với hệ dựa trên trích dẫn
nguyên văn, đây là rủi ro âm thầm: `evidence` lưu trong đồ thị không còn khớp tuyệt đối với văn bản gốc.

**Đo lường:** 114/400 chunk bị sửa, nhưng chỉ **43 chunk thực sự thay từ**; **71 chunk chỉ khác dấu câu** —
model tự ý viết lại ngoài nhiệm vụ. → Bản production nên bắt model trả **span thay thế** (offset + chuỗi mới)
thay vì trả toàn bộ văn bản.

---

## 2. Ngưỡng cosine và Lexical Guard

**Ngưỡng dùng:** `threshold = 0.90`; guard `SequenceMatcher ≥ 0.72` sau khi bỏ hậu tố `Inc/Corp/Ltd/LLC`.

**Cặp similarity > 0.85 bị Lexical Guard chặn:**

| Trái | Phải | Cosine | Lexical ratio | Quyết định |
|---|---|---|---|---|
| `generative AI` | `Generative AI Capabilities` | **0.856** | 0.667 | **REJECT_GUARD** |
| `cloud computing` | `cloud services` | 0.830 | 0.483 | REJECT_GUARD |

**Lý do:** `Generative AI Capabilities` đến từ sản phẩm “Now Assist Generative AI Capabilities” của ServiceNow —
quan hệ *khái niệm tổng quát vs. tính năng sản phẩm cụ thể*. Gộp lại thì mọi phát biểu về một sản phẩm riêng lẻ
sẽ bị quy cho cả ngành công nghệ (đúng dạng lỗi `Apple` nuốt `Apple Watch`).

**Cơ sở chọn 0.90 — thí nghiệm độ nhạy** (`outputs/entity_resolution_threshold_sweep.csv`):
0.80 → 7 merge / 2 bị guard chặn / 387 entity; 0.85 → 5 / 1 / 389; **0.90 → 3 / 0 / 391**; 0.95 → 1 / 0 / 393.

**Hai giới hạn tự phát hiện** (`outputs/lexical_guard_probe.csv`):
- `Sam Altman` vs `Steve Altman`: cosine 0.824, ratio **0.727 > 0.72** → **guard sẽ cho gộp**; thứ cứu ta là ngưỡng
  cosine. Guard 0.72 quá lỏng cho `Person`, nên nâng lên ~0.85 hoặc bắt khớp cả họ lẫn tên.
- `Synopsys` vs `Synopsys Inc.`: ratio 1.0 nhưng cosine 0.832 < 0.90 → **gộp hụt**, đồ thị bị phân mảnh.
  Thứ tự đúng phải là **lexical-first**: khớp tuyệt đối sau chuẩn hoá hậu tố thì gộp ngay.

**Audit:** 23 dòng (`MERGE_MANUAL` 3 · `MERGE_VECTOR` 3 · `REJECT_THRESHOLD` 17). Audit ghi **mọi cặp cosine ≥ 0.75**,
kể cả cặp bị từ chối — vì cặp bị từ chối mới chứng minh cơ chế phòng vệ đang chạy.

---

## 3. Top 3 super-node và rủi ro của chính sách “50 cạnh mới nhất”

| Hạng | Thực thể | Loại | Degree |
|---|---|---|---|
| 1 | Microsoft | Company | **21** |
| 2 | Amazon | Company | 12 |
| 3 | ServiceNow | Company | 7 |

**Kiểm chứng cơ chế:** ở quy mô lab max degree = 21 < 100 nên nhánh cắt tỉa không bao giờ chạy. Tôi hạ ngưỡng
**tạm thời** xuống 20 để ép Microsoft thành super-node: `supernode_events = [{degree: 21, limit: 50}]`,
thu 31 cạnh, rồi khôi phục `SUPER_NODE_DEGREE = 100`. Kiểm tra trần toàn cục với hop=3, `edge_limit=200`:
**52 cạnh ≤ 250**, context **11.484 ≤ 14.000 ký tự**.

**Ưu điểm:** trần cứng cho chi phí truy vấn; ưu tiên `published_date DESC` khớp bản chất tin tức; kết quả xác định,
dễ tái lập.

**Rủi ro:** (a) **mù lịch sử** — câu hỏi về sự kiện cũ bị cắt mất nếu node có >50 cạnh mới hơn;
(b) **thiên lệch theo mật độ đưa tin** — một sự kiện được 30 báo đăng lại chiếm 30 slot (đây là lý do near-dedup
có giá trị thực tế); (c) **cắt theo node chứ không theo câu hỏi** — bản production nên xếp hạng cạnh theo độ liên quan
với câu hỏi trước, rồi mới cắt theo thời gian.

---

## 4. Bảng benchmark và 2 ca lỗi điển hình

### Bảng tổng hợp (25 câu, LLM-as-a-Judge)

| Tiêu chí | Flat RAG | GraphRAG | Δ |
|---|---|---|---|
| Comprehensiveness | 3.84 | **4.36** | +0.52 |
| Faithfulness | 3.76 | **4.08** | +0.32 |
| Multi-hop reasoning | 3.28 | **4.12** | **+0.84** |
| Evidence recall (khách quan) | 0.77 | **0.91** | +0.14 |
| Latency (s) | 2.21 | 2.16 | −0.05 |
| Token/câu | 699 | 1.083 | +384 (**1.55×**) |

Theo nhóm: factoid **hoà 5.00/5.00**; multi-hop 3.58 → 4.17; cross-doc **2.64 → 3.91**.
→ Lợi ích của GraphRAG **tỉ lệ thuận với số chặng suy luận**; với factoid nó chỉ làm tốn thêm token.

### Ca 1 — Flat RAG thất bại, GraphRAG thành công: `G5000-43` (cross-doc, Δ = +10)

Câu hỏi về **thứ tự thời gian** giữa thương vụ HPE mua Axis Security và thông báo dịch vụ cloud LLM.
Flat RAG chỉ lấy được bài Axis Security (evidence recall 0.5), thừa nhận thiếu dữ kiện rồi vẫn suy diễn.
GraphRAG khớp seed `HPE → Hewlett Packard Enterprise` (alias thủ công) và thu đúng hai cạnh mang `published_date`
2023-03-02 và 2023-06-21 → sắp xếp thời gian trở thành thao tác dữ liệu. Judge: 5/5/5.

### Ca 2 — GraphRAG thất bại: `G5000-28` (multi-hop, Δ = −5)

Hỏi các **nhà cung cấp mô hình** liên quan Google Cloud Next '23. GraphRAG gán sai `Llama 2` cho Google.
Nguyên nhân **không phải retrieval** (evidence recall = 1.0) mà là **phép chiếu mất mát của schema**: allowlist
8 quan hệ không có loại nào diễn đạt “Meta sở hữu Llama 2”, nên mọi thứ bị ép vào `Google Cloud -USES-> <model>`,
đẩy chủ thể sở hữu xuống trường `evidence`.
**Khắc phục:** thêm quan hệ `OFFERS`/`DISTRIBUTES` tách khỏi `DEVELOPED`; ràng buộc prompt sinh **hai** cạnh khi gặp
mẫu “X from Y”.

---

## 5. Trade-offs, kiểm soát AI Agent, và scale 350MB

**Đánh đổi:** chi phí GraphRAG nằm ở **offline** (92 lượt LLM cho NER+RE ~14 phút + 33 lượt coref ~6 phút),
không nằm ở latency truy vấn (2.16s vs 2.21s). Khi chạy thì đắt hơn **1.55× token**. Chi phí lớn nhất thực ra là
**bảo trì**: schema, entity resolution, super-node policy, provenance.

**Bốn đề xuất của AI Agent tôi từ chối:**
1. Bỏ `merge_guard()` để tăng recall → false merge không thể phát hiện từ câu trả lời cuối.
2. Pairwise cosine O(N²) cho near-dedup → thay bằng MinHash+LSH: **1.128 cặp** thay vì 2.232.241 cặp.
3. Nới `top_k`/`edge_limit` “cho chắc” → nhiễu tăng nhanh hơn tín hiệu; thay bằng self-correction có điều kiện.
4. `chunks_df.head(400)` → dữ liệu sắp theo `companyName` nên chỉ lấy được công ty vần 0/1/A/B; thay bằng
   entity-anchored stratified sampling phủ đủ 28/28 chunk bằng chứng vàng.

**Scale 350MB — bottleneck #1 là số lượt gọi LLM cho extraction.** Đo thật: 400 chunk ≈ 92 lượt ≈ 110k token
(~275 token/chunk). Ngoại suy toàn dump 8,08 triệu dòng × tỉ lệ hữu ích 42% ≈ 3,4 triệu chunk ≈ **930 triệu token**.
Ghi Neo4j không phải nút thắt (271 cạnh/1,6s), embedding cũng không (2.114 chunk/~60s CPU).
**Giải pháp:** lọc bằng NER cục bộ trước khi gọi LLM (−~50% lượt gọi) → hàng đợi worker bất đồng bộ với token bucket
→ model phân tầng (model nhỏ sàng lọc, model lớn xử lý phần đã lọc) → entity resolution chuyển HNSW + blocking
thay cho FAISS Flat → near-dedup chạy **trước** extraction → phân vùng đồ thị theo community.

---

## 6–10. Các câu hỏi bổ sung

**6. Vì sao `source_chunk_id` nằm trong `MERGE` chứ không phải `SET`?**
Để cùng một quan hệ được hai bài báo khác nhau khẳng định sẽ tạo **hai cạnh riêng biệt**. Đó chính là tín hiệu
cross-document (một sự kiện được nhiều nguồn xác nhận) — nếu gộp vào một cạnh thì mất khả năng đếm mức độ đồng thuận
và mất luôn dấu vết nguồn.

**7. Vì sao đo Evidence Recall bên cạnh điểm Judge?**
Điểm Judge là chủ quan và trộn lẫn hai loại lỗi. Evidence Recall (tỉ lệ chunk bằng chứng vàng thực sự có trong
ngữ cảnh) tách bạch: **recall = 0 → hỏng ở retrieval**; **recall = 1 nhưng điểm thấp → hỏng ở generation/schema**.
Nhờ đó ca `G5000-28` được chẩn đoán đúng là lỗi mô hình hoá dữ liệu, không phải lỗi truy hồi.

**8. Vì sao seed matching có 3 tầng?**
Exact (`name_norm`/`aliases_norm`) → lexical containment (`CONTAINS`, để “Dell” chạm được “Dell Technologies”) →
vector ≥ 0.66. Mỗi seed ghi lại nó được giải bằng tầng nào (`EXACT`/`LEXICAL`/`VECTOR`/`BELOW_THRESHOLD`), nên khi
GraphRAG trả lời kém ta biết ngay là do **không tìm được điểm xuất phát** hay do **đồ thị thiếu cạnh**.
Trong 25 câu có **1 câu** không thu được cạnh nào (`G5000-29`) và **1 câu** seed quá nghèo (`G5000-30`, recall 0).

**9. Vì sao dùng hai nhà cung cấp LLM và điều đó có làm hỏng phép so sánh không?**
Groq free tier giới hạn **200.000 token/ngày cho mỗi model** và hạn mức của `gpt-oss-120b` đã cạn (199.770/200.000).
Pipeline được định tuyến lại: NER+RE giữ trên Groq `gpt-oss-20b`, còn coref/seed/sinh câu trả lời/judge dùng
OpenAI `gpt-4o-mini`. Phép so sánh **không bị ảnh hưởng** vì Flat RAG và GraphRAG dùng **chung một generator và chung
một judge** — biến duy nhất thay đổi là ngữ cảnh đưa vào prompt.
*(Hạn chế cần ghi nhận: generator và judge cùng là `gpt-4o-mini` nên có thể có thiên vị tự chấm; vì thiên vị đó
áp dụng như nhau cho cả hai nhánh, thứ hạng tương đối vẫn có giá trị, và chỉ số Evidence Recall là thước đo độc lập
không phụ thuộc judge.)*

**10. Nếu chỉ được giữ lại một cơ chế phòng vệ, giữ cái nào?**
**Edge provenance.** Không có nó thì không kiểm chứng được câu trả lời, không sắp xếp được theo thời gian
(super-node policy trở nên vô nghĩa), và không truy được nguyên nhân gốc của bất kỳ ca lỗi nào. Trong lần chạy này
`invalid_provenance_edges = 0` trên toàn bộ 271 cạnh, kiểm tra cả `NULL` lẫn chuỗi rỗng.
