# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Dương Ngọc Hải — 2A202601748
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 19/08/2026
**Notebook:** `Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb` (42 cell, chạy Run All không lỗi)

---

## 0. Tóm tắt cấu hình thực nghiệm

| Hạng mục | Giá trị thực tế |
|---|---|
| Dataset | `HackerNoon/tech-company-news-data-dump` — **5.000 dòng đầu** |
| Corpus sau làm sạch | 5.000 → 2.687 (lọc snippet <120 ký tự) → **2.114 chunk** (exact dedup SHA-1) |
| Chunk đưa vào trích xuất KG | **400** (bao phủ **28/28** chunk bằng chứng của Golden Dataset) |
| Đồ thị Neo4j | **393 node** (Company 257 · Technology 107 · Person 29), **271 cạnh**, **0 cạnh thiếu provenance** |
| Golden Dataset | **25 câu** — 2 factoid · 12 multi-hop · 11 cross-doc, tất cả gán nhãn `hard` |
| LLM trích xuất (NER+RE) | Groq `openai/gpt-oss-20b` |
| LLM coref / seed / sinh câu trả lời | OpenAI `gpt-4o-mini` (dùng **chung** cho cả Flat RAG và GraphRAG) |
| LLM-as-a-Judge | OpenAI `gpt-4o-mini` |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` (384 chiều), FAISS `IndexFlatIP` |

> **Vì sao chọn phạm vi 5.000 dòng đầu:** bộ Golden Dataset trong `data/graphrag_golden_50_first5000_detailed.csv`
> được xây trên đúng phạm vi này, cột `evidence_row_ids_0based` trỏ thẳng vào chỉ số dòng gốc. Lấy đúng
> phạm vi đó làm corpus cho ba lợi ích: (1) mọi câu hỏi vàng đều có bằng chứng nằm trong corpus nên benchmark
> đo **năng lực kiến trúc** chứ không đo may rủi dữ liệu; (2) `article_id = row<row_id>` cho phép kiểm chứng
> provenance tới từng dòng dữ liệu gốc; (3) tính được **Evidence Recall** khách quan.

> **Vì sao chia hai nhà cung cấp LLM:** Groq free tier giới hạn **200.000 token/ngày cho mỗi model**; trong
> quá trình phát triển, hạn mức của `gpt-oss-120b` đã cạn (199.770/200.000). Pipeline được định tuyến lại:
> bước dựng đồ thị (NER+RE) giữ trên Groq, các bước còn lại chuyển sang OpenAI. Vì **cả hai kiến trúc dùng
> chung một generator**, phép so sánh vẫn công bằng — chỉ khác ngữ cảnh đưa vào.

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)

**Cơ chế:** Trước khi gọi LLM, một regex rẻ tiền lọc chunk có đại từ/tham chiếu chung
(`it`, `they`, `the company`, `the deal`…). Kết quả: **197/400 chunk cần LLM, 171 chunk bỏ qua hoàn toàn** —
cắt 43% chi phí ở bước này mà không mất gì, vì snippet tin tức thường đã nêu tên đầy đủ.

**Đo lường:** 114/400 chunk bị sửa. Nhưng khi phân tích kỹ sự khác biệt ở mức token:

| Loại thay đổi | Số chunk | Đánh giá |
|---|---|---|
| Chỉ khác dấu câu / khoảng trắng | **71** | ❌ Model tự ý viết lại — vượt quá nhiệm vụ được giao |
| Thực sự thay thế từ (đại từ → tên) | **43** | ✅ Đúng mục tiêu |

**Ca phân giải sai cụ thể — `row00219::c0000`:**

```
BEFORE: ... "we are excited ..."  ... "our platform" ...
AFTER : ... "the company is excited ..." ... "its platform" ...
```

**Hiện tượng:** Model gặp ngôi thứ nhất (`we`, `our`) trong đoạn trích dẫn phát ngôn của lãnh đạo. Thay vì
phân giải về **tên công ty đích thực**, nó thay bằng một **cụm danh từ chung “the company”**. Đây là phân giải
*sai loại*: chuỗi kết quả trông “sạch” hơn nhưng vẫn không trỏ tới thực thể nào.

**Hậu quả đối với Knowledge Graph:** nếu bước NER+RE tin vào chuỗi này, nó có thể tạo node tên là
`The company` — một thực thể rác nối tới nhiều bài báo khác nhau, tức là một **super-node giả** hợp nhất
nhiều công ty không liên quan. Kiểm tra graph thực tế cho thấy **không có node rác dạng này** (chỉ có
`The Container Store` là tên riêng hợp lệ), nhờ hai lớp chặn: schema allowlist yêu cầu `source_type`/`target_type`
hợp lệ, và prompt trích xuất yêu cầu bằng chứng trích dẫn cho mọi quan hệ.

**Ca hỏng thứ hai — mất ký tự (`row00391::c0000`):** khác biệt token cho thấy model **xoá mất một ký tự `f`**
khi viết lại. Với hệ thống dựa trên trích dẫn nguyên văn (`evidence`), việc LLM tự ý chỉnh văn bản là rủi ro
âm thầm: bằng chứng lưu trong đồ thị **không còn khớp tuyệt đối** với văn bản gốc.

**Bài học rút ra:** bước coref nên bị ràng buộc mạnh hơn — chỉ cho phép thay thế **span** (vị trí bắt đầu/kết thúc
+ chuỗi thay thế) thay vì trả về toàn bộ văn bản đã viết lại. Như vậy mọi thay đổi đều kiểm chứng được và
không thể mất ký tự.

---

### 2. Entity Resolution Threshold & Lexical Guard

**Ngưỡng đang dùng:** `threshold = 0.90` (cosine trên embedding MiniLM), guard `SequenceMatcher ≥ 0.72`
sau khi bỏ hậu tố `Inc/Corp/Ltd/LLC`.

**Cơ sở chọn ngưỡng — thí nghiệm độ nhạy** (`outputs/entity_resolution_threshold_sweep.csv`):

| Ngưỡng | MERGE_VECTOR | REJECT_GUARD | Số entity canonical còn lại |
|---|---|---|---|
| 0.80 | 7 | **2** | 387 |
| 0.85 | 5 | 1 | 389 |
| **0.90** | **3** | 0 | **391** |
| 0.95 | 1 | 0 | 393 |

**Cặp similarity cao bị Lexical Guard chặn** (quan sát ở ngưỡng 0.80–0.85):

| Trái | Phải | Cosine | Lexical ratio | Quyết định |
|---|---|---|---|---|
| `generative AI` | `Generative AI Capabilities` | **0.856** | 0.667 | **REJECT_GUARD** |
| `cloud computing` | `cloud services` | 0.830 | 0.483 | REJECT_GUARD |

**Lý do chặn `generative AI` vs `Generative AI Capabilities`:** vector similarity cao vì hai cụm nằm cùng miền
ngữ nghĩa, nhưng đây là quan hệ **khái niệm tổng quát vs. tính năng cụ thể của một sản phẩm** (cụm thứ hai đến từ
“Now Assist Generative AI Capabilities” của ServiceNow). Gộp chúng sẽ khiến mọi phát biểu về một sản phẩm riêng lẻ
bị quy cho toàn bộ ngành công nghệ — đúng dạng lỗi “Apple Watch bị nuốt vào Apple” mà đề bài cảnh báo.

**Kiểm chứng có kiểm soát — `outputs/lexical_guard_probe.csv`:** tôi chạy thêm 10 cặp “bẫy” kinh điển
qua đúng hàm `merge_guard()`:

| Cặp | Cosine | Lexical ratio | Guard cho gộp? | Kết quả cuối |
|---|---|---|---|---|
| `Apple` vs `Apple Watch` | 0.609 | 0.625 | ❌ Không | REJECT |
| `Llama 2` vs `Code Llama` | 0.784 | 0.588 | ❌ Không | REJECT |
| **`Sam Altman` vs `Steve Altman`** | **0.824** | **0.727** | ⚠️ **Có** | REJECT (nhờ ngưỡng 0.90) |
| `Synopsys` vs `Synopsys Inc.` | 0.832 | 1.000 | ✅ Có | REJECT (ngưỡng chặn — **gộp hụt**) |

Hai dòng cuối là phát hiện quan trọng và tôi ghi lại trung thực:

- **`Sam Altman` vs `Steve Altman`**: ratio 0.727 **vượt** ngưỡng guard 0.72 → nếu chỉ dựa vào guard thì hai người
  khác nhau đã bị gộp. Thứ cứu ta là ngưỡng cosine 0.90. Kết luận: **guard 0.72 quá lỏng cho `Person`** —
  với người, nên yêu cầu khớp cả họ và tên, hoặc nâng ratio lên ~0.85.
- **`Synopsys` vs `Synopsys Inc.`**: lexical ratio 1.0 sau khi bỏ hậu tố, rõ ràng là cùng một công ty, nhưng
  cosine 0.832 < 0.90 nên **không được gộp** → đồ thị bị phân mảnh. Kết luận: thứ tự đúng phải là
  **lexical-first** (khớp hoàn toàn sau chuẩn hoá hậu tố thì gộp ngay, không cần hỏi vector).

**Kết quả audit:** `outputs/entity_resolution_audit.csv` có **23 dòng**, phân loại
`MERGE_MANUAL` 3 · `MERGE_VECTOR` 3 · `REJECT_THRESHOLD` 17. Tôi cố ý ghi audit cho **mọi cặp có cosine ≥ 0.75**
chứ không chỉ cặp được gộp — vì cặp *bị từ chối* mới là bằng chứng cho thấy cơ chế phòng vệ đang hoạt động.

---

### 3. Đồ thị & Super-node Mitigation

**Top 3 thực thể bậc cao nhất:**

| Hạng | Tên thực thể | Loại | Bậc kết nối |
|------|--------------|------|-------------|
| 1 | Microsoft | Company | **21** |
| 2 | Amazon | Company | 12 |
| 3 | ServiceNow | Company | 7 |

*(Amazon Web Services và L&T Technology Services cùng bậc 7.)*

**Vấn đề kiểm chứng:** ở quy mô lab (400 chunk), bậc lớn nhất chỉ đạt **21 < 100**, nghĩa là nhánh
`if degree > SUPER_NODE_DEGREE` **không bao giờ chạy** → không có bằng chứng nào chứng minh code đúng.
Tôi xử lý bằng cách hạ ngưỡng **tạm thời** xuống 20 để ép Microsoft trở thành super-node và quan sát hành vi thật:

```
supernode_events: [{'node_id': 'fb0f4df5…', 'degree': 21, 'limit': 50}]
edges thu được : 31        (bị chặn đúng theo cap)
Đã khôi phục SUPER_NODE_DEGREE = 100
```

Đồng thời kiểm tra hai trần còn lại bằng truy vấn 3 hop, `edge_limit=200`:

```
Hop=3, edge_limit=200 -> 52 edges  (GLOBAL_EDGE_CAP = 250)   ✅
Độ dài graph context : 11.484 ký tự (trần 14.000)            ✅
```

**Ưu điểm của chính sách “50 cạnh mới nhất”:**
- Chặn được bùng nổ context: chi phí truy vấn có trần cứng, không phụ thuộc độ “nổi tiếng” của thực thể.
- Ưu tiên `published_date DESC` khớp với bản chất dữ liệu tin tức: câu hỏi thường về hiện trạng gần nhất.
- Giữ tính xác định: cùng câu hỏi luôn cho cùng tập cạnh, dễ debug và tái lập.

**Rủi ro:**
- **Mù lịch sử:** câu hỏi kiểu “năm 2019 Microsoft mua công ty nào” sẽ bị cắt mất nếu node có hơn 50 cạnh mới hơn.
- **Thiên lệch theo mật độ đưa tin:** một sự kiện được 30 báo đăng lại sẽ chiếm 30 slot, đẩy các quan hệ hiếm
  nhưng quan trọng ra khỏi cửa sổ — chính là lý do bonus near-dedup (MinHash) có giá trị thực tế.
- **Cắt theo node, không theo câu hỏi:** chính sách hiện tại không biết câu hỏi đang hỏi gì. Bản production nên
  xếp hạng cạnh theo *độ liên quan tới câu hỏi* (ví dụ lọc theo loại quan hệ được suy ra từ câu hỏi) trước khi
  cắt theo thời gian.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

#### Bảng tổng hợp Benchmark — 25 câu hỏi, LLM-as-a-Judge (`outputs/graphrag_vs_flatrag_summary.csv`)

| Tiêu chí đánh giá | Flat RAG | GraphRAG | Δ | Nhận xét phân tích |
|-------------------|----------|----------|---|--------------------|
| **Comprehensiveness (1–5)** | 3.84 | **4.36** | +0.52 | GraphRAG bổ sung được các thực thể/ngày tháng nằm ở bài báo khác |
| **Faithfulness (1–5)** | 3.76 | **4.08** | +0.32 | Cạnh mang `evidence` + `chunk_id` giúp model bớt suy diễn |
| **Multi-hop Reasoning (1–5)** | 3.28 | **4.12** | **+0.84** | Khoảng cách lớn nhất — đúng nơi lý thuyết dự đoán |
| **Evidence Recall (khách quan)** | 0.77 | **0.91** | +0.14 | Hybrid lấy về 91% chunk bằng chứng vàng so với 77% |
| **Latency trung bình (s)** | 2.21 | **2.16** | −0.05 | Ngang nhau: chi phí duyệt Neo4j nhỏ hơn dao động của LLM |
| **Token usage trung bình** | 699 | 1.083 | **+384 (1.55×)** | Giá phải trả: graph context làm prompt dài hơn |

**Tách theo nhóm câu hỏi** (chỗ kết luận trở nên sắc nét):

| Nhóm | Multi-hop reasoning (Flat → Graph) | Evidence recall (Flat → Graph) |
|---|---|---|
| factoid (2 câu) | 5.00 → 5.00 (hoà) | 1.00 → 1.00 |
| multi-hop (12 câu) | 3.58 → **4.17** | 0.77 → 0.86 |
| cross-doc (11 câu) | 2.64 → **3.91** | 0.73 → **0.96** |

> **Kết luận đọc được từ bảng:** với câu hỏi factoid, GraphRAG **không mang lại lợi ích nào** mà vẫn tốn thêm
> 30% token — đây là lập luận định lượng cho việc **định tuyến truy vấn** (factoid → Flat, multi-hop/cross-doc
> → Graph) thay vì dùng GraphRAG cho mọi câu.

#### Ca lỗi 1 — Flat RAG thất bại, GraphRAG thành công (`G5000-43`, cross-doc, Δ = **+10**)

**Câu hỏi:** *“Which came first in the selected HPE timeline: the Axis Security acquisition agreement or the
LLM-focused cloud service announcement, and what does that ordering show?”*

- **Flat RAG (evidence recall 0.5):** chỉ lấy được bài về Axis Security (2023-03-02); bài về dịch vụ cloud LLM
  (2023-06-21) **không vào top-6** vì nó không dùng từ “acquisition” hay “Axis”. Câu trả lời thừa nhận
  *“the LLM-focused cloud service announcement is not explicitly mentioned”* rồi vẫn suy diễn về “chiến lược ưu tiên
  bảo mật” — vừa thiếu vừa không có căn cứ. Judge cho comprehensiveness 2, faithfulness 2.
- **GraphRAG (evidence recall 1.0):** seed `HPE` khớp **EXACT** vào node `Hewlett Packard Enterprise` (nhờ
  alias thủ công `hpe →`), BFS thu về đúng hai cạnh:

```
Hewlett Packard Enterprise -ACQUIRED-> Axis Security | date=2023-03-02 | chunk=row04762::c0000
Hewlett Packard Enterprise -USES->     supercomputers | date=2023-06-21 | chunk=row03289::c0000
```

  Vì `published_date` là **thuộc tính hạng nhất của cạnh**, việc sắp xếp thời gian trở thành thao tác dữ liệu,
  không còn phụ thuộc việc model có “nhìn thấy” cả hai đoạn văn hay không. Judge cho 5/5/5.

**Nguyên nhân gốc rễ:** Flat RAG hỏng ở **khâu truy hồi** (recall 0.5), không phải khâu sinh câu trả lời.
Đây chính là lý do tôi thêm chỉ số Evidence Recall — nó tách bạch hai loại lỗi này ngay trên số liệu.

#### Ca lỗi 2 — GraphRAG thất bại (`G5000-28`, multi-hop, Δ = **−5**)

**Câu hỏi:** *“Which model providers are connected to Google Cloud Next '23, and which models are associated
with each provider?”*

- **Flat RAG:** trả lời đúng hoàn toàn — Meta → Llama 2 + Code Llama, TII → Falcon LLM, Anthropic → Claude 2.
- **GraphRAG:** **gán sai Llama 2 cho Google**. Nhìn vào subgraph thấy ngay nguyên nhân:

```
Google Cloud -USES-> Code Llama  | evidence=availability of Code Llama from Meta
Google Cloud -USES-> Claude 2    | evidence=pre-announcing Claude 2 from Anthropic
Google Cloud -USES-> Falcon LLM  | evidence=Technology Innovative Institute's Falcon LLM
Google Cloud -USES-> Llama 2     | evidence=Google Cloud is announcing availability of Llama 2
```

**Nguyên nhân gốc rễ — phép chiếu mất mát của schema allowlist.** Quan hệ cần biểu diễn là bộ ba
*(nhà cung cấp) → (mô hình) → (nền tảng phân phối)*, nhưng allowlist 8 quan hệ không có loại nào diễn đạt
“Meta là chủ sở hữu của Llama 2”. Model buộc phải ép mọi thứ vào `USES`, khiến **chủ thể sở hữu bị đẩy xuống
trường `evidence`** — một chuỗi văn bản mà LLM sinh câu trả lời không được yêu cầu đọc kỹ. Evidence recall vẫn = 1.0
(chunk đúng đã được lấy về), nên đây **không phải lỗi truy hồi mà là lỗi mô hình hoá dữ liệu**.

**Đề xuất khắc phục:**
1. Bổ sung quan hệ `OFFERS` / `DISTRIBUTES` (nền tảng phân phối mô hình) tách khỏi `DEVELOPED` (chủ sở hữu);
2. Ràng buộc trong prompt: khi câu văn có mẫu *“X from Y”*, bắt buộc sinh **hai** cạnh `Y-DEVELOPED->X` và
   `Platform-USES->X`;
3. Ở tầng sinh câu trả lời, nhắc model đọc `evidence` như dữ kiện bổ trợ chứ không chỉ đọc phần cấu trúc.

**Ca hỏng thứ ba đáng ghi nhận (`G5000-30`, evidence recall graph = 0):** seed chỉ khớp được `Meta`,
BFS thu 3 cạnh, không cạnh nào thuộc chunk bằng chứng. Nhánh vector của hybrid cũng trượt. Đây là ca
**seed nghèo** — câu hỏi hỏi về “hai bối cảnh AI khác nhau của Meta” nhưng không nêu thực thể thứ hai, nên
điểm xuất phát của BFS quá hẹp.

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent

#### 5.1 Quality vs Cost vs Latency

| Chiều | Flat RAG | Hybrid GraphRAG | Ghi chú |
|---|---|---|---|
| Chi phí **index** | 1 lần embedding 2.114 chunk (~1 phút CPU) | + **92 lượt gọi LLM** cho NER+RE (~14 phút) + coref 33 lượt (~6 phút) | Chênh lệch nằm hầu hết ở khâu offline |
| Chi phí **truy vấn** | 699 token/câu | 1.083 token/câu (**1.55×**) | Thêm 1 lượt LLM cho seed extraction |
| Độ trễ | 2.21 s | 2.16 s | Duyệt Neo4j (2 hop) ~50–150 ms, nhỏ hơn phương sai của LLM |
| Chất lượng | mốc | +0.84 multi-hop, +0.14 evidence recall | Lợi ích tập trung ở cross-doc/multi-hop |
| Bảo trì | index lại là xong | phải quản lý schema, entity resolution, super-node, provenance | Chi phí **con người** mới là phần đắt nhất |

**Kết luận vận hành:** GraphRAG không “đắt hơn khi chạy” mà **đắt hơn khi xây và khi bảo trì**. Với hệ thống
mà 80% câu hỏi là factoid, cách hợp lý là router phân loại câu hỏi rồi mới chọn nhánh.

#### 5.2 Đề xuất của AI Coding Agent mà tôi từ chối

1. **“Bỏ `merge_guard()` để tăng recall của entity resolution.”** Từ chối. False merge là loại lỗi **không thể
   phát hiện từ câu trả lời cuối cùng**: đồ thị vẫn trả lời trôi chảy, chỉ có điều nó khẳng định sai. Bằng chứng
   ủng hộ quyết định này nằm ngay trong audit: `generative AI` vs `Generative AI Capabilities` cosine 0.856 —
   nếu bỏ guard thì mọi phát biểu về một tính năng của ServiceNow sẽ bị quy cho cả ngành generative AI.
2. **“Dùng pairwise cosine O(N²) trên toàn corpus để near-dedup.”** Từ chối. Với 2.114 chunk là 2,2 triệu cặp —
   ở scale 350MB thì bất khả thi. Tôi cài **MinHash + LSH**: chỉ so **1.128 cặp** thay vì 2.232.241 cặp
   (giảm ~2.000 lần) mà vẫn bắt được các bản viết lại.
3. **“Tăng `top_k` và `edge_limit` cho chắc ăn.”** Từ chối. Bảng token cho thấy graph context đã đắt hơn 1,55×;
   nới trần chỉ làm nhiễu tăng nhanh hơn tín hiệu. Thay vào đó tôi thêm **self-correction** — chỉ mở rộng khi
   LLM tự đánh giá là context chưa đủ.
4. **“Lấy `chunks_df.head(400)` cho tiện.”** Từ chối. Dữ liệu sắp xếp theo `companyName` nên 400 dòng đầu chỉ có
   công ty vần 0/1/A/B. Tôi thay bằng **entity-anchored stratified sampling**: bắt buộc phủ 28/28 chunk bằng chứng
   vàng, ưu tiên chunk giàu thực thể, phần còn lại lấy ngẫu nhiên.

#### 5.3 Scale lên 350MB — bottleneck đầu tiên và cách xử lý

**Ngoại suy từ số đo thật:** 400 chunk cần 92 lượt gọi LLM, chạy ~14 phút dưới trần 8.000 token/phút
→ khoảng **110.000 token cho 400 chunk ≈ 275 token/chunk**. Toàn dump có 8.079.363 dòng; theo tỉ lệ hữu ích đo được
(2.114/5.000 = 42%) thì có ~3,4 triệu chunk → **≈ 930 triệu token chỉ cho riêng bước trích xuất**.

**Bottleneck #1 là số lượt gọi LLM cho NER+RE**, không phải throughput ghi Neo4j (271 cạnh nạp trong 1,6 giây —
ngoại suy tuyến tính thì 3,4 triệu cạnh vẫn chỉ khoảng vài giờ), cũng không phải embedding (MiniLM chạy CPU đã
xử lý 2.114 chunk trong ~60 giây).

**Kiến trúc đề xuất:**
1. **Lọc trước khi gọi LLM:** dùng NER cục bộ (spaCy/GLiNER) để loại chunk không chứa thực thể tổ chức nào —
   theo tỉ lệ đo được trên dataset này sẽ cắt ~50% lượt gọi.
2. **Hàng đợi worker bất đồng bộ** (Celery/Ray) + batch động theo token, thay vì vòng lặp tuần tự; trần rate-limit
   được quản lý tập trung bằng token bucket.
3. **Mô hình phân tầng:** model nhỏ cho pass 1 (phát hiện chunk có quan hệ), model lớn chỉ chạy trên phần đã lọc.
4. **Entity resolution phải bỏ FAISS Flat:** O(N²) trong từng loại là không khả thi ở 3,4 triệu mention →
   chuyển sang **HNSW + blocking** theo ký tự đầu/ticker, cộng cache canonical map.
5. **Near-dedup bắt buộc chạy trước trích xuất:** tin syndicated chiếm tỉ lệ đáng kể (đo được 24/2.114 ≈ 1,1% chỉ
   riêng phần MinHash bắt thêm được), mỗi bản trùng là một lượt LLM lãng phí và một cạnh trùng trong đồ thị.
6. **Phân vùng đồ thị theo community** để truy vấn không phải quét toàn cục.

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code

| Khái niệm trong bài giảng | Module | Hàm / khối code cụ thể | Quan sát thực tế & đánh giá |
|---|---|---|---|
| **Conservative Coreference** | M1 | `needs_coref()`, `resolve_coref_batch()`, `run_coref()` | Rule-gate cắt 43% lượt gọi. Nhưng 71/114 thay đổi chỉ là viết lại dấu câu → prompt cần ràng buộc span thay vì trả về full text |
| **Chunking & Exact dedup** | M1 | `chunk_text()`, `standardize_news()` | 2.687 → 2.114 bài; snippet ngắn nên tỉ lệ 1 chunk/bài, mọi multi-hop buộc phải nối giữa các tài liệu |
| **Schema & Allowlist Guard** | M2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS`, `parse_extraction_items()` | Allowlist chặn được rác nhưng cũng **làm mất thông tin** — chính là nguyên nhân ca lỗi G5000-28 |
| **Bulk Cypher Ingestion** | M2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | `UNWIND` batch 1.000: 393 node + 271 cạnh trong **1,6 giây** |
| **Edge Provenance** | M2 | `MERGE (s)-[r {source_chunk_id}]->(t)` | Đặt `source_chunk_id` trong `MERGE` (không phải `SET`) để hai bài báo khác nhau tạo hai cạnh — giữ tín hiệu cross-doc |
| **Entity Resolution & Union-Find** | M3 | `build_resolution_map()`, `merge_guard()`, `UF` | 23 dòng audit; phát hiện guard 0.72 quá lỏng cho `Person` |
| **Super-node Degree Cap** | M4 | `retrieve_graph_context()`, `recent_edges()` | Ở scale lab max degree = 21 → phải hạ ngưỡng tạm thời mới kiểm chứng được cơ chế |
| **Hybrid Context** | M4 | `answer_graph_rag()`, `textualize()` | `=== GRAPH ===` + `=== VECTOR ===`; nhánh vector cứu được các câu graph không phủ |
| **LLM-as-a-Judge** | M5 | `judge_answer()`, `run_evaluation()` | Bổ sung `evidence_recall()` để có thước đo khách quan song song với điểm chủ quan |
| **Community Detection** | Bonus | `build_communities()`, `community_report()` | 5 cụm lớn nhất sinh được “community report” dùng cho câu hỏi vĩ mô |

### 2. Quá trình Debugging & Bài học

**Lỗi khó nhất: trần token/ngày của Groq (200.000 TPD/model) cạn giữa chừng.**
Triệu chứng ban đầu gây hiểu nhầm: coref chạy 43 giây/batch trong khi extraction chỉ 17 giây/batch — tôi tưởng
prompt coref quá dài. Đọc kỹ thông điệp 429 mới thấy hai trần khác nhau: **8.000 token/phút** (rolling) và
**200.000 token/ngày**. Chuỗi xử lý:

1. Đo lại bằng header `x-ratelimit-*` → xác nhận trần theo **từng model**, không phải theo tài khoản.
2. Giảm nhu cầu: `reasoning_effort="low"` cho model gpt-oss, rule-gate cho coref (−43% lượt gọi),
   batch 4 cho extraction để tránh hỏng schema.
3. Định tuyến lại theo vai trò: NER+RE giữ trên Groq, phần còn lại sang OpenAI.
4. Chống mất mát: cache đĩa cho coref/triples + checkpoint tăng dần **sau mỗi batch**, và checkpoint đánh giá theo
   từng câu hỏi. Nhờ vậy lần chạy lại chỉ mất **3 phút** thay vì 28 phút, và một lần notebook bị lỗi ghi file
   (Windows `Errno 22` do IDE giữ file) cũng không làm mất kết quả nào.

**Bài học:** với pipeline LLM dài, **checkpoint và idempotency quan trọng ngang thuật toán**. Chi phí thêm ~30 dòng
code, nhưng nó biến mọi lần sửa lỗi từ “chạy lại nửa tiếng” thành “chạy lại ba phút”.

**Lỗi đáng nhớ thứ hai:** model nhỏ trả về `items` là **list chuỗi** thay vì list object → `AttributeError` làm chết
cả batch. Sửa bằng `parse_extraction_items()` phòng thủ: phần tử sai schema chỉ bị bỏ và đếm vào
`schema_violations`, kèm cơ chế hạ batch xuống từng chunk khi cả lô hỏng.

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)

**Bài toán:** hệ hỏi–đáp trên kho tài liệu nội bộ (hợp đồng, biên bản, quyết định) của một tổ chức.

**Có cần GraphRAG không?** — **Hybrid, không thuần graph.** Phần lớn câu hỏi là tra cứu điều khoản (factoid) —
số liệu của lab cho thấy GraphRAG không cải thiện gì ở nhóm này mà tốn thêm 30% token. Nhưng nhóm câu hỏi
“ai đã ký những hợp đồng nào với nhà cung cấp mà công ty X sở hữu” là multi-hop thật sự, nơi Flat RAG chỉ đạt
mức 3.28/5 trong khi Graph đạt 4.12/5.

**Cấu trúc Node/Relation dự kiến:**
- **Nodes:** `Organization`, `Person`, `Document`, `Contract`, `Department`, `Project`
- **Relations:** `SIGNED_BY`, `SUPERSEDES`, `REFERENCES`, `BELONGS_TO`, `SUPPLIES_TO`, `APPROVED_BY`
- **Provenance bắt buộc trên mọi cạnh:** `doc_id`, `page`, `clause_id`, `effective_date`, `evidence`, `confidence` —
  văn bản pháp lý bắt buộc trích dẫn được tới **điều khoản**, không chỉ tới tài liệu.

**Chiến lược Entity Resolution:** khối tổ chức có mã số thuế/mã đơn vị → dùng **định danh cứng làm khoá chính**,
vector chỉ dùng để *gợi ý* cho người kiểm duyệt, không tự động gộp. Đây là bài học trực tiếp từ cặp
`Sam Altman` / `Steve Altman`: khi thực thể là người hoặc pháp nhân, false merge có hậu quả pháp lý.

**Chiến lược Super-node:** node kiểu “Phòng Hành chính” sẽ nối tới hàng nghìn văn bản. Cắt theo thời gian là chưa đủ;
kế hoạch là **cắt theo loại quan hệ suy ra từ câu hỏi** (câu hỏi về ký kết → chỉ mở cạnh `SIGNED_BY`/`APPROVED_BY`),
kết hợp trần cứng theo hop và `effective_date DESC` làm tiêu chí phụ.

---

## 🎁 Bonus đã thực hiện

| Bonus | Kết quả định lượng | File |
|---|---|---|
| **B. Global Search qua Community Detection** | 145 community; 5 cụm lớn nhất được LLM tóm tắt (“Microsoft and AI Innovations” 27 cạnh, “AI and Cloud Innovations” 21 cạnh…). Câu hỏi vĩ mô trả lời bằng community report thay vì BFS cục bộ | `outputs/community_reports.csv` |
| **C. Self-Correction Retrieval** | 3 câu khó nhất: `G5000-33` hop2→**hop3** (8→27 cạnh, judge 5→7), `G5000-50` → **hop3+vector** (judge 8→11), `G5000-30` giữ hop2 (judge 6→4). Tổng điểm judge **+3** | `outputs/self_correction_results.csv` |
| **A. Near-dedup MinHash + LSH** | 96 hash, 24 band; so **1.128 cặp** thay vì 2.232.241 cặp O(N²); phát hiện thêm **24 chunk trùng gần** mà SHA-1 bỏ sót (ví dụ `CereCore® expands…` vs `CereCore expands…`, Jaccard 1.0) | `outputs/near_duplicate_pairs.csv` |

---

## 🎯 TỰ ĐÁNH GIÁ

| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | 5 | Giải thích được vì sao lợi ích tập trung ở cross-doc (+1.27 multi-hop) và vì sao factoid hoà |
| Khả năng kiểm soát AI Coding Agent | 5 | Từ chối 4 đề xuất có hại, mỗi lần đều kèm bằng chứng số liệu |
| Chất lượng đồ thị tri thức xây dựng | 4 | Provenance 100%, audit minh bạch; trừ điểm vì allowlist quá hẹp gây mất thông tin (ca G5000-28) |
| Khả năng phân tích và debug hệ thống | 5 | Truy được nguyên nhân gốc của cả rate-limit lẫn ca lỗi GraphRAG bằng chỉ số Evidence Recall |

---

## 📎 Phụ lục — tệp kết quả

| Tệp | Nội dung |
|---|---|
| `outputs/graphrag_eval_results.csv` | 25 câu × điểm judge, latency, token, evidence recall, seed, số cạnh |
| `outputs/graphrag_vs_flatrag_summary.csv` | bảng so sánh theo nhóm câu hỏi + toàn bộ |
| `outputs/failure_case_ranking.csv` | xếp hạng ca lỗi theo `delta_total` |
| `outputs/entity_resolution_audit.csv` · `..._threshold_sweep.csv` · `lexical_guard_probe.csv` | bằng chứng Entity Resolution |
| `outputs/community_reports.csv` · `self_correction_results.csv` · `near_duplicate_pairs.csv` · `dedup_summary.csv` | bằng chứng 3 bonus |
| `outputs/eval_traces.json` | trace retrieval từng câu (chunk, cạnh, seed) |
| `outputs/demo_data.json` | dữ liệu cho giao diện demo `demo/index.html` |
