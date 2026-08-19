# Phân tích Ca lỗi (Root-Cause Analysis) — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Dương Ngọc Hải — 2A202601748 · AICB-K34 Track 3
**Nguồn số liệu:** `outputs/failure_case_ranking.csv`, `outputs/eval_traces.json`, `outputs/graphrag_eval_results.csv`

---

## Phương pháp truy vết

Mỗi câu hỏi được chấm bởi LLM Judge trên 3 trục (1–5). Đặt:

```
delta_total = Σ (điểm GraphRAG − điểm Flat RAG)   trên 3 trục
```

- `delta_total` **cao nhất** → ca Flat RAG thất bại / GraphRAG thắng.
- `delta_total` **thấp nhất** → ca GraphRAG thất bại.

Song song, mỗi câu có một chỉ số **khách quan** không phụ thuộc Judge:

```
evidence_recall = |chunk bằng chứng vàng ∩ chunk hệ thống lấy về| / |chunk bằng chứng vàng|
```

Chỉ số này là chìa khoá của toàn bộ phân tích dưới đây, vì nó **tách bạch hai loại lỗi**:

| evidence_recall | Điểm Judge | Chẩn đoán |
|---|---|---|
| thấp | thấp | Hỏng ở khâu **truy hồi** (retrieval) |
| = 1.0 | thấp | Hỏng ở khâu **sinh câu trả lời hoặc mô hình hoá dữ liệu** |

---

## Ca lỗi 1 — Flat RAG thất bại, GraphRAG thành công

**`G5000-43`** · nhóm `cross-doc` · `delta_total = +10` (ca chênh lệch lớn nhất trong 25 câu)

> **Câu hỏi:** *Which came first in the selected HPE timeline: the Axis Security acquisition agreement or the
> LLM-focused cloud service announcement, and what does that ordering show?*

> **Đáp án chuẩn:** Thương vụ Axis Security đến trước (02/03/2023); bài về dịch vụ cloud hướng LLM của HPE
> đến sau (21/06/2023). Thứ tự cho thấy HPE mở rộng năng lực bảo mật trước khi triển khai dịch vụ AI-cloud.

### Số liệu

| | Flat RAG | GraphRAG |
|---|---|---|
| Evidence recall | **0.50** | **1.00** |
| Comprehensiveness | 2 | **5** |
| Faithfulness | 2 | **5** |
| Multi-hop reasoning | 2 | **5** |

### Flat RAG đã làm gì

Top-6 chunk chỉ chứa bài về Axis Security. Bài thứ hai (`row03289::c0000`) **không lọt vào top-k** vì nó nói về
“cloud computing service for artificial intelligence” — không dùng từ *acquisition*, *Axis*, hay *timeline*, nên
cosine similarity với câu hỏi thấp hơn các chunk khác.

Câu trả lời sinh ra tự tố cáo chính nó:

> *“The Axis Security acquisition agreement came first on March 2, 2023, while the LLM-focused cloud service
> announcement **is not explicitly mentioned in the provided context**. … The ordering of the acquisition suggests
> that HPE is prioritizing the enhancement of its security capabilities…”*

Đây là hai lỗi chồng lên nhau: **thiếu dữ kiện** (không có mốc thời gian thứ hai) và **suy diễn không căn cứ**
(vẫn kết luận về chiến lược dù thừa nhận thiếu bằng chứng). Judge ghi rõ: *“makes unsupported claims about HPE's
strategic focus without evidence.”*

### GraphRAG đã giải quyết ra sao

1. **Seed resolution:** `HPE` khớp **EXACT** vào node `Hewlett Packard Enterprise` — nhờ alias thủ công
   `"hpe" → "Hewlett Packard Enterprise"` trong `MANUAL_ALIASES`. Không có alias này, seed sẽ phải rơi xuống
   tầng vector và có nguy cơ trượt.
2. **BFS 2 hop** thu về đúng hai cạnh, mỗi cạnh mang `published_date` là **thuộc tính hạng nhất**:

```
Hewlett Packard Enterprise [Company] -ACQUIRED-> Axis Security [Company]
    | date=2023-03-02 | chunk=row04762::c0000
    | evidence=HPE today announced that it entered into a definitive agreement to acquire Axis Security…

Hewlett Packard Enterprise [Company] -USES-> supercomputers [Technology]
    | date=2023-06-21 | chunk=row03289::c0000
    | evidence=HPE to offer cloud computing service for artificial intelligence…
```

3. **Sắp xếp thời gian trở thành thao tác dữ liệu**, không còn phụ thuộc việc LLM có “nhìn thấy” cả hai đoạn văn.

### Nguyên nhân gốc rễ

Flat RAG hỏng ở **khâu truy hồi**, không phải khâu sinh câu trả lời (recall 0.5). Cơ chế xếp hạng của nó — độ tương
đồng ngữ nghĩa giữa câu hỏi và từng chunk **độc lập** — về bản chất không thể phát hiện rằng hai tài liệu *cần đi
cùng nhau*. GraphRAG đảo ngược thứ tự: nó tìm **thực thể chung** trước, rồi mới thu thập mọi điều đã biết về thực thể đó.

### Bài học tổng quát

Loại câu hỏi mà GraphRAG thắng đậm không phải “câu hỏi khó”, mà là **câu hỏi có ràng buộc quan hệ hoặc thời gian
trải trên nhiều tài liệu**. Số liệu theo nhóm xác nhận: cross-doc là nhóm chênh lệch multi-hop reasoning lớn nhất
(2.64 → 3.91, **+1.27**), trong khi factoid hoà tuyệt đối 5.00/5.00.

---

## Ca lỗi 2 — GraphRAG thất bại

**`G5000-28`** · nhóm `multi-hop` · `delta_total = −5` (ca GraphRAG kém nhất)

> **Câu hỏi:** *Which model providers are connected to Google Cloud Next '23 in the selected data, and which models
> are associated with each provider?*

> **Đáp án chuẩn:** Meta ↔ Llama 2 và Code Llama; Technology Innovation Institute ↔ Falcon LLM;
> Anthropic ↔ Claude 2 (Google Cloud pre-announce).

### Số liệu

| | Flat RAG | GraphRAG |
|---|---|---|
| Evidence recall | **1.00** | **1.00** |
| Comprehensiveness | **5** | 3 |
| Số cạnh thu được | — | 6 |
| Seed khớp | — | `Google Cloud` (khớp 2 lần) |

**Điểm mấu chốt: cả hai đều có evidence recall = 1.0.** Chunk bằng chứng `row03395::c0000` nằm trong ngữ cảnh
của cả hai hệ. Vậy đây **không phải lỗi truy hồi**.

### Hai câu trả lời

**Flat RAG — đúng hoàn toàn:**
> Meta → Llama 2 và Code Llama · Technology Innovative Institute → Falcon LLM · Anthropic → Claude 2 (pre-announced)

**GraphRAG — sai một dòng:**
> Meta → Code Llama · Anthropic → Claude 2 · Technology Innovative Institute → Falcon LLM · **Google → Llama 2** ❌

### Truy vết vào subgraph

```
Google Cloud -USES-> Code Llama  | evidence=availability of Code Llama from Meta
Google Cloud -USES-> Claude 2    | evidence=pre-announcing Claude 2 from Anthropic
Google Cloud -USES-> Falcon LLM  | evidence=Technology Innovative Institute's Falcon LLM
Google Cloud -USES-> Llama 2     | evidence=Google Cloud is announcing availability of Llama 2
Google Cloud -DEVELOPED-> Multiomics Suite | date=2023-05-20 | chunk=row02794::c0000
Form Bio     -USES-> Google Cloud          | date=2023-05-20 | chunk=row02794::c0000
```

Cấu trúc đồ thị nói: *Google Cloud dùng Llama 2*. Thông tin “Llama 2 **thuộc về Meta**” chỉ tồn tại trong chuỗi
`evidence` của ba cạnh đầu — và ở cạnh thứ tư thì **không tồn tại ở đâu cả**, vì câu văn gốc chỉ nói
“Google Cloud is announcing availability of Llama 2”.

### Nguyên nhân gốc rễ — phép chiếu mất mát của schema allowlist

Quan hệ thực tế trong dữ liệu là **bộ ba**:

```
(nhà cung cấp)  --sở hữu-->  (mô hình)  --được phân phối bởi-->  (nền tảng)
     Meta                     Llama 2                            Google Cloud
```

`ALLOWED_RELATIONS` chỉ có 8 loại: `ACQUIRED, DEVELOPED, INVESTED_IN, FOUNDED, WORKED_AT, PARTNERED_WITH, USES, LEADS`.
**Không loại nào diễn đạt được quan hệ “nền tảng phân phối mô hình của bên thứ ba”.** Model buộc phải ép tất cả
vào `USES`, và trong phép ép đó **chủ thể sở hữu bị đẩy khỏi cấu trúc, rơi xuống trường văn bản tự do `evidence`**.

Đây là một dạng lỗi rất đáng chú ý về mặt kiến trúc: **schema allowlist vừa là cơ chế phòng vệ vừa là nguồn gây
mất thông tin**. Nó chặn được rác (ca lỗi coref ở phần 1 của thuyết minh), nhưng khi thực tế phong phú hơn schema,
nó tạo ra những cạnh *đúng cú pháp nhưng sai ngữ nghĩa*.

Flat RAG **không gặp lỗi này** chính vì nó không mô hình hoá gì cả — nó đưa nguyên văn bản cho LLM, và câu văn gốc
đã nói rõ “Code Llama **from Meta**”.

### Đề xuất khắc phục

1. **Mở rộng schema có kiểm soát:** thêm `OFFERS` / `DISTRIBUTES` (nền tảng phân phối) tách bạch khỏi `DEVELOPED`
   (chủ sở hữu). Chi phí: một dòng trong allowlist; lợi ích: bộ ba trên biểu diễn được đúng.
2. **Ràng buộc mẫu trong prompt trích xuất:** khi câu văn khớp mẫu *“X from Y”* / *“Y's X”*, bắt buộc sinh **hai**
   cạnh — `Y -DEVELOPED-> X` và `Platform -OFFERS-> X` — thay vì một.
3. **Ở tầng sinh câu trả lời:** nhắc model rằng `evidence` là dữ kiện bổ trợ cần đọc, không chỉ là chú thích nguồn.
4. **Kiểm thử hồi quy:** thêm chính câu `G5000-28` vào bộ kiểm thử để lần sửa schema sau này có mốc đối chiếu.

---

## Ca lỗi 3 (bổ sung) — Seed nghèo làm BFS xuất phát sai chỗ

**`G5000-30`** · `multi-hop` · **evidence_recall của GraphRAG = 0.00** (thấp nhất trong 25 câu)

> **Câu hỏi:** *Meta appears in two different AI contexts in the selected data. What are they, and what distinct
> relation should the graph store in each case?*

| | Flat RAG | GraphRAG |
|---|---|---|
| Evidence recall | 0.50 | **0.00** |
| Comprehensiveness | 2 | 2 |
| Seed khớp | — | chỉ `Meta` |
| Số cạnh | — | 3 |

**Nguyên nhân gốc rễ:** câu hỏi **cố ý không nêu thực thể thứ hai** (“hai bối cảnh AI khác nhau” là thứ hệ thống
phải tự tìm ra). Seed extraction chỉ rút được `Meta`, BFS xuất phát từ một điểm duy nhất và 3 cạnh thu được đều
không thuộc chunk bằng chứng. Nhánh vector của hybrid cũng trượt.

**Bài học:** GraphRAG **kế thừa toàn bộ điểm yếu của bước seed extraction**. Câu hỏi khám phá (“có những gì…”)
khác hẳn câu hỏi xác minh (“X có quan hệ gì với Y”): nó cần **entry point rộng** — ví dụ lấy top-N chunk bằng vector
trước, rút thực thể từ chính các chunk đó rồi mới làm seed, thay vì chỉ rút seed từ câu hỏi. Đây cũng chính là
động lực của bonus **Self-Correction** (mở rộng hop khi context chưa đủ) và của **Global Search** qua community
report cho câu hỏi vĩ mô.

---

## Tổng hợp: bản đồ chế độ hỏng

| # | Chế độ hỏng | Ca quan sát được | evidence_recall | Tầng chịu trách nhiệm |
|---|---|---|---|---|
| 1 | **Fragmented context** — hai dữ kiện ở hai tài liệu | `G5000-43` (Flat thua) | Flat 0.50 | Retrieval (Flat RAG) |
| 2 | **Lossy schema projection** — quan hệ thực tế không có trong allowlist | `G5000-28` (Graph thua) | cả hai 1.00 | Mô hình hoá dữ liệu (M2) |
| 3 | **Seed nghèo** — câu hỏi khám phá, không nêu đủ thực thể | `G5000-30` (cả hai kém) | Graph 0.00 | Seed extraction (M4) |
| 4 | **False coreference** — đại từ phân giải về cụm chung | `row00219::c0000` | — | Tiền xử lý (M1) — đã chặn được |
| 5 | **False merge thực thể** — cosine cao, ngữ nghĩa khác | `generative AI` vs `Generative AI Capabilities` (0.856) | — | Entity Resolution (M3) — Guard chặn |
| 6 | **Gộp hụt thực thể** — lexical khớp nhưng cosine dưới ngưỡng | `Synopsys` vs `Synopsys Inc.` (0.832) | — | Entity Resolution (M3) — **chưa khắc phục** |

**Ba chế độ đầu đã có bằng chứng định lượng từ 25 câu benchmark; ba chế độ sau được kiểm chứng bằng probe có kiểm
soát** (`outputs/lexical_guard_probe.csv`, `outputs/entity_resolution_audit.csv`).

Chế độ **#6 là nợ kỹ thuật đang tồn tại**: thứ tự đúng phải là *lexical-first* (khớp tuyệt đối sau khi chuẩn hoá hậu tố
thì gộp ngay, không cần hỏi vector). Tôi ghi nhận thay vì âm thầm nới ngưỡng cosine xuống, vì hạ ngưỡng sẽ kéo theo
`Sam Altman` / `Steve Altman` (cosine 0.824, lexical ratio 0.727 > guard 0.72) — đánh đổi một lỗi phân mảnh lấy một
lỗi gộp sai người, tức là làm mọi thứ tệ hơn.
