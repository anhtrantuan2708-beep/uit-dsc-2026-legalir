# LegalIR baseline — Implementation Guide cho người mới

## 1. Bức tranh tổng quát

LegalIR không sinh câu trả lời. Nó nhận một câu hỏi và tìm ra những văn bản pháp luật liên quan nhất.

```text
questions.json                         contexts/*.json
      │                                       │
      └──────────────┬────────────────────────┘
                     ▼
              legalir_baseline.py
                     │
                     ▼
       submission.json: document IDs theo câu hỏi
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
  validate_submission.py   evaluate_legalir.py
  kiểm tra format           đo Recall/Precision
```

Khi chạy public test, phần đáp án đúng bị ẩn. Vì vậy local evaluator chỉ dùng cho warmup/train hoặc dữ liệu mẫu; điểm public do Codabench trả về sau submission.

## 2. Cấu trúc project

```text
UIT Data Science 2026/
├── data/
│   ├── raw/warmup.json              # warmup thật đang có
│   └── sample_legalir/              # sân tập giả lập
│       ├── queries.json             # câu hỏi + document ID đúng
│       └── contexts/                # corpus passage giả lập
├── src/
│   ├── legalir_baseline.py          # tìm văn bản bằng lexical/BM25 đơn giản
│   ├── validate_legalir_submission.py # kiểm tra submission.json
│   ├── evaluate_legalir.py          # đo local Recall/Precision
│   └── inspect_warmup.py             # xem nhanh warmup.json
├── scripts/
│   └── run_legalir_sample.sh        # chạy toàn bộ pipeline mẫu
├── submissions/                     # các file dự đoán sinh ra
├── reports/experiments/             # metric và ghi chú thí nghiệm
└── docs/                            # guide, plan và research
```

## 3. Dữ liệu vào

### 3.1. File câu hỏi

```json
{
  "q001": {
    "question": "Thời hạn giải quyết hồ sơ đăng ký kinh doanh là bao lâu?",
    "answer": ["doc_kinh_doanh"]
  }
}
```

Trong dữ liệu thi thật, `answer` dùng để đánh giá warmup/train. Trong public test, file câu hỏi thường không cho bạn biết đáp án đúng.

### 3.2. File corpus

```json
{
  "id": "doc_kinh_doanh",
  "name": "Tên văn bản",
  "link": "https://...",
  "passage": "Nội dung điều khoản pháp luật..."
}
```

Baseline đọc tất cả file `.json` trong thư mục corpus, sau đó dùng `id` làm kết quả trả về.

## 4. Code đang làm gì?

### `legalir_baseline.py`

1. Đọc câu hỏi.
2. Đọc các passage trong corpus.
3. Tách từ bằng regex, chuyển thành chữ thường.
4. Đếm tần suất từ trong từng document.
5. Tính điểm BM25:
   - từ xuất hiện trong câu hỏi và document thì được cộng điểm;
   - từ hiếm trong corpus được ưu tiên hơn;
   - document quá dài bị điều chỉnh điểm.
6. Sắp xếp document theo điểm.
7. Lấy tối đa `top-k` ID và ghi thành submission JSON.

Index được tạo một lần rồi tái sử dụng cho mọi câu hỏi, nên có thể chạy trên corpus thật 8.533 passage. Đây là baseline để có hệ thống chạy được, chưa phải model AI hay semantic search.

### `validate_legalir_submission.py`

Kiểm tra:

- Có đủ question ID không.
- Có question ID lạ không.
- `answer` có phải list không.
- Có quá 5 ID không.
- Có ID trùng nhau không.

### `evaluate_legalir.py`

Với mỗi câu hỏi:

```text
Recall    = số ID đúng được tìm thấy / tổng số ID đúng
Precision = số ID đúng được tìm thấy / số ID model trả về
```

Sau đó lấy trung bình trên toàn bộ câu hỏi.

### `run_legalir_sample.sh`

Đây chỉ là file điều phối. Nó chạy lần lượt:

```text
baseline → validator → evaluator
```

## 5. Cách chạy từ đầu

Mở Terminal và chạy:

```bash
cd "/Users/anhtran/Documents/ChatGPT/UIT Data Science 2026"
bash scripts/run_legalir_sample.sh
```

Kết quả hiện tại:

```text
queries: 5
missing predictions: 0
hit@k: 1.0000
macro recall: 1.0000
macro precision: 0.4400
```

File kết quả nằm ở:

```text
submissions/sample_legalir_k5.json
```

## 6. Tự chạy từng bước

### Bước A — Sinh dự đoán

```bash
python3 src/legalir_baseline.py \
  data/sample_legalir/queries.json \
  data/sample_legalir/contexts \
  submissions/my_prediction.json \
  --top-k 5
```

Phiên bản lexical tốt hơn hiện tại thêm các option:

```bash
--include-title --fold-accents --remove-stopwords
```

Nó index cả `name` lẫn `passage`, đưa query/title về dạng không dấu để khớp nhau, và bỏ các từ đệm phổ biến.

### Bước B — Kiểm tra file

```bash
python3 src/validate_legalir_submission.py \
  data/sample_legalir/queries.json \
  submissions/my_prediction.json
```

### Bước C — Đo điểm local

```bash
python3 src/evaluate_legalir.py \
  data/sample_legalir/queries.json \
  submissions/my_prediction.json
```

## 7. Khi có corpus thật của BTC

Không sửa code ngay. Chỉ thay đường dẫn:

```bash
python3 src/legalir_baseline.py \
  data/real/public-questions.json \
  data/real/contexts \
  submissions/legalir_public_baseline.json \
  --top-k 5
```

Sau đó validate. Không thể đo Recall/Precision public local nếu không có ground truth; hãy upload đúng file theo submission contract của Codabench và ghi lại public score.

Với bộ Public Test hiện tại trong project, chạy:

```bash
bash scripts/run_legalir_public_baseline.sh
```

Lệnh này tạo `submissions/legalir_public_bm25_k5.json` và kiểm tra format. Nó không upload tự động lên Codabench.

## 8. Thứ tự nâng cấp sau baseline

1. Normalize Unicode và xử lý tiếng Việt tốt hơn.
2. Tăng trọng số cho mã văn bản, số Điều/Khoản và thuật ngữ pháp lý.
3. Thử `top-k = 1, 3, 5`.
4. Thêm dense embedding.
5. Kết hợp lexical + dense bằng RRF.
6. Rerank 20–50 kết quả đầu.

Không nâng cấp bước sau khi baseline còn chưa chạy ổn định.

## 9. Điều cần nhớ

- Dữ liệu trong `data/sample_legalir/` là giả lập.
- Điểm 1.0 trên dữ liệu giả lập không dự đoán điểm thi.
- `warmup.json` một mình không đủ để retrieval vì thiếu passage.
- Public Test có thể chấm ẩn; public score là tín hiệu, không phải ground truth toàn bộ.
