# LegalIR Project Catch-up — đọc một lần để hiểu toàn bộ project

## 1. Nếu ai cũng dùng AI, khác biệt nằm ở đâu?

AI chỉ giúp mọi đội viết code nhanh hơn. AI không tự biết cấu trúc dữ liệu của cuộc thi, không tự tạo validation đúng, không tự phát hiện pipeline nào thật sự tăng Recall và cũng không chịu trách nhiệm khi submission sai format.

Cuộc thi giống một cuộc đua xe:

- AI là bộ dụng cụ và người trợ lý cơ khí.
- Dataset là đường đua.
- Local validation là đồng hồ đo tốc độ.
- Error analysis là dữ liệu telemetry.
- Retriever, fusion và reranker là động cơ.
- Team thắng là team đo đúng, sửa đúng điểm yếu và lặp thử nghiệm tốt hơn.

Lợi thế cạnh tranh thực tế nằm ở sáu chỗ:

1. **Hiểu dữ liệu:** Điều/Khoản/Điểm, mã văn bản, title, passage và quan hệ question → document ID.
2. **Đánh giá đúng:** dùng validation cố định, không chọn model dựa trên cảm giác hoặc một lần public submit.
3. **Candidate generation tốt:** BM25 tìm từ khóa; dense retrieval tìm nghĩa tương tự.
4. **Fusion và reranking:** hợp nhất các candidate rồi chọn đúng 5 ID cuối.
5. **Error analysis:** biết model sai vì thiếu từ khóa, khác cách diễn đạt, nhầm văn bản gần giống hay mất metadata.
6. **Kỷ luật thử nghiệm:** mỗi thay đổi có hypothesis, metric và artifact để tái lập.

Nếu hai đội đều hỏi AI “viết BM25 cho tôi”, họ gần như có cùng baseline. Đội hơn nhau ở những vòng đo → phân tích lỗi → cải tiến → đo lại sau đó.

## 2. Bài toán đang giải

Input:

```json
{
  "38096": {
    "question": "Đề nghị xem xét lại quyết định đình chỉ...",
    "answer": null
  }
}
```

Output **mà model tự dự đoán và cần nộp**:

```json
{
  "38096": {
    "answer": ["20457", "151530", "278236", "275852", "216294"]
  }
}
```

Các ID trên chỉ là ví dụ prediction của baseline, được chọn từ 8.532 document trong corpus; chúng không phải đáp án BTC cung cấp. `answer` trong `public-official.json` vẫn là `null` vì BTC giấu ground truth. Hệ thống của mình phải tự tìm tối đa 5 document ID từ corpus.

## 3. Luồng chạy hiện tại

```text
train.json (7.000 câu có nhãn)
      │
      ├── split_legalir_train.py
      │       ├── legalir_train.json (5.997)
      │       └── legalir_dev.json   (1.003)
      │
      ├── build_legalir_query_profiles.py
      │       └── question profile cho từng document
      │
contexts/*.json (8.532 văn bản)
      │
      └───────────────┬─────────────────────────┐
                      ▼                         │
              legalir_baseline.py              │
              tạo BM25 index                    │
              xếp hạng document                 │
                      │                         │
                      ▼                         │
              prediction top 5                 │
                │             │                 │
                ▼             ▼                 │
        evaluate_legalir   validate_submission  │
        Recall/Precision   kiểm tra JSON         │
                                                │
public-official.json (1.000 câu, không nhãn) ───┘
                      │
                      ▼
         submission.json → submission.zip
```

## 4. Cấu trúc thư mục

```text
UIT Data Science 2026/
├── data/
│   ├── real/
│   │   ├── LegalIR - Public Test/
│   │   │   ├── train.json
│   │   │   ├── public-official.json
│   │   │   ├── selected-contexts.zip
│   │   │   └── DSC2026_Task1_LegalIR_Data_Overview.docx
│   │   └── contexts/                 # 8.532 context JSON đã giải nén
│   ├── derived/                      # dữ liệu do code mình sinh ra
│   └── sample_legalir/               # dữ liệu giả để học pipeline
├── src/                              # source code Python
├── scripts/                          # lệnh chạy end-to-end
├── submissions/                      # prediction và ZIP chuẩn bị nộp
├── reports/experiments/              # kết quả từng experiment
├── docs/                             # tài liệu học và quản lý
└── tmp/                              # file trung gian
```

### Nguyên tắc

- `data/real/`: dữ liệu BTC, không tự sửa.
- `data/derived/`: có thể tái tạo bằng script.
- `src/`: logic của hệ thống.
- `submissions/`: output, không phải source code.
- `reports/`: bằng chứng một thay đổi tốt hay xấu.

## 5. Ý nghĩa từng source file

### `src/legalir_baseline.py`

File trung tâm của retriever.

Nó làm các bước:

1. Đọc toàn bộ corpus.
2. Chuẩn hóa text và tách token.
3. Có thể bỏ dấu tiếng Việt để query khớp với title dạng URL slug.
4. Có thể bỏ stopword.
5. Index `name + passage`.
6. Có thể thêm các câu train từng trỏ đến document vào query profile.
7. Tính BM25 score.
8. Trả về 5 document ID điểm cao nhất.

Các option quan trọng:

```text
--include-title       index cả tên văn bản
--fold-accents        chuẩn hóa chữ có dấu/không dấu
--remove-stopwords    bỏ từ đệm phổ biến
--query-profiles      thêm kiến thức từ câu hỏi train
--top-k 5             trả tối đa 5 ID
```

### `src/split_legalir_train.py`

Chia 7.000 câu có nhãn thành train/dev cố định. Việc chia cố định giúp mọi experiment được so trên cùng 1.003 câu dev.

### `src/build_legalir_query_profiles.py`

Từ dữ liệu có nhãn:

```text
question → correct document ID
```

script gom các câu hỏi đúng theo document. Khi có câu hỏi mới diễn đạt tương tự, BM25 có thêm cơ hội tìm đúng document đó.

Khi đo dev, profile chỉ được xây từ train split để không nhìn trước đáp án dev. Khi chạy Public, profile được xây bằng toàn bộ 7.000 câu train.

### `src/evaluate_legalir.py`

So prediction với đáp án thật và tính:

```text
Recall    = ID đúng tìm được / tổng ID đúng
Precision = ID đúng tìm được / tổng ID trả về
```

File này chỉ đo được train/dev/warmup có nhãn. Không thể đo Public Test vì `answer` bị ẩn.

### `src/validate_legalir_submission.py`

Kiểm tra đủ question ID, answer đúng kiểu list, không trùng ID và không vượt quá 5 kết quả.

### `src/inspect_warmup.py`

Script học tập để xem cấu trúc và thống kê `warmup.json`; không tham gia trực tiếp vào pipeline Public hiện tại.

## 6. Ý nghĩa các script chạy

### `scripts/run_legalir_sample.sh`

Chạy dữ liệu giả lập để học. Không dùng output này để nộp.

### `scripts/run_legalir_public_baseline.sh`

Chạy candidate tốt nhất hiện tại:

1. Xây query profile từ 7.000 câu train.
2. Index corpus bằng normalized BM25.
3. Predict 1.000 câu Public Test.
4. Validate output.

Chạy bằng:

```bash
cd "/Users/anhtran/Documents/ChatGPT/UIT Data Science 2026"
bash scripts/run_legalir_public_baseline.sh
```

## 7. Các kết quả hiện tại

Trên validation 1.003 câu:

| Experiment | Recall | Precision | Hit@5 |
|---|---:|---:|---:|
| Raw BM25 | 0.3431 | 0.0724 | 0.3549 |
| + title, normalize, stopword | 0.4189 | 0.0889 | 0.4347 |
| + train query profiles | 0.4234 | 0.0899 | 0.4397 |
| + Dense E5 + RRF (top-5 candidates) | 0.5571 | 0.1174 | 0.5773 |
| **+ Dense E5 + RRF (100 candidates each)** | **0.5768** | **0.1214** | **0.5962** |
| **+ tune RRF k=40** | **0.5847** | **0.1232** | **0.6052** |
| **+ labelled-question nearest neighbour + 3-way fusion** | **0.7311** | **0.1545** | **0.7468** |
| **V5: safe reranker only for slot 5** | **0.8059** | **0.1703** | **0.8325** |

Leaderboard đang khoảng Recall 0.93–0.96. Điều này chứng minh lexical retrieval đã cải thiện nhưng chưa đủ cạnh tranh.

## 8. Những file submission khác nhau

- `legalir_public_bm25_k5.*`: BM25 thô đầu tiên.
- `legalir_public_bm25_normalized_k5.*`: thêm title/normalize/stopword.
- `legalir_public_hybrid_top100_rrf_k5.json`: output của pipeline tốt nhất hiện tại.
- `legalir_public_best.zip`: file sẵn sàng upload, bên trong có đúng một file `submission.json`.

Không nộp nhầm file train/dev/sample. File chuẩn nộp phải là ZIP chứa duy nhất `submission.json`.

## 9. Bạn nên đọc theo thứ tự nào?

1. Đọc `data/real/.../public-official.json` để hiểu input.
2. Mở một file trong `data/real/contexts/` để hiểu corpus.
3. Đọc `src/legalir_baseline.py`, tập trung `tokens`, `Bm25Index.__init__` và `rank`.
4. Đọc `src/evaluate_legalir.py` để hiểu điểm số.
5. Đọc `reports/experiments/legalir_dev_bm25_comparison.md` để thấy code thay đổi metric ra sao.
6. Cuối cùng đọc `scripts/run_legalir_public_best.sh` để nối tất cả thành một pipeline.

## 10. Roadmap để cạnh tranh

```text
Current best: V5 safe reranker (local Recall 0.8059 / Precision 0.1703;
Public Recall 0.8020 / Precision 0.1724)
    ↓
Dense retriever chạy local
    ↓
BM25 + dense Reciprocal Rank Fusion
    ↓
Cross-encoder reranker cho top 50
    ↓
Legal-aware features: mã văn bản, Điều/Khoản/Điểm
    ↓
Error analysis và tune cutoff/fusion weight
```

AI có thể giúp viết từng khối, nhưng lợi thế của mình đến từ việc mỗi khối đều được kiểm tra trên validation và chỉ giữ thay đổi thật sự tăng Recall.
