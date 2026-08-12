# Research status — 2026-08-08

## 1. Kết quả kiểm tra Drive

Folder Drive:

https://drive.google.com/drive/folders/13FimbavEhVcSWHcNQ8fSiWyfZxIwlkKA

Hiện folder chỉ hiển thị một file:

- `warmup.json`

Không thấy `selected-contexts.zip`, `train.json`, `public-official.json` hoặc `private-official.json` trong folder này.

## 2. Nội dung `warmup.json`

File này có 500 mẫu dạng:

```json
{
  "147194": {
    "question": "...",
    "answer": ["14681"]
  }
}
```

Đây là format của **Legal Information Retrieval (LegalIR)**: câu hỏi trỏ đến các `document_id` đúng.

## 3. Kết quả kiểm tra Codabench

URL:

https://www.codabench.org/competitions/10153/#/phases-tab

Trang hiện hiển thị:

- Tên: `UIT-DSC 2025: UIT Data Science Challenge`.
- Organizer: `UIT-DSC 2026 Organizers`.
- Current Active Phase: `None`.
- Vòng Warmup: 29/08/2025–03/09/2025.
- Warmup yêu cầu file `submit.csv` với đúng 2 cột:
  - `id`
  - `predict_label`
- `predict_label` thuộc một trong ba nhãn:
  - `no`
  - `intrinsic`
  - `extrinsic`

Trang này đang mô tả **hallucination classification**, không khớp với LegalIR `warmup.json`.
Forum của trang (`/forums/9996/`) chỉ hiển thị yêu cầu gửi registration request, không có file dữ liệu công khai.

## 4. Kết luận hiện tại

Chưa được phép giả định rằng `selected-contexts.zip` bị mất. Khả năng cao là một trong các trường hợp:

1. File nằm trong một Drive folder khác.
2. File được cấp qua Codabench sau khi vào đúng competition/team.
3. Link `10153` là competition khác hoặc phiên bản cũ.
4. LegalIR và hallucination classification là hai bảng/task khác nhau.

## 5. Những câu hỏi cần xác nhận với BTC/team

1. Đội đang thi LegalIR, LegalQA hay cả hai?
2. Competition ID chính thức của UIT Data Science Challenge 2026 là gì?
3. `selected-contexts.zip` được tải từ Drive hay từ Codabench phase nào?
4. Với `warmup.json`, output cần là `submission.json` hay `submit.csv`?
5. Có tài liệu riêng cho LegalQA/hallucination classification không?

## 6. Không nên làm lúc này

- Không xây BM25 dựa trên `warmup.json` một mình vì file không chứa corpus passage.
- Không dùng `submit.csv` hallucination format cho LegalIR.
- Không coi mốc thời gian 2025 trên trang Codabench là lịch thi 2026 nếu chưa có xác nhận.

## 7. Kế hoạch cá nhân cho Public Test

Khi đã lấy được đúng corpus LegalIR:

```bash
python3 src/legalir_baseline.py \
  data/public/questions.json \
  data/public/contexts \
  submissions/legalir_bm25_k5.json \
  --top-k 5

python3 src/validate_legalir_submission.py \
  data/public/questions.json \
  submissions/legalir_bm25_k5.json
```

Đo trên warmup/train có nhãn:

```bash
python3 src/evaluate_legalir.py \
  data/raw/warmup.json \
  submissions/legalir_bm25_k5.json
```

Evaluator local báo macro Recall, macro Precision và hit@k. Đây là công cụ chẩn đoán để so sánh các phiên bản; nếu scorer chính thức có cách aggregate khác thì điểm Codabench là chuẩn cuối cùng.
