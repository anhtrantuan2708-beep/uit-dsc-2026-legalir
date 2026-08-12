# Sân tập LegalIR khi chưa có corpus thật

Có thể dùng bộ dữ liệu giả lập trong `data/sample_legalir/` để kiểm tra toàn bộ pipeline:

```bash
python3 src/legalir_baseline.py \
  data/sample_legalir/queries.json \
  data/sample_legalir/contexts \
  submissions/sample_legalir.json \
  --top-k 5

python3 src/validate_legalir_submission.py \
  data/sample_legalir/queries.json \
  submissions/sample_legalir.json

python3 src/evaluate_legalir.py \
  data/sample_legalir/queries.json \
  submissions/sample_legalir.json
```

Expected output trên bộ mẫu này thường có Recall/Precision cao vì mỗi câu hỏi được viết để khớp với một passage. Đây chỉ là kiểm tra code; điểm này không dự đoán điểm public.

Khi BTC cấp corpus thật, chỉ cần thay hai đường dẫn đầu tiên bằng file question và thư mục `context_*.json` thật. Không trộn dữ liệu giả lập vào submission.
