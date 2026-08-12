# Warmup walkthrough — Hallucination classification

Trang Codabench competition 10153 hiện mô tả một warmup khác với LegalIR.

## Ba nhãn

### `no`

Câu trả lời đúng và có căn cứ trong context.

### `intrinsic`

Câu trả lời tự mâu thuẫn với context.

Ví dụ context ghi “10 ngày”, nhưng answer ghi “20 ngày”.

### `extrinsic`

Câu trả lời thêm thông tin không có trong context.

Ví dụ context chỉ nói điều kiện A, nhưng answer tự thêm mức phạt B mà context không đề cập.

## Output theo trang Codabench hiện tại

```csv
id,predict_label
1001,no
1002,intrinsic
1003,extrinsic
```

Đây là bài classification, không phải LegalIR retrieval. Không dùng `submission.json` của LegalIR cho bài này.

## Cách học bằng warmup

1. Đọc một mẫu.
2. Đọc context và answer.
3. Tự giải thích tại sao là `no`, `intrinsic` hoặc `extrinsic`.
4. Ghi lại các trường hợp dễ nhầm.
5. Chỉ sau đó mới train classifier.
