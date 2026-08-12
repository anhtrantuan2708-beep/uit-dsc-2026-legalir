# UIT Data Science Challenge — Beginner Overview

## Nếu chỉ nhớ một câu

Cuộc thi yêu cầu máy tính đọc câu hỏi pháp luật tiếng Việt, nhìn vào dữ liệu được cung cấp, rồi xuất ra một dự đoán đúng format.

Bạn không cần bắt đầu bằng AI phức tạp. Warmup chỉ cần giúp bạn hiểu:

```text
input → xử lý → dự đoán → file nộp → hệ thống chấm điểm
```

## Hai bài toán đang xuất hiện trong các nguồn

### A. LegalIR — tìm văn bản liên quan

Ví dụ:

```text
Câu hỏi: Điều kiện để hành nghề quản tài viên là gì?
Dự đoán: ["277391"]
```

Máy không cần viết câu trả lời. Máy cần tìm đúng ID văn bản.

### B. Hallucination classification — phân loại câu trả lời

Trang Codabench hiện có mô tả Warmup dạng:

```text
id,predict_label
123,no
124,intrinsic
125,extrinsic
```

Ý nghĩa:

- `no`: câu trả lời được hỗ trợ bởi thông tin đúng.
- `intrinsic`: câu trả lời mâu thuẫn với context.
- `extrinsic`: câu trả lời đưa thêm thông tin không có trong context.

Hai dạng này không được trộn input/output với nhau. Project giữ cả hai phần để học, còn task chính thức sẽ xác định bằng link/notice của BTC.

## Warmup dùng để làm gì?

Warmup không phải lúc để đạt điểm cao. Mục tiêu là:

1. Đọc được file input.
2. Hiểu từng cột/field.
3. Tạo output đúng format.
4. Chạy validator trước khi nộp.
5. Biết hệ thống chấm điểm hoạt động thế nào.

## Thứ tự học

1. JSON/CSV và Python cơ bản.
2. Input/output của competition.
3. Train/validation/test.
4. Recall, Precision, F1.
5. Baseline đơn giản.
6. Sau đó mới học BM25, embeddings, reranker hoặc LLM.
