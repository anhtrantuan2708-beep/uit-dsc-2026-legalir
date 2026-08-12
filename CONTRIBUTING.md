# Cách làm việc chung

## Trước khi bắt đầu

1. Tạo branch riêng, ví dụ `feat/legal-chunking` hoặc `fix/evaluator-duplicates`.
2. Không commit dataset, embeddings, prediction JSON, submission ZIP, model weights, `.env` hoặc token.
3. Chạy trên cùng dev split trước khi báo kết quả.

## Commit gì?

Mỗi commit chỉ nên là một thay đổi có thể mô tả rõ. Ví dụ:

```text
feat(bm25): add legal phrase normalization
feat(dense): benchmark bge-m3 on dev split
feat(rerank): add legal-aware features for top-50 candidates
fix(eval): ignore duplicate document IDs before scoring
docs(experiment): record v7 hypothesis and local metrics
```

Nói đơn giản: commit **code đã sửa**, **test/evaluator mới**, hoặc **report thí nghiệm**. Không commit file prediction và ZIP để nộp.

## Khi bàn giao một hướng thử nghiệm

Gửi kèm trong pull request hoặc tin nhắn team:

- Hypothesis: đang thử cải thiện điều gì?
- File code đã đổi.
- Recall/Precision trước và sau trên dev split.
- Runtime và cách chạy ngắn gọn.
- Kết luận: giữ, loại, hay cần kiểm tra thêm.

## Quy tắc merge

- Không tự merge kết quả chỉ vì leaderboard Public tăng.
- Ít nhất một người khác kiểm tra lại metric local và format prediction.
- Chỉ người nộp submission mới cần giữ ZIP cuối cùng.
