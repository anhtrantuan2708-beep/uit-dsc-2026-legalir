# Research roadmap

## Phase 0 — Xác định đúng task

- [ ] Xác định competition ID chính thức.
- [ ] Xác định đội đang thi LegalIR, LegalQA hay cả hai.
- [ ] Xác định input file.
- [ ] Xác định output file.
- [ ] Xác định metric.

## Phase 1 — Warmup

- [ ] Mở file bằng `jq` hoặc Python.
- [ ] Đếm số mẫu.
- [ ] In 3 mẫu đầu.
- [ ] Kiểm tra field bắt buộc.
- [ ] Tạo output mẫu.
- [ ] Tạo validator.

## Phase 2 — Baseline

### Nếu là LegalIR

```text
keyword matching → BM25 → top-k document IDs
```

### Nếu là hallucination classification

```text
question + context + answer → classifier → one of 3 labels
```

## Phase 3 — Evaluation

- [ ] Tính metric local.
- [ ] Tạo error analysis.
- [ ] Nhóm lỗi theo nguyên nhân.
- [ ] Ghi lại từng thử nghiệm.

## Phase 4 — Optimization

Chỉ làm sau khi baseline chạy được:

- xử lý Unicode/Vietnamese;
- BM25 tuning;
- embeddings;
- hybrid retrieval;
- reranking;
- threshold calibration cho Macro-F1.

## Từ khóa research nên dùng

1. `information retrieval for beginners`
2. `BM25 explained`
3. `precision recall F1 explained`
4. `Vietnamese text preprocessing`
5. `hallucination detection intrinsic extrinsic`
6. `macro F1 classification`
