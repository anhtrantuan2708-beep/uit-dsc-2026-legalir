# Warmup walkthrough — LegalIR

## Dữ liệu hiện có

`data/raw/warmup.json` có 500 mẫu. Mỗi mẫu có dạng:

```json
{
  "147194": {
    "question": "Kiểm ngư viên trung cấp là gì? ...",
    "answer": ["14681"]
  }
}
```

Trong đó:

- `147194` là ID của câu hỏi.
- `question` là input.
- `answer` là đáp án đúng để học format.

## Tư duy bài toán

```text
question
  ↓
đọc các passage pháp luật
  ↓
tìm passage liên quan nhất
  ↓
lấy document_id
  ↓
đưa vào answer
```

## Output minh họa

```json
{
  "147194": {
    "answer": ["14681"]
  }
}
```

Nếu trả về nhiều ID, tối đa 5 ID/query theo tài liệu LegalIR overview.

## Vì sao chưa làm model từ file này?

`warmup.json` chỉ có câu hỏi và ID đáp án. Nó không có `passage` để tìm kiếm. Muốn làm retrieval thật cần thêm corpus/context.

Vì vậy warmup hiện phù hợp để học:

- đọc JSON;
- hiểu query ID;
- hiểu output;
- viết validator;
- kiểm tra duplicate/missing IDs.
