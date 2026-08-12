# Research - Hạn chế sử dụng AI trong UIT DSC 2026

## Kết luận ngắn

Chưa tìm thấy quy định chính thức nào cấm sử dụng AI/LLM. Ngược lại, thông báo chính thức của CITD mô tả Chủ đề 1 là xây dựng phương pháp truy vấn pháp luật “dựa trên các mô hình ngôn ngữ lớn (LLMs)”, và Chủ đề 2 yêu cầu hệ thống hiểu, truy xuất và sinh câu trả lời pháp luật. Điều này cho thấy việc dùng AI/LLM là hướng được cuộc thi dự kiến.

Tuy nhiên, **được dùng AI không đồng nghĩa với được gửi Task Data cho mọi dịch vụ AI**. Cam kết dữ liệu cấm tiết lộ, chia sẻ hoặc chuyển giao Data Collection cho bên thứ ba.

## Điều đã xác nhận

### 1. AI/LLM là một phần của hướng bài toán

Thông báo chính thức của CITD mô tả:

- Chủ đề 1: truy vấn/tìm kiếm/truy xuất văn bản pháp luật dựa trên LLMs.
- Chủ đề 2: hiểu, truy xuất và sinh câu trả lời pháp luật; yêu cầu tính minh bạch và trích dẫn căn cứ.

Nguồn: [CITD - UIT Data Science Challenge 2026](https://www.citd.edu.vn/cuoc-thi-uit-data-science-challenge-2026/)

### 2. Cam kết dữ liệu không nói “cấm AI”

Cam kết hiện có quy định:

- Task Data thuộc quyền sở hữu trí tuệ của BTC.
- Data Collection chỉ dùng cho cuộc thi và nghiên cứu liên quan.
- Không bán, cho mượn, tiết lộ, chia sẻ hoặc chuyển giao Data Collection cho bên thứ ba.
- Sau 31/12/2026 phải xóa toàn bộ bản sao Data Collection.

Nguồn bản gốc: [docs/sources/cam-ket-su-dung-du-lieu-uit-dsc-2026.pdf](sources/cam-ket-su-dung-du-lieu-uit-dsc-2026.pdf)

## Vậy dùng AI thế nào cho an toàn?

### Mức rủi ro thấp

- Dùng ChatGPT/Codex để học thuật ngữ, viết code không chứa dữ liệu cuộc thi.
- Dùng model pretrained để xây baseline, nếu thể lệ không có hạn chế riêng.
- Chạy model local trên máy với Task Data.
- Dùng BM25/embedding local để lập index.

### Cần hỏi BTC trước

- Gửi `question`, `passage`, `train.json`, `warmup.json` hoặc test data vào API ChatGPT/Claude/Gemini.
- Dùng Google Colab hoặc cloud notebook có upload corpus.
- Dùng API embedding/reranking bên ngoài cho dữ liệu BTC.
- Upload vector index, embedding dump, log chứa passage hoặc raw submission lên dịch vụ public/private của bên thứ ba.
- Dùng dữ liệu ngoài hoặc model đã fine-tune trên dữ liệu pháp luật khác nếu thể lệ chi tiết chưa nói rõ.

Lý do: các hành động này có thể bị xem là chia sẻ/chuyển giao Data Collection cho bên thứ ba, dù mục đích chỉ là gọi model.

## Khuyến nghị pipeline ban đầu

```text
Task Data
  -> xử lý và index local
  -> BM25/embedding local
  -> model local hoặc API chỉ nhận context đã được BTC cho phép
  -> output submission
```

Không nên gửi toàn bộ corpus lên một API chỉ để thử nghiệm khi chưa có xác nhận bằng văn bản.

## Chưa thể kết luận từ các nguồn hiện có

Các nguồn đã đọc chưa nêu rõ:

- Có được dùng API LLM thương mại hay không.
- Có được dùng dữ liệu ngoài hay không.
- Có bắt buộc khai báo model/API hay không.
- Có yêu cầu nộp source code/model để kiểm tra tính tái lập hay không.
- Có cấm dùng model đóng hoặc dịch vụ không tái lập hay không.

Đây là các điểm cần hỏi BTC, không nên tự suy đoán.

## Mẫu câu hỏi gửi BTC

> BTC cho em hỏi đội thi có được sử dụng API LLM/embedding thương mại (ví dụ OpenAI, Gemini, Claude) với dữ liệu Task Data không? Việc gửi question/passage lên API bên thứ ba có được xem là phù hợp với cam kết không? Ngoài ra, BTC có hạn chế về dữ liệu ngoài, model pretrained/fine-tuned và yêu cầu khai báo model khi nộp bài không?
