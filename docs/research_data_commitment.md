# Research - Cam kết sử dụng dữ liệu UIT DSC 2026

> Đây là bản tóm tắt để nghiên cứu và quản lý project, không phải tư vấn pháp lý và không thay thế việc đọc/ký bản gốc.

## Nguồn

- Bản gốc trong project: [cam-ket-su-dung-du-lieu-uit-dsc-2026.pdf](sources/cam-ket-su-dung-du-lieu-uit-dsc-2026.pdf)
- File gốc được cung cấp: `Cam kết sử dụng dữ liệu cuộc thi.pdf`
- Tên tài liệu: **Cam kết về việc sử dụng bộ dữ liệu UIT Data Science Challenge 2026**
- Tài liệu dài 3 trang, không có form điện tử và còn để trống ngày/tháng/năm cùng tên người sử dụng/nhóm nghiên cứu.

## Tóm tắt một câu

BTC cho đội quyền sử dụng bộ dữ liệu có giới hạn để tham gia cuộc thi và nghiên cứu liên quan; đội không được chia sẻ lại dữ liệu và phải xóa toàn bộ bản sao sau ngày 31/12/2026.

## Các khái niệm trong cam kết

| Khái niệm | Ý nghĩa trong tài liệu |
|---|---|
| Task Data | Bộ dữ liệu chính thức do BTC phát hành |
| Raw Submission Data | File dự đoán đội nộp |
| Evaluation Data | Dữ liệu sinh ra trong quá trình BTC chấm điểm |
| Data Collection | Gồm Task Data, Raw Submission Data và Evaluation Data |

## Quyền sở hữu

- BTC giữ quyền sở hữu trí tuệ đối với **Task Data**.
- Đội/người sử dụng giữ quyền sở hữu trí tuệ đối với hệ thống hoặc mô hình do đội phát triển.
- BTC giữ quyền sở hữu trí tuệ đối với các phân tích, thống kê hoặc cải tiến sinh ra từ Raw Submission Data và Evaluation Data.

## Đội được phép làm gì?

- Khai thác Data Collection để nghiên cứu và tham gia UIT Data Science Challenge 2026.
- Sử dụng trong các nhiệm vụ thuộc cuộc thi và nghiên cứu liên quan cho đến khi kết thúc vòng Private Test.
- Công bố kết quả nghiên cứu có sử dụng dữ liệu, nhưng phải ghi rõ nguồn UIT Data Science Challenge 2026.

## Đội không được làm gì?

- Không bán, cho mượn, tiết lộ, chia sẻ hoặc chuyển giao Data Collection cho bên thứ ba.
- Không phát hành lại toàn bộ dữ liệu trong bài viết, báo cáo, GitHub, demo public hoặc sản phẩm.
- Không tự ý mở quyền truy cập cho người ngoài đội đã đăng ký.

## Thời hạn

- Cam kết có hiệu lực từ ngày ký đến hết ngày **31/12/2026**.
- Sau thời điểm này, người sử dụng/nhóm nghiên cứu phải xóa toàn bộ bản sao Data Collection khỏi hệ thống lưu trữ.
- Nếu có ấn phẩm khoa học sử dụng dữ liệu, cần gửi báo cáo ngắn gọn cho BTC.

## Khi vi phạm

BTC có quyền yêu cầu đội:

1. Ngừng ngay việc sử dụng dữ liệu.
2. Xóa toàn bộ bản sao Data Collection.

## Việc cần làm trong project này

- [x] Lưu bản PDF gốc trong `docs/sources/`.
- [x] Không đưa Task Data vào Git/public repository.
- [ ] Nếu nhận corpus mới, lưu trong thư mục local/private được bảo vệ.
- [ ] Không upload corpus lên GitHub, public demo, notebook công khai hoặc dịch vụ AI bên ngoài nếu chưa được phép.
- [ ] Ghi lại ngày nhận dữ liệu và ngày phải xóa: **31/12/2026**.
- [ ] Trước khi công bố kết quả, ghi citation nguồn và gửi thông tin công bố cho BTC.
- [ ] Trước khi ký, điền đúng ngày và tên người sử dụng/nhóm nghiên cứu.

## Câu hỏi nên hỏi BTC trước khi dùng dữ liệu

1. Có được dùng Google Colab, API LLM hoặc cloud storage bên thứ ba không?
2. Có được lưu corpus trong private Git repository không?
3. “Nghiên cứu liên quan” có cho phép xây prototype portfolio sau cuộc thi không?
4. Có được chia sẻ một vài passage minh họa trong báo cáo không, và giới hạn bao nhiêu?
5. Sau 31/12/2026 có được giữ lại model/embedding/index đã tạo từ dữ liệu không, hay phải xóa cả các bản dẫn xuất?

## Kết luận thực hành

Trong project hiện tại, hãy coi toàn bộ corpus/context được BTC cấp là **dữ liệu hạn chế**. Có thể viết code và model của riêng mình, nhưng không nên commit hoặc public dữ liệu, index, embedding dump, raw submission hoặc evaluation data khi chưa có hướng dẫn rõ ràng từ BTC.
