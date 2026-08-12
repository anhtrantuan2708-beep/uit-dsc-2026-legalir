# UIT Data Science Challenge 2026 — Team Execution Plan

## 1. Quyết định phạm vi

**Competition 17716 là UIT-DSC 2026 Subtask 2: Legal Question Answering (LegalQA).** Đây là bài sinh câu trả lời pháp luật bằng tiếng Việt, không phải bài phân loại hallucination của competition 10153 năm 2025.

Nếu team đăng ký cả hai topic thì phải quản lý thành hai track độc lập:

| Track | Nhiệm vụ | Output | Metric chọn đội |
|---|---|---|---|
| LegalIR | Tìm văn bản phù hợp với câu hỏi | Danh sách ID văn bản | Recall chính, Precision phụ |
| LegalQA / 17716 | Trả lời câu hỏi bằng văn xuôi | Chuỗi `answer` tiếng Việt | METEOR chính, ROUGE-L phụ |

**Kế hoạch trong file này ưu tiên LegalQA 17716.** Không dùng điểm Recall/Precision của LegalIR để kết luận đội thắng ở 17716.

## 2. Kết quả research đã xác minh

- Tên: `UIT-DSC 2026 Subtask 2: Legal Question Answering (LegalQA)`.
- Đơn vị tổ chức hiển thị trên Codabench: UIT-DSC 2026 Organizers; email `dsc@uit.edu.vn`.
- Kho dữ liệu chung được mô tả khoảng 8.500 văn bản hành chính; hai bộ câu hỏi của LegalIR và LegalQA độc lập, không trùng nhau.
- WarmUp: 01/08/2026 00:00 đến 05/08/2026 23:59 (GMT+7).
- Public Test: 06/08/2026 00:00 đến 18/09/2026 23:59 (GMT+7).
- Private Test: 19/09/2026 00:00 đến 23/09/2026 23:59 (GMT+7).
- Submission: một file ZIP tên `submission.zip`, bên trong **duy nhất** `submission.json`.
- `submission.json` là JSON object. Mỗi key là `question_id`; value là object có đúng một trường `answer`, kiểu string.
- Mỗi câu hỏi phải xuất hiện đúng một lần, không được thiếu câu hỏi, file UTF-8.
- Ví dụ:

```json
{
  "147194": {"answer": "Theo quy định tại Điều 37 Luật Doanh nghiệp..."},
  "147195": {"answer": "Người lao động có quyền..."}
}
```

- METEOR là metric chính để xếp hạng; ROUGE-L là metric phụ; cả hai càng cao càng tốt.
- Trang Codabench hiện hiển thị 101 participants và 209 submissions; đây chỉ là trạng thái tại thời điểm research, không phải kết quả cuối.
- Chưa thấy trên trang 17716 một quy định công khai về số submission tối đa mỗi ngày. PM phải kiểm tra lại trước mỗi đợt submit và hỏi BTC nếu cần.

Nguồn: [Codabench 17716](https://www.codabench.org/competitions/17716/#/pages-tab), [thông tin UIT-DSC 2026 của CITD](https://www.citd.edu.vn/cuoc-thi-uit-data-science-challenge-2026/).

## 3. Mục tiêu của team

1. Có một baseline chạy lại được từ đầu đến cuối.
2. Có local validation gần với public test, không tối ưu mù theo leaderboard.
3. Tạo câu trả lời đúng nội dung pháp luật, có Điều/Khoản/Điểm khi cần, không tự bịa.
4. Mỗi submission có mã experiment, commit, config và log để biết vì sao điểm tăng/giảm.
5. Trước Private Test có một bản đã freeze, đóng gói và kiểm tra format nhiều lần.

## 4. Phân công team

### Nếu có 5 người

| Vai trò | Người phụ trách | Deliverable bắt buộc |
|---|---|---|
| PM / Rules & Submission | Team lead | Rulebook, lịch, registry submission, quyết định candidate cuối |
| Data & Evaluation | Member A | Loader, data dictionary, split local, evaluator METEOR/ROUGE-L |
| Retrieval | Member B | BM25 baseline, dense retrieval, hybrid/RRF và retrieval report |
| QA / Grounding | Member C | Prompt, citation format, answer verifier, error taxonomy |
| MLOps / Packaging | Member D | Script one-command, schema validator, ZIP submission, reproducibility |

### Nếu có 3 người

- Người 1: PM + Data/Evaluation.
- Người 2: Retrieval + context construction.
- Người 3: QA/Grounding + MLOps/Packaging.

### Quy tắc làm việc

- Mỗi người sở hữu một **experiment family**, không sửa lẫn pipeline của người khác mà không ghi log.
- Không submit chỉ vì “thử cho biết”. Phải có local score, hypothesis và expected risk.
- Một candidate chỉ được submit sau khi PM duyệt và đã được validator kiểm tra.
- Public leaderboard là tín hiệu tham khảo, không được dùng để overfit; private test mới là kết quả quyết định.

## 5. Kiến trúc baseline đề xuất

```text
question
  -> normalize Vietnamese text
  -> retrieve relevant legal passages
  -> build compact context with document/article metadata
  -> generate answer in Vietnamese
  -> citation / entailment / unsupported-claim checks
  -> answer string
  -> validate submission.json
  -> zip submission.zip
```

Thứ tự triển khai:

1. **Baseline A — extractive:** lấy passage liên quan và trả lời bằng đoạn có sẵn; mục tiêu là có điểm hợp lệ.
2. **Baseline B — RAG:** BM25 hoặc dense retrieval + prompt trả lời chỉ từ context.
3. **Baseline C — hybrid:** BM25 + dense bằng RRF; giới hạn context và loại trùng.
4. **Baseline D — grounded QA:** bắt buộc trích dẫn Điều/Khoản/Điểm; claim không map được vào context thì bỏ hoặc trả lời thận trọng.
5. **Tối ưu cuối:** synonym pháp lý, query expansion, reranking, prompt và độ dài câu trả lời.

## 6. Timeline từ hiện tại đến Private Test

| Thời gian | Mục tiêu | Gate để chuyển tuần |
|---|---|---|
| 08–10/08 | Chốt đúng track, quyền truy cập, schema, nguồn dữ liệu; tạo registry | Có sample `submission.zip` hợp lệ |
| 11–17/08 | Chạy baseline extractive và BM25/RAG; dựng local evaluator | Baseline chạy một lệnh, có score và log |
| 18–24/08 | Mỗi member làm một experiment family; so sánh retrieval, prompt, model | Có bảng ablation, không chỉ có một điểm leaderboard |
| 25–31/08 | Error analysis 50–100 câu; phân loại thiếu context, sai luật, dài dòng, format | Có top lỗi và backlog sửa |
| 01–07/09 | Hybrid/RRF, reranking, citation/grounding, answer-length tuning | Candidate vượt baseline local và không tăng lỗi nghiêm trọng |
| 08–14/09 | Public-test candidate selection; chỉ submit các bản đã duyệt | Chọn tối đa 1–2 candidate chính, có rollback |
| 15–18/09 | Freeze public phase; kiểm tra format, backup, ghi nhận điểm cuối | Không thay đổi code không có phê duyệt |
| 19–23/09 | Private Test: nộp bản frozen theo hướng dẫn BTC | Có log, checksum, người chịu trách nhiệm và biên nhận |
| Sau 23/09 | Không xóa dữ liệu sớm; chờ hướng dẫn kết quả/báo cáo | Tuân thủ cam kết dữ liệu đến hết thời hạn |

Lịch chính thức trên trang giới thiệu cuộc thi ghi kết quả ngày 13/11/2026; PM cần cập nhật nếu BTC thay đổi.

## 7. Quy trình mỗi experiment

Mỗi thử nghiệm phải có một record gồm:

- `experiment_id`: ví dụ `qa-rag-003`.
- Owner và ngày chạy.
- Hypothesis: thay đổi gì và vì sao sẽ tốt hơn.
- Data split / seed / model / prompt / retrieval top-k.
- Local METEOR, local ROUGE-L, số câu lỗi và thời gian chạy.
- Commit hash và đường dẫn artifact.
- Quyết định: `keep`, `reject`, hoặc `needs_public_check`.

Chỉ submit public khi:

1. `submission.json` có đủ toàn bộ question ID.
2. Mỗi answer là string UTF-8, không có list/object lồng sai schema.
3. Đã lưu bản ZIP bất biến và checksum.
4. PM ghi submission vào registry trước khi nộp.
5. Sau khi có điểm, ghi cả public METEOR và ROUGE-L; không ghi mỗi “điểm cao hơn”.

## 8. Checklist quản lý

### A. Competition / quyền truy cập

- [ ] Tất cả thành viên vào đúng competition 17716.
- [ ] Xác nhận team, leader, email sinh viên và deadline.
- [ ] Lưu screenshot/link của Rules, Submission, Evaluation, Phases.
- [ ] Xác nhận giới hạn submission với BTC hoặc forum; không tự đoán.

### B. Data

- [ ] Có corpus/context và question file đúng của LegalQA.
- [ ] Không dùng `warmup.json` của LegalIR để đánh giá LegalQA.
- [ ] Không trộn public/private test vào local training.
- [ ] Có data dictionary, thống kê missing/duplicate/Unicode.
- [ ] Lưu dữ liệu theo đúng cam kết sử dụng; không upload sang API/cloud bên thứ ba nếu chưa được BTC cho phép.

### C. Model / evaluation

- [ ] Baseline chạy được trên máy sạch.
- [ ] Local evaluator tính METEOR và ROUGE-L.
- [ ] Error analysis có ví dụ câu hỏi, context, answer dự đoán, answer tham chiếu.
- [ ] Kiểm tra citation và unsupported claim.
- [ ] Có seed/config để tái lập.

### D. Submission

- [ ] Tên ngoài: `submission.zip`.
- [ ] Bên trong chỉ có `submission.json`.
- [ ] JSON object, key là question ID.
- [ ] Value có đúng `answer` là string.
- [ ] Không thiếu, không lặp question ID.
- [ ] UTF-8 và parse được bằng validator.
- [ ] Lưu ZIP, checksum, commit và người submit.

### E. Final freeze

- [ ] Chọn candidate dựa trên local + public + error analysis.
- [ ] Không chọn chỉ dựa trên một lần leaderboard.
- [ ] Backup code/config/model metadata.
- [ ] Ghi rõ phần chưa chắc chắn và kế hoạch rollback.
- [ ] Xác nhận nghĩa vụ xóa dữ liệu theo cam kết sau thời hạn áp dụng.

## 9. Việc team phải làm ngay trong 24 giờ tới

1. PM tạo nhóm và điền tên người vào bảng vai trò.
2. Data lead xác nhận đã có **LegalQA question + corpus**; nếu chưa có thì gửi email `dsc@uit.edu.vn`, chưa xây pipeline trên dữ liệu đoán.
3. MLOps tạo validator cho format 17716 và sinh sample ZIP.
4. Evaluation lead dựng local split và ghi rõ reference answer đang dùng từ file nào.
5. Retrieval/QA lead chạy baseline đầu tiên, không tối ưu prompt trước khi có baseline.
6. PM mở `reports/submission_registry.csv`, mọi public submission phải đi qua file này.

## 10. Điều chưa thể kết luận từ trang công khai

- Số submission tối đa mỗi ngày chưa hiển thị rõ trên trang 17716.
- Chi tiết file download/corpus LegalQA cần quyền truy cập đúng của team; không dùng file warmup LegalIR thay thế.
- Quy định AI/API bên ngoài phải đối chiếu thêm Terms và cam kết dữ liệu; nguyên tắc an toàn hiện tại là không đưa task data lên dịch vụ bên thứ ba khi chưa có phép BTC.

