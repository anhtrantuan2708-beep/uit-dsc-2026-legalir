# UIT Data Science Challenge

Workspace nghiên cứu và chuẩn bị submission cho UIT Data Science Challenge.

> Repo này chỉ chứa source code, tài liệu và report. Dataset, model, prediction và submission ZIP không được đưa lên GitHub.

Xem [CONTRIBUTING.md](CONTRIBUTING.md) để biết cách tạo branch, commit và bàn giao thí nghiệm trong team.

## Visual reports

Các file HTML để đọc trực quan nằm trong thư mục [`visuals/`](visuals/). Tải file `.html` về và mở bằng trình duyệt để xem giao diện.

## Trạng thái hiện tại

Đã xác minh lại competition đúng cho năm 2026:

- **Competition 17716 — LegalQA**: sinh câu trả lời tiếng Việt; metric chính METEOR, phụ ROUGE-L; submission là `submission.zip` chứa duy nhất `submission.json`.
- **LegalIR** là track khác: trả về ID văn bản và dùng Recall/Precision. Không dùng nhầm metric giữa hai track.
- Các ghi chú về link 10153 là lịch sử research của competition cũ, không dùng làm contract cho 17716.

Kế hoạch điều hành team nằm tại [docs/team_execution_plan_legalqa_2026.md](docs/team_execution_plan_legalqa_2026.md).

## Beginner guide

- [00 — Tổng quan cho newbie](docs/00_beginner_overview.md)
- [01 — Glossary](docs/01_glossary_for_newbie.md)
- [02 — LegalIR walkthrough](docs/02_warmup_legalir_walkthrough.md)
- [03 — Hallucination walkthrough](docs/03_warmup_hallucination_walkthrough.md)
- [04 — Research roadmap](docs/04_research_roadmap.md)
- [05 — Daily checklist](docs/05_daily_checklist.md)
- [Cam kết sử dụng dữ liệu](docs/research_data_commitment.md)
- [Research hạn chế sử dụng AI](docs/research_ai_usage.md)
- [Team execution plan 2026](docs/team_execution_plan_legalqa_2026.md)
- [Experiment card template](docs/experiment_template.md)
- [Submission registry](reports/submission_registry.csv)
- [Sân tập LegalIR bằng dữ liệu giả lập](docs/synthetic_legalir_practice.md)
- [Implementation guide LegalIR](docs/06_implementation_guide_legalir.md)
- [Project catch-up và lợi thế cạnh tranh](docs/07_project_catchup_legalir.md)

## Việc tiếp theo

Chưa tối ưu model. Việc cần làm trước tiên là xác nhận team access và lấy đúng question/corpus của LegalQA 17716; không dùng `warmup.json` LegalIR thay thế.

## Chạy ngay

```bash
python3 src/inspect_warmup.py data/raw/warmup.json
python3 src/validate_legalir_submission.py data/raw/warmup.json data/raw/warmup.json
```

Khi có corpus chứa các file `context_*.json`, chạy baseline:

```bash
python3 src/legalir_baseline.py \
  data/raw/warmup.json \
  data/raw/contexts \
  submissions/legalir_baseline.json
```

Baseline hiện dùng lexical/BM25 đơn giản, không cần cài thêm thư viện.
