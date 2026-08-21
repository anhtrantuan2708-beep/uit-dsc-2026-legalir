# UIT DSC 2026 LegalIR — research plan sau V12

Ngày chốt plan: 21/08/2026

Mục tiêu cuối: một submission Public hợp lệ có Recall lớn hơn `0.9591`.
Baseline phải giữ nguyên: `submissions/legalir_public_v10.zip` — Public Recall `0.85467`, Precision `0.1828`.

## 1. Ta đang đứng ở đâu

- V10 local full-dev: Recall khoảng `0.8562`, Precision `0.1821`.
- V12 local full-dev, 1.003 query: Recall `0.8671`, Precision `0.1850`.
- V12 giữ bốn ID đầu của V10 và chỉ thay ID thứ năm bằng chunk-aware BGE khi confidence từ `0.93` trở lên.
- FTS5 hiện có 463.253 chunk, nhưng dùng splitter tương đối phẳng. Candidate Recall@100 từng đạt `0.9467` trên smoke A; con số này chưa đủ để suy ra có thể vượt Public `0.9591`, và các smoke block khác yếu hơn.
- V11 learned fusion, V13 rerank rộng, V14 weak-label chunk fine-tune và question-KNN BGE đã bị reject. Không mở lại bốn nhánh đó trong sprint này.

Ba tầng đánh giá cần phân biệt:

1. **Smoke:** 100 query/block, dùng A, B và C để loại ý tưởng rẻ.
2. **Full dev:** toàn bộ 1.003 query có nhãn ở local. Đây vẫn không phải Public.
3. **Public:** Codabench chấm bằng nhãn ẩn. Chỉ tầng này mới xác nhận thắng leaderboard.

Baseline top-5 cố định cho smoke:

| Block | V10 Recall | V12 Recall | V10 Precision | V12 Precision |
|---|---:|---:|---:|---:|
| A | 0.8692 | 0.8817 | 0.1860 | 0.1900 |
| B | 0.8750 | 0.8800 | 0.1800 | 0.1800 |
| C | 0.8000 | 0.8450 | 0.1640 | 0.1740 |

## 2. Đúng ba hướng kỹ thuật nên làm

### Hướng 1 — Hierarchy-first child → parent retrieval

**Giả thuyết**

Một document dài đang làm các Điều/Khoản/Điểm quan trọng bị pha loãng. Ta cần index đơn vị nhỏ để tìm, nhưng đưa đủ ngữ cảnh cấp cha cho model chọn đúng document.

Pipeline đề xuất:

```text
Văn bản
  → Chương/Mục
    → Điều
      → Khoản
        → Điểm (child dùng để retrieve, khoảng 450 ký tự)
             ↓
      parent evidence 1.500–2.000 ký tự dùng để rerank
             ↓
      aggregate chunk score → source document ID
```

Mỗi child phải có prefix lấy **chỉ từ corpus của BTC**:

```text
[Loại + số văn bản] | [Tiêu đề] | [Chương/Mục] | [Điều]
[Nội dung Khoản/Điểm]
```

Không crawl web và chưa sinh title bằng LLM. `name`, số văn bản trong passage, Điều/Khoản/Điểm là nguồn dữ liệu đủ an toàn cho smoke đầu.

**Vì sao hợp với luật Việt Nam**

- Câu hỏi thường khớp một Khoản/Điểm ngắn, còn ý nghĩa pháp lý phụ thuộc Điều và văn bản cha.
- ViDRILL dùng chunk tối đa 450 ký tự cho retrieval và chunk tối đa 2.000 ký tự cho rerank; đồng thời chỉ chọn chunk đại diện thay vì gán mọi chunk của document là positive. [ViDRILL, VLSP 2025](https://aclanthology.org/2025.vlsp-1.17.pdf)
- Một hệ thống DRiLL khác tăng Recall từ `0.6471` lên `0.7564` khi thêm title hierarchy, nhưng kết quả thuộc dataset khác nên chỉ là evidence chọn hướng, không phải dự báo điểm UIT. [DRiLL title enrichment](https://aclanthology.org/2025.vlsp-1.20.pdf)
- Passage aggregation có cơ sở tốt hơn việc luôn lấy một passage duy nhất khi evidence phân tán. [PARADE](https://arxiv.org/abs/2008.09093)

**Rủi ro**

- Regex parse nhầm do format văn bản không đồng nhất.
- Child quá ngắn mất điều kiện/ngoại lệ; prefix quá dài lấn át nội dung.
- Một document có nhiều chunk giống nhau làm top-100 bị chiếm chỗ.
- Top2 aggregation có thể ưu tiên document dài nếu không chuẩn hóa.

**Smoke rẻ nhất**

1. Chỉ build structured FTS5 mới; chưa chạy BGE.
2. Chạy A, B, C với cùng `top_k=100`.
3. So sánh ba cách chunk→document: `MaxP`, `Top2Sum-normalized`, `logsumexp-normalized`.
4. Với phương án tốt nhất, rerank top-20 bằng `path + top1 child`; chỉ khi pass mới thử `path + top2 child`.

**Scale khi**

- Parser nhận diện hierarchy ở ít nhất 85% document có Điều/Khoản; và
- pooled candidate Recall@100 A+B+C tăng ít nhất `+0.015` so với FTS hiện tại, không block nào giảm hơn `0.005`; và
- final Recall@5 tăng ít nhất `+0.010` so với V12 trên ít nhất 2/3 block.

**Dừng khi**

- Candidate Recall@100 không tăng; hoặc
- gain chỉ có ở A nhưng B/C giảm; hoặc
- top2 evidence không hơn top1 nhưng runtime tăng rõ rệt.

### Hướng 2 — BGE-M3 hybrid: dense + learned sparse, multi-vector chỉ cho shortlist

**Giả thuyết**

FTS/BM25 giữ được số hiệu và từ pháp lý chính xác; dense bắt được diễn đạt đời thường ↔ thuật ngữ luật; multi-vector MaxSim tìm đúng cụm token trong child. Ba tín hiệu này bổ sung nhau tốt hơn việc chỉ rerank candidate FTS hiện tại.

**Vì sao hợp với dữ liệu**

- BGE-M3 hỗ trợ hơn 100 ngôn ngữ, dense, sparse và multi-vector trong cùng model, với input đến 8.192 token. [BGE-M3, Findings ACL 2024](https://aclanthology.org/2024.findings-acl.137/)
- Benchmark IR tiếng Việt 2026 cho thấy lexical–semantic hybrid ổn định qua nhiều domain và đáng thử riêng trên legal; metric/dataset của paper không giống UIT nên không được chuyển thẳng thành dự báo Public. [Vietnamese IR study, EACL 2026](https://aclanthology.org/2026.findings-eacl.110/)
- ViDRILL cũng dùng union BM25 và nhiều dense retriever trước BGE rerank trong retrieval tiếng Việt pháp luật. [ViDRILL](https://aclanthology.org/2025.vlsp-1.17.pdf)

**Thiết kế tiết kiệm compute**

1. Stage 2A: encode đúng 8.532 `representative documents`, không encode 463.253 chunk ngay.
2. Lấy union `structured FTS top100 ∪ BGE dense top100 ∪ BGE sparse top100`, deduplicate theo source ID.
3. Dùng fixed RRF, không learned fusion: thử đúng hai cấu hình `1:1:1` và `2:1:1` (FTS:dense:sparse).
4. Multi-vector chỉ tính cho top-20 child của shortlist; tuyệt đối chưa build full multi-vector index.
5. Chỉ khi Stage 2A pass mới encode structured child theo shard có cache/resume.

**Rủi ro**

- Model lớn, MPS/CPU chậm; full multi-vector index tốn RAM/disk theo số token.
- Dense và sparse score khác thang; weighted sum dễ overfit.
- Multilingual không có nghĩa mặc định tối ưu cho corpus UIT.
- Full document embedding có thể bỏ sót chi tiết, vì vậy nó chỉ là smoke candidate-diversity trước khi scale child.

**Smoke rẻ nhất**

- Dùng representative corpus đã có hoặc build lại ở budget 4.000–6.000 ký tự.
- Chạy A và B trước; C chỉ chạy khi A+B đều không âm.
- Báo riêng `FTS`, `dense`, `sparse`, `union oracle@100`, `RRF@100`, rồi mới đo final top-5.
- Không train, không grid hơn hai weight nêu trên.

**Scale khi**

- Candidate Recall@100 tăng ít nhất `+0.010` trên cả A và B; và
- union có document mới đúng, không chỉ reorder ID cũ; và
- final Recall@5 tăng ít nhất `+0.010` trên cả A và B sau multi-vector/rerank shortlist.

Trước khi kỳ vọng vượt top 1, full-dev candidate Recall@100 phải vượt khoảng `0.97`; đây chỉ là **trần candidate**, chưa phải score top-5 và càng chưa phải Public score.

**Dừng khi**

- Một block giảm; hoặc
- union oracle tăng nhưng reranking không chuyển gain xuống top-5; hoặc
- full multi-vector là điều kiện bắt buộc mới thấy gain trên mini-index.

### Hướng 3 — Legal reference graph one-hop, rule-based trước

**Giả thuyết**

82/1.003 query dev có nhiều hơn một gold ID. V12 đạt khoảng `0.6189` Recall trên nhóm này, thấp hơn nhiều so với single-gold. Các văn bản luật dẫn chiếu, sửa đổi, hướng dẫn và căn cứ lẫn nhau; một hop từ document seed có thể tìm thêm ID đúng mà lexical/dense bỏ sót.

Ta chưa dùng GNN. Bước đầu chỉ build graph từ corpus BTC:

- node: source document ID;
- edge: `REFERS_TO`, `AMENDS`, `GUIDES`, `PURSUANT_TO`;
- resolver: số văn bản chuẩn hóa → `source_id` trong `legalir_metadata.json`;
- retrieval: mở rộng đúng một hop từ top-20 seed, tối đa 10 neighbor.

**Vì sao hợp với dữ liệu**

- Nghiên cứu statute retrieval cho thấy cấu trúc và dependency pháp luật có thể cải thiện dense retrieval. [G-DSR, EACL 2023](https://aclanthology.org/2023.eacl-main.203/)
- Reference/relation extraction trên 5.031 văn bản luật Việt Nam với 61.446 reference đạt F1 cao trong paper; quan trọng hơn cho sprint này, paper mô tả rõ dictionary + regex theo loại văn bản, số hiệu và relation phrase để dựng baseline rẻ. [Vietnamese legal reference extraction](https://cit.iict.bas.bg/CIT-2023/v-23-2/10341-Volume23_Issue_2-05_paper.pdf)

**Rủi ro**

- “Được dẫn chiếu” không đồng nghĩa “trả lời query”.
- Cùng số hoặc phiên bản sửa đổi có thể resolve nhầm ID.
- Graph expansion dễ làm nhiễu single-gold và chiếm một trong năm slot.
- Corpus có metadata lỗi; cần report coverage và collision.

**Smoke rẻ nhất**

1. Build graph bằng regex, chưa rerank.
2. Oracle test: với gold đang thiếu khỏi top-20 seed, kiểm tra nó có nằm trong one-hop neighbor không.
3. Chỉ khi oracle pass mới score neighbor theo seed rank + edge prior.
4. Rerank neighbor bằng chunk-aware BGE đã cache; final router được phép thay tối đa một slot.

**Scale khi**

- Oracle one-hop tăng overall candidate recall ít nhất `+0.020` **hoặc** tăng multi-gold recall ít nhất `+0.050`; và
- mapping số văn bản → source ID có coverage ít nhất 70%, collision được report; và
- sau rerank, multi-gold Recall@5 tăng `+0.030`, overall tăng ít nhất `+0.005`, single-gold không giảm quá `0.003`.

**Dừng khi**

- Oracle không đạt gate; hoặc
- phần lớn neighbor chỉ là văn bản căn cứ chung, không phải gold; hoặc
- muốn có gain phải thay hơn một slot V12.

## 3. Quy tắc đánh giá chung

Mỗi experiment phải ghi đúng sáu số:

```text
candidate Recall@20
candidate Recall@50
candidate Recall@100
final Recall@5
final Precision@5
runtime + peak disk/RAM (ước lượng cũng được, phải ghi cách đo)
```

Không scale chỉ vì một block tăng. Gate tối thiểu:

```text
Smoke A pass
AND Smoke B pass
AND Smoke C không regression đáng kể
→ mới chạy full dev 1.003 query
→ validate <= 5 ID/query
→ mới tạo Public ZIP
```

Full-dev candidate gate: Recall@100 `>= 0.97`.

Full-dev final gate để đáng nộp Public: Recall@5 `>= 0.8771` (V12 + 0.010) và document-disjoint không giảm.
Thắng cuộc chỉ được ghi nhận khi Codabench Public thật `> 0.9591`.

## 4. Năm ticket tách việc

Các ticket độc lập về **file sở hữu, command và output** để năm người không sửa đè nhau. Dependency dữ liệu được khai báo rõ: T02 đọc output T01; T05 chỉ thu kết quả cuối; T03 và T04 có thể chạy song song ngay từ ngày 1.

### T01 — Build hierarchy corpus

**Mục tiêu dễ hiểu:** biến 8.532 document dài thành child ngắn có đầy đủ đường dẫn pháp luật và parent evidence.

**File sở hữu:** `src/build_hierarchical_legalir_corpus.py`.

**Input:** `data/real/contexts/*.json`.

**Output bắt buộc:**

- `data/derived/legalir_hierarchy_children.jsonl`
- `data/derived/legalir_hierarchy_parents.jsonl`
- `reports/experiments/legalir_hierarchy_audit.json`

Audit phải có: tổng document/chunk, % parse được Điều/Khoản/Điểm, 20 ví dụ path, số duplicate, số child vượt limit.

**Command contract:**

```bash
.venv/bin/python src/build_hierarchical_legalir_corpus.py \
  data/real/contexts data/derived \
  --child-chars 450 --parent-chars 2000 \
  --audit reports/experiments/legalir_hierarchy_audit.json
```

**Done:** command chạy lại được, audit coverage đạt 85%, không đổi file dữ liệu gốc.

### T02 — Structured FTS + child→document aggregation

**Mục tiêu dễ hiểu:** dùng child của T01 để tìm top-100 document và so sánh ba cách gộp điểm.

**File sở hữu:** `src/retrieve_hierarchical_fts_legalir.py`.

**Output bắt buộc:** ba prediction top100/block và một report comparison.

**Command contract:**

```bash
.venv/bin/python src/retrieve_hierarchical_fts_legalir.py \
  data/derived/legalir_dev_smoke100.json \
  data/derived/legalir_hierarchy_children.jsonl \
  data/derived/legalir_hierarchy_fts5.sqlite \
  tmp/hier_fts_smoke_a_top100.json \
  --aggregate maxp --top-k 100

.venv/bin/python src/evaluate_legalir.py \
  data/derived/legalir_dev_smoke100.json \
  tmp/hier_fts_smoke_a_top100.json --top-k 100
```

Lặp lại với B/C và `top2sum`, `logsumexp`; không đổi hyperparameter khác.

**Done:** report chọn đúng một aggregation theo gate Hướng 1.

### T03 — BGE-M3 hybrid candidate retriever

**Mục tiêu dễ hiểu:** bổ sung document semantic bị FTS bỏ sót, nhưng không build index multi-vector khổng lồ.

**File sở hữu:** `src/retrieve_bgem3_hybrid_legalir.py`.

**Output bắt buộc:** dense top100, sparse top100, hybrid top100, cache embedding, runtime report cho A/B.

**Command contract:**

```bash
.venv/bin/python src/build_legalir_representative_corpus.py \
  data/real/contexts data/derived/legalir_context_representative_v2.json \
  --budget 6000 --slices 6

.venv/bin/python src/retrieve_bgem3_hybrid_legalir.py \
  data/derived/legalir_dev_smoke100.json \
  data/derived/legalir_context_representative_v2.json \
  tmp/bgem3_smoke_a \
  --model BAAI/bge-m3 --modes dense,sparse \
  --top-k 100 --batch-size 4 --cache-dir models/cache
```

**Done:** báo riêng từng mode và fixed RRF; chỉ viết script scale-by-shard khi A+B pass.

### T04 — Reference graph oracle

**Mục tiêu dễ hiểu:** trả lời câu hỏi “nếu đi một hop từ top-20 hiện tại, ta có chạm được gold ID còn thiếu không?” trước khi tốn BGE.

**Files sở hữu:** `src/build_legalir_reference_graph.py`, `src/evaluate_legalir_graph_oracle.py`.

**Output bắt buộc:**

- `data/derived/legalir_reference_graph.json`
- `reports/experiments/legalir_graph_stats.json`
- `reports/experiments/legalir_graph_oracle.json`

**Command contract:**

```bash
.venv/bin/python src/build_legalir_reference_graph.py \
  data/real/contexts data/derived/legalir_metadata.json \
  data/derived/legalir_reference_graph.json \
  --stats reports/experiments/legalir_graph_stats.json

.venv/bin/python src/evaluate_legalir_graph_oracle.py \
  data/derived/legalir_dev.json \
  tmp/dev_v12_fts_chunkaware_bge_top20.json \
  data/derived/legalir_reference_graph.json \
  --seed-k 20 --hops 1 \
  --output reports/experiments/legalir_graph_oracle.json
```

**Done:** oracle pass/stop rõ ràng theo gate Hướng 3. Nếu fail, ticket kết thúc; không viết GNN.

### T05 — Gate runner, integration và Public artifact

**Mục tiêu dễ hiểu:** mọi branch được chấm cùng cách; chỉ branch pass mới ghép với V12 và tạo ZIP.

**Files sở hữu:** `scripts/run_legalir_research_gates.sh`, `reports/experiments/legalir_7day_decision.md`.

**Output bắt buộc:** bảng A/B/C/full-dev, validator result, tên JSON/ZIP cuối và SHA-256.

**Command contract:**

```bash
bash scripts/run_legalir_research_gates.sh <candidate_top100.json> <candidate_k5.json>

.venv/bin/python src/validate_legalir_submission.py \
  'data/real/LegalIR - Public Test/public-official.json' \
  submissions/submission.json
```

T05 không tự thay V10. Mỗi candidate Public phải là ZIP chỉ chứa `submission.json` và tối đa năm ID/query.

**Done:** report ghi `PASS`, `REJECT` hoặc `PUBLIC-PENDING`; tuyệt đối không đổi local score thành Public claim.

## 5. Thứ tự bảy ngày

| Ngày | Việc chính | Gate cuối ngày |
|---|---|---|
| 1 | T05 đóng băng baseline/gate; T01 dựng parser; chuẩn bị V12 Public script/validator nhưng chưa gọi nó là win | Tất cả baseline tái lập đúng |
| 2 | T01 build hierarchy full corpus; T04 build graph + mapping audit | Hierarchy coverage ≥85%; graph có stats/collision |
| 3 | T02 chạy structured FTS A/B/C; T04 chạy oracle one-hop | Reject ngay branch không qua candidate gate |
| 4 | T03 chạy BGE-M3 representative dense+sparse trên A/B, rồi C nếu pass | Chỉ scale khi A+B cùng tăng |
| 5 | Rerank shortlist của tối đa hai branch sống; không grid rộng | Final Recall@5 tăng ≥0.01 ở ít nhất 2/3 block |
| 6 | Chạy full dev 1.003 query, document-disjoint và validator; integrate tối đa một thay đổi/lần | Full-dev ≥0.8771, candidate@100 ≥0.97, disjoint không giảm |
| 7 | Tạo tối đa hai Public candidate: V12 calibration và winner mới; ghi Codabench receipt thật | Giữ V10 nếu không có Public score tốt hơn |

## 6. Plan để Terra High thực thi

Terra chạy theo đúng thứ tự này, mỗi lần chỉ một command nhỏ:

1. **Không train.** Re-run baseline evaluator và validator; ghi kết quả vào report T05.
2. Implement T01, chỉ chạy `--audit` trên 50 document trước; pass mới build 8.532 document.
3. Implement T02, chạy smoke A; pass mới chạy B, rồi C. Nếu gate fail thì dừng Hướng 1.
4. Chạy T04 oracle song song vì không cần model. Nếu oracle fail thì đóng graph, không code expansion/GNN.
5. Implement T03 chỉ với representative corpus và dense+sparse. Không encode 463.253 child, không multi-vector full index.
6. Chọn tối đa hai branch qua gate để rerank top-20; giữ output/cache để resume.
7. Chạy full dev đúng một lần cho mỗi branch sống; T05 so với V10, V12 và document-disjoint.
8. Chỉ khi full-dev gate pass mới build/validate Public ZIP. Thành công chỉ được ghi khi Codabench thật vượt Recall `0.9591`.
