import { useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  ArrowUpRight,
  ChartLineUp,
  Check,
  CheckCircle,
  Circle,
  ClipboardText,
  Files,
  Flag,
  Flask,
  FolderOpen,
  House,
  Info,
  MagnifyingGlass,
  Medal,
  Scales,
  Sparkle,
  Target,
  UsersThree,
  X,
} from "@phosphor-icons/react";
import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import "@fontsource/inter/600.css";
import "@fontsource/inter/700.css";
import "./styles.css";

const NAV_ITEMS = [
  { id: "overview", label: "Tổng quan", icon: House },
  { id: "progress", label: "Tiến độ", icon: ChartLineUp },
  { id: "experiments", label: "Thử nghiệm", icon: Flask },
  { id: "team", label: "Gợi ý phân công", icon: UsersThree },
  { id: "sprint", label: "Sprint tiếp theo", icon: Flag },
  { id: "files", label: "Tài liệu chung", icon: Files },
];

const MILESTONES = [
  { label: "BM25", score: "0.3431", date: "Baseline", note: "BM25 thô" },
  { label: "Hybrid", score: "0.5847", date: "V1", note: "BM25 + E5 + RRF" },
  { label: "Kết hợp 3 nguồn", score: "0.7502", date: "Public · V3", note: "Mã V3 · BM25 + E5 + Question-KNN" },
  { label: "Rerank an toàn", score: "0.8020", date: "Public · V5", note: "Mã V5 · giữ 4 kết quả, rerank slot 5" },
  { label: "Rerank top-50", score: "0.8264", date: "Local · V6", note: "Mã V6 · rerank 50 kết quả, giữ 3 base" },
];

const EXPERIMENTS = [
  { id: "E01", pipeline: "BM25 thô", recall: "0.3431", precision: "0.0724", stage: "Local", decision: "replaced", note: "Mốc baseline đầu tiên" },
  { id: "E02", pipeline: "BM25 + title + normalize", recall: "0.4189", precision: "0.0889", stage: "Local", decision: "keep", note: "Chuẩn hoá và bỏ stopwords" },
  { id: "E03", pipeline: "BM25 + query profiles", recall: "0.4234", precision: "0.0899", stage: "Local", decision: "keep", note: "Lexical branch hiện tại" },
  { id: "E04", pipeline: "Dense E5", recall: "0.4688", precision: "0.0985", stage: "Local", decision: "keep", note: "Semantic retrieval" },
  { id: "E05", pipeline: "BM25 + E5 · RRF", recall: "0.5571", precision: "0.1174", stage: "Local", decision: "keep", note: "Hybrid đầu tiên" },
  { id: "V1", pipeline: "Hybrid top 100 + tune RRF", recall: "0.5843", precision: "0.1248", stage: "Public", decision: "replaced", note: "Submission 886134" },
  { id: "E07", pipeline: "Question-KNN", recall: "0.5550", precision: "0.1186", stage: "Local", decision: "keep", note: "Tận dụng câu hỏi train" },
  { id: "V3", pipeline: "BM25 + E5 + Question-KNN", recall: "0.7502", precision: "0.1610", stage: "Public", decision: "replaced", note: "Submission 886603" },
  { id: "E09", pipeline: "4-way + lexical Question-KNN", recall: "0.7624", precision: "0.1611", stage: "Local", decision: "keep", note: "Candidate fusion" },
  { id: "E10", pipeline: "5-branch + doc association", recall: "0.7651", precision: "0.1619", stage: "Local", decision: "keep", note: "Nền cho reranking" },
  { id: "V5", pipeline: "Safe rerank · giữ 4 + slot 5", recall: "0.8020", precision: "0.1724", stage: "Public", decision: "best", note: "Submission 886623" },
  { id: "R01", pipeline: "Representative-text retrieval", recall: "0.5561", precision: "0.1168", stage: "Local", decision: "reject", note: "Không hơn full text" },
  { id: "R02", pipeline: "Fine-tune E5 đơn giản", recall: "0.7567", precision: "—", stage: "Smoke", decision: "reject", note: "Giảm so với baseline smoke" },
  { id: "R03", pipeline: "LTR feature đơn giản", recall: "0.0422", precision: "0.0090", stage: "Local", decision: "reject", note: "Overfit mạnh" },
  { id: "R04", pipeline: "Hard-negative CE · 20 query", recall: "0.7233", precision: "0.1540", stage: "Smoke", decision: "reject", note: "Safe blend không hơn V5" },
  { id: "V6", pipeline: "Top-50 rerank · giữ 3 base", recall: "0.8264", precision: "0.1753", stage: "Local", decision: "candidate", note: "ZIP Public đã validate" },
];

const DEFAULT_TEAM = [
  {
    id: "shared",
    initials: "CT",
    role: "Code & submission chung",
    owner: "Cả team · luân phiên",
    status: "Việc chung",
    summary: "Cùng giữ pipeline chuẩn, hợp nhất kết quả và kiểm tra submission trước khi nộp.",
    done: "V1–V6, validator, ZIP và experiment log",
    next: "Nộp V6, ghi Public score và freeze candidate tốt nhất",
    deliverable: "V6 receipt + submission registry",
    accent: "blue",
  },
  {
    id: "legal",
    initials: "LB",
    role: "Legal / BM25",
    owner: "01 người mạnh xử lý văn bản / rule-based",
    status: "Ưu tiên P0",
    summary: "Cải thiện lexical retrieval theo cấu trúc văn bản pháp luật.",
    done: "BM25 normalize, title, stopwords, query profiles",
    next: "Tách Điều/Khoản/Điểm và tạo legal synonym dictionary",
    deliverable: "legal_bm25_top100.json + ablation report",
    accent: "sand",
  },
  {
    id: "dense",
    initials: "DR",
    role: "Dense Retrieval",
    owner: "01 người có nền tảng embeddings / model",
    status: "Ưu tiên P1",
    summary: "Tìm văn bản theo ngữ nghĩa và tăng candidate Recall.",
    done: "E5 embeddings, dense retrieval, question-KNN",
    next: "Hard-negative mining và benchmark model embedding mới",
    deliverable: "dense_v2_top100.json + model card",
    accent: "violet",
  },
  {
    id: "reranking",
    initials: "RR",
    role: "Reranking",
    owner: "01 người thích thử nghiệm ranking / model",
    status: "Ưu tiên P0",
    summary: "Chọn đúng 5 ID cuối từ candidate top 100.",
    done: "Generic CE, safe rerank và deep top-50 rerank",
    next: "Feature ablation và legal-aware reranker",
    deliverable: "reranker_v2_k5.json + error slices",
    accent: "mint",
  },
  {
    id: "evaluation",
    initials: "EV",
    role: "Evaluation / MLOps",
    owner: "01 người cẩn thận về data & metric",
    status: "Ưu tiên P0",
    summary: "Bảo vệ validation, đo metric và theo dõi lỗi.",
    done: "Dev split, evaluator, candidate ceiling analysis",
    next: "Cross-validation và dashboard error taxonomy",
    deliverable: "cv_report.md + error_analysis.json",
    accent: "amber",
  },
];

const SPRINT_TASKS = [
  { id: "submit-v6", priority: "P0", title: "Nộp V6 và ghi lại Public score", owner: "Cả team", gate: "Có submission ID, Recall và Precision thật", done: false },
  { id: "legal-chunk", priority: "P0", title: "Legal-aware chunking theo Điều/Khoản/Điểm", owner: "Legal/BM25", gate: "Recall@100 tăng hoặc rerank top 5 tăng ≥ 0.01", done: false },
  { id: "error-slice", priority: "P0", title: "Phân loại 221 query V5 còn sai", owner: "Evaluation", gate: "Có taxonomy + 20 ví dụ đại diện", done: false },
  { id: "dense-v2", priority: "P1", title: "Benchmark dense model và hard negatives", owner: "Dense Retrieval", gate: "Cùng dev split, cùng top 100, có runtime", done: false },
  { id: "learned-fusion", priority: "P1", title: "Learned fusion theo query group", owner: "Reranking", gate: "Vượt RRF cố định trên full dev", done: false },
  { id: "cv", priority: "P1", title: "Cross-validation chống overfit", owner: "Evaluation", gate: "Cải thiện ổn định trên ≥ 3 folds", done: false },
];

const DECISION_LABELS = {
  keep: "Giữ branch",
  best: "Public best",
  candidate: "Candidate",
  reject: "Đã loại",
  replaced: "Đã thay thế",
};

function loadStored(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value ? JSON.parse(value) : fallback;
  } catch {
    return fallback;
  }
}

function Metric({ label, value, helper, tone = "blue" }) {
  return (
    <div className="metric">
      <span className="metric__label">{label}</span>
      <strong className={`metric__value metric__value--${tone}`}>{value}</strong>
      <span className="metric__helper">{helper}</span>
    </div>
  );
}

function MilestoneTrack({ expanded = false }) {
  return (
    <div className={`milestone-track ${expanded ? "milestone-track--expanded" : ""}`} aria-label="Hành trình điểm số">
      {MILESTONES.map((item, index) => (
        <div className="milestone" key={item.label}>
          <div className="milestone__copy">
            <strong>{item.label}</strong>
            <span>{item.score}</span>
            {expanded && <small>{item.note}</small>}
          </div>
          <div className="milestone__rail">
            <span className={`milestone__dot ${index === MILESTONES.length - 1 ? "is-current" : ""}`} />
            {index < MILESTONES.length - 1 && <span className={`milestone__line ${index === MILESTONES.length - 2 ? "is-current" : ""}`} />}
          </div>
          <small className="milestone__date">{item.date}</small>
        </div>
      ))}
    </div>
  );
}

function StatusDot({ status }) {
  const tone = status === "Sẵn sàng" ? "green" : status === "Blocked" ? "red" : "blue";
  return <span className={`status-inline status-inline--${tone}`}><span />{status}</span>;
}

function Overview({ onNavigate, team }) {
  return (
    <div className="page page--overview">
      <header className="page-header">
        <div>
          <span className="eyebrow">LegalIR project report</span>
          <h1>Project Mission Control</h1>
          <p>Cập nhật 13/08/2026 · Public phase</p>
        </div>
        <button className="primary-button" onClick={() => onNavigate("sprint")}>
          Xem sprint tiếp theo <ArrowUpRight weight="bold" />
        </button>
      </header>

      <section className="score-hero" aria-label="Mục tiêu Recall">
        <h2>Public Recall <strong>0.8020</strong> <ArrowRight weight="regular" /> Mục tiêu <em>0.9591</em></h2>
        <p>"Rerank an toàn" là điểm Public tốt nhất. "Rerank top-50" đã vượt local gate và đang chờ điểm Codabench.</p>
      </section>

      <section className="metric-row" aria-label="Chỉ số chính">
        <Metric label="Điểm Public tốt nhất" value="0.8020" helper="Rerank an toàn · Mã V5" />
        <Metric label="Ứng viên đang kiểm tra" value="0.8264" helper="Rerank top-50 · Mã V6 · +0.0205" />
        <Metric label="Precision" value="0.1753" helper="Full dev · 1.003 queries" />
      </section>

      <section className="overview-section">
        <div className="section-heading section-heading--compact">
          <div>
            <span className="eyebrow">Progress</span>
            <h2>Hành trình thử nghiệm</h2>
          </div>
          <button className="text-button" onClick={() => onNavigate("progress")}>Xem chi tiết <ArrowRight /></button>
        </div>
        <MilestoneTrack />
      </section>

      <div className="success-banner">
        <CheckCircle weight="fill" />
        <div><strong>V6 ZIP đã sẵn sàng</strong><span>Đã validate 1.000 query × 5 ID · Chờ điểm Codabench</span></div>
        <button onClick={() => onNavigate("files")}>Xem artifact</button>
      </div>

      <section className="workstream-section">
        <div className="section-heading section-heading--compact">
          <div>
            <span className="eyebrow">Suggested split</span>
            <h2>Gợi ý chia 5 mảng</h2>
          </div>
          <button className="text-button" onClick={() => onNavigate("team")}>Xem đề xuất <ArrowRight /></button>
        </div>
        <div className="workstream-table" role="table" aria-label="Tóm tắt workstream">
          <div className="workstream-row workstream-row--head" role="row">
            <span>Mảng việc</span><span>Người phù hợp</span><span>Ưu tiên</span><span>Việc đề xuất</span>
          </div>
          {team.map((member) => (
            <div className="workstream-row" role="row" key={member.id}>
              <div className="workstream-role"><span className={`avatar avatar--${member.accent}`}>{member.initials}</span><strong>{member.role}</strong></div>
              <span>{member.owner}</span>
              <StatusDot status={member.status} />
              <span>{member.next}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function ProgressPage() {
  return (
    <div className="page">
      <header className="page-header page-header--simple">
        <div><span className="eyebrow">Từ baseline đến candidate</span><h1>Tiến độ dự án</h1><p>Mỗi bước chỉ được giữ khi tăng điểm trên cùng validation split.</p></div>
      </header>
      <section className="progress-summary">
        <div className="progress-summary__main">
          <span>Tăng Recall local</span>
          <strong>+0.4833</strong>
          <p>BM25 0.3431 → V6 0.8264</p>
        </div>
        <div className="progress-summary__secondary"><Target weight="duotone" /><div><strong>0.9422</strong><span>Candidate Recall@100</span></div></div>
        <div className="progress-summary__secondary"><MagnifyingGlass weight="duotone" /><div><strong>158 query</strong><span>Sai top 5 nhưng gold còn trong top 100</span></div></div>
      </section>
      <section className="panel">
        <div className="section-heading"><div><span className="eyebrow">Score timeline</span><h2>Năm mốc quan trọng</h2></div></div>
        <MilestoneTrack expanded />
      </section>
      <section className="insight-grid">
        <article><span className="insight-icon"><Scales weight="duotone" /></span><h3>Candidate generation đã khá mạnh</h3><p>Recall@100 đạt 0.9422. Pipeline thường đã tìm thấy gold nhưng chưa đưa đúng tài liệu vào 5 vị trí cuối.</p></article>
        <article><span className="insight-icon"><Sparkle weight="duotone" /></span><h3>Reranking là leverage lớn nhất</h3><p>Top-50 rerank + giữ 3 base tăng full-dev Recall từ 0.8059 lên 0.8264.</p></article>
        <article><span className="insight-icon"><Info weight="duotone" /></span><h3>Leaderboard không thay validation</h3><p>Mỗi submission phải có hypothesis, metric local, artifact và quyết định keep/reject trước khi nộp.</p></article>
      </section>
    </div>
  );
}

function ExperimentsPage() {
  const [filter, setFilter] = useState("all");
  const filtered = useMemo(() => EXPERIMENTS.filter((item) => {
    if (filter === "all") return true;
    if (filter === "kept") return ["keep", "best", "candidate"].includes(item.decision);
    return item.decision === "reject";
  }), [filter]);

  return (
    <div className="page">
      <header className="page-header page-header--simple">
        <div><span className="eyebrow">Controlled experiments</span><h1>Experiment history</h1><p>Nhìn thấy cả thử nghiệm thành công lẫn thất bại để team không chạy lại việc cũ.</p></div>
      </header>
      <div className="toolbar" role="group" aria-label="Lọc experiment">
        {[["all", "Tất cả"], ["kept", "Đang giữ"], ["rejected", "Đã loại"]].map(([id, label]) => (
          <button key={id} className={filter === id ? "is-active" : ""} onClick={() => setFilter(id)}>{label}</button>
        ))}
        <span>{filtered.length} experiments</span>
      </div>
      <div className="experiment-table" role="table" aria-label="Lịch sử thử nghiệm LegalIR">
        <div className="experiment-row experiment-row--head" role="row">
          <span>ID</span><span>Pipeline</span><span>Recall</span><span>Precision</span><span>Stage</span><span>Decision</span>
        </div>
        {filtered.map((item) => (
          <div className="experiment-row" role="row" key={item.id}>
            <code>{item.id}</code>
            <div><strong>{item.pipeline}</strong><small>{item.note}</small></div>
            <strong className="number">{item.recall}</strong>
            <span className="number number--muted">{item.precision}</span>
            <span>{item.stage}</span>
            <span className={`decision decision--${item.decision}`}>{DECISION_LABELS[item.decision]}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function TeamPage() {
  return (
    <div className="page">
      <header className="page-header">
        <div><span className="eyebrow">Suggested split for 5 people</span><h1>Gợi ý phân công cho team</h1><p>Đây là đề xuất để cả team thảo luận và tự nhận việc; report này không theo dõi hay gán người chính thức.</p></div>
      </header>
      <div className="team-grid">
        {DEFAULT_TEAM.map((member) => (
          <article className="team-card" key={member.id}>
            <div className="team-card__header">
              <span className={`avatar avatar--large avatar--${member.accent}`}>{member.initials}</span>
              <div><span>{member.role}</span><h2>{member.owner}</h2></div>
              <StatusDot status={member.status} />
            </div>
            <p className="team-card__summary">{member.summary}</p>
            <dl>
              <div><dt>Nền tảng đã có</dt><dd>{member.done}</dd></div>
              <div><dt>Việc đề xuất</dt><dd>{member.next}</dd></div>
              <div><dt>Bàn giao tối thiểu</dt><dd><code>{member.deliverable}</code></dd></div>
            </dl>
          </article>
        ))}
      </div>
      <div className="team-rule">
        <ClipboardText weight="duotone" />
        <div><strong>Quy tắc phối hợp đề xuất</strong><p>Người nhận việc bàn giao command chạy, prediction dev, metric before/after và runtime. Chỉ đưa sang Public khi tăng ít nhất +0.01 Recall local.</p></div>
      </div>
    </div>
  );
}

function SprintPage() {
  const [tasks, setTasks] = useState(() => loadStored("legalir-sprint", SPRINT_TASKS));
  const completed = tasks.filter((task) => task.done).length;

  function toggle(id) {
    const next = tasks.map((task) => task.id === id ? { ...task, done: !task.done } : task);
    setTasks(next);
    localStorage.setItem("legalir-sprint", JSON.stringify(next));
  }

  return (
    <div className="page">
      <header className="page-header page-header--simple">
        <div><span className="eyebrow">Sprint 13–20/08/2026</span><h1>Việc cần làm tiếp theo</h1><p>Mục tiêu: tăng khả năng chọn đúng top 5 và vượt local Recall 0.84 một cách ổn định.</p></div>
      </header>
      <section className="sprint-goal">
        <div><Target weight="duotone" /><span>North-star gate</span><strong>Recall local ≥ 0.84</strong><p>Không giảm Precision và cải thiện ổn định qua cross-validation.</p></div>
        <div className="sprint-progress"><span>{completed}/{tasks.length} tasks hoàn thành</span><div><i style={{ width: `${(completed / tasks.length) * 100}%` }} /></div></div>
      </section>
      <div className="task-list">
        {tasks.map((task) => (
          <button className={`task-row ${task.done ? "is-done" : ""}`} key={task.id} onClick={() => toggle(task.id)}>
            <span className="task-check">{task.done ? <Check weight="bold" /> : <Circle />}</span>
            <span className={`priority priority--${task.priority.toLowerCase()}`}>{task.priority}</span>
            <span className="task-copy"><strong>{task.title}</strong><small>Gate: {task.gate}</small></span>
            <span className="task-owner">{task.owner}</span>
          </button>
        ))}
      </div>
      <section className="do-not-list">
        <h2>Quy tắc tiết kiệm compute & token</h2>
        <div><span><X weight="bold" /></span><p>Không scale full khi smoke test chưa tăng điểm.</p></div>
        <div><span><X weight="bold" /></span><p>Không để mỗi người tự tạo dev split hoặc submission format riêng.</p></div>
        <div><span><X weight="bold" /></span><p>Không dùng Public leaderboard thay cho local validation.</p></div>
      </section>
    </div>
  );
}

function FilesPage() {
  return (
    <div className="page">
      <header className="page-header page-header--simple">
        <div><span className="eyebrow">How to share this report</span><h1>Tài liệu chung</h1><p>Đây là report để cả team đọc. Nó không chứa source code, dataset hay command chạy trên máy của bạn.</p></div>
      </header>
      <div className="file-grid">
        <section className="file-group">
          <div className="file-group__title"><Files weight="duotone" /><h2>1. Gửi cho cả team ngay bây giờ</h2></div>
          <div className="file-row"><div><strong>File HTML report</strong><span>Gửi file này qua Zalo, Drive hoặc Discord. Mọi người chỉ cần mở để đọc tiến độ và hướng chia việc.</span></div></div>
          <div className="file-row"><div><strong>Không cần gửi command</strong><span>Command chỉ chạy được khi người đó đã có source code, dataset và môi trường cài đặt giống nhau.</span></div></div>
        </section>
        <section className="file-group">
          <div className="file-group__title"><FolderOpen weight="duotone" /><h2>2. Khi team bắt đầu code chung</h2></div>
          <div className="file-row"><div><strong>Tạo một GitHub repository riêng</strong><span>Đặt source code, README và cấu trúc thư mục vào một repo private; mỗi người clone repo về máy mình.</span></div></div>
          <div className="file-row"><div><strong>Chia sẻ dataset theo thể lệ</strong><span>Chỉ người được phép mới nhận dữ liệu. Không nhét dataset vào report HTML hoặc public GitHub.</span></div></div>
        </section>
      </div>
      <section className="submission-checklist">
        <div><CheckCircle weight="fill" /><strong>Phân biệt 3 thứ</strong></div>
        <ul><li><strong>HTML report:</strong> gửi cho tất cả để đọc.</li><li><strong>Source code:</strong> chia sẻ sau qua một GitHub repo private.</li><li><strong>Submission ZIP:</strong> chỉ người nộp trên Codabench cần dùng.</li></ul>
        <code>Không cần gửi source hay ZIP chỉ để mọi người xem report.</code>
      </section>
    </div>
  );
}

export function App() {
  const [activeTab, setActiveTab] = useState("overview");

  useEffect(() => {
    const handler = (event) => {
      const button = event.target.closest("button[data-nav]");
      if (button) setActiveTab(button.dataset.nav);
    };
    document.addEventListener("click", handler);
    return () => document.removeEventListener("click", handler);
  }, []);

  const page = {
    overview: <Overview onNavigate={setActiveTab} team={DEFAULT_TEAM} />,
    progress: <ProgressPage />,
    experiments: <ExperimentsPage />,
    team: <TeamPage />,
    sprint: <SprintPage />,
    files: <FilesPage />,
  }[activeTab];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button className="brand" onClick={() => setActiveTab("overview")} aria-label="Về tổng quan">
          <span className="brand-mark"><img src="/assets/legalir-mark.png" alt="" /></span>
          <span><strong>UIT DSC 2026</strong><small>LegalIR Mission Control</small></span>
        </button>
        <nav aria-label="Điều hướng chính">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return <button key={item.id} className={activeTab === item.id ? "is-active" : ""} aria-current={activeTab === item.id ? "page" : undefined} onClick={() => setActiveTab(item.id)}><Icon weight={activeTab === item.id ? "fill" : "regular"} /><span>{item.label}</span></button>;
          })}
        </nav>
        <div className="sidebar-footer">
          <span className="sidebar-footer__icon"><UsersThree weight="duotone" /></span>
          <div><strong>Team LegalIR Five</strong><small>Public phase · 2026</small></div>
        </div>
      </aside>
      <div className="mobile-header">
        <button className="mobile-brand" onClick={() => setActiveTab("overview")}><img src="/assets/legalir-mark.png" alt="" /><span>LegalIR Mission Control</span></button>
        <span>13/08/2026</span>
      </div>
      <nav className="mobile-nav" aria-label="Điều hướng mobile">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return <button key={item.id} className={activeTab === item.id ? "is-active" : ""} onClick={() => setActiveTab(item.id)}><Icon /><span>{item.label}</span></button>;
        })}
      </nav>
      <main className="main-content" id="main-content">{page}</main>
    </div>
  );
}
