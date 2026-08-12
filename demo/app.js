const examples = {
  "quản tài viên": {
    answer: "Quản tài viên phải đáp ứng các điều kiện hành nghề theo quy định pháp luật về phá sản, bao gồm điều kiện về chuyên môn, kinh nghiệm và chứng chỉ hành nghề.",
    sources: [
      ["277391", "Điều kiện hành nghề quản tài viên", "Chứa các điều kiện và tiêu chuẩn hành nghề."],
      ["281042", "Quy định về quản tài viên", "Nêu phạm vi quyền và nghĩa vụ trong quá trình hành nghề."],
      ["295057", "Hoạt động quản lý, thanh lý tài sản", "Bổ sung quy định liên quan đến nghiệp vụ."],
    ],
  },
  "khai sinh": {
    answer: "Người đi đăng ký khai sinh cần chuẩn bị giấy tờ theo quy định về hộ tịch, thường gồm tờ khai đăng ký khai sinh và giấy chứng sinh hoặc giấy tờ thay thế phù hợp.",
    sources: [
      ["157168", "Thủ tục đăng ký khai sinh", "Nêu thành phần hồ sơ đăng ký khai sinh."],
      ["60283", "Đăng ký và quản lý hộ tịch", "Quy định cơ quan và trình tự tiếp nhận hồ sơ."],
      ["160112", "Giấy tờ hộ tịch", "Giải thích giấy tờ dùng trong thủ tục hộ tịch."],
    ],
  },
  "kiểm ngư": {
    answer: "Kiểm ngư viên trung cấp thực hiện nhiệm vụ chuyên môn kiểm ngư theo vị trí việc làm và quy định của cơ quan quản lý chuyên ngành.",
    sources: [
      ["14681", "Tiêu chuẩn và nhiệm vụ Kiểm ngư viên", "Passage minh họa cho câu hỏi warmup."],
      ["18032", "Quy định về kiểm ngư", "Nêu phạm vi hoạt động quản lý, kiểm tra."],
      ["177504", "Tổ chức và hoạt động chuyên ngành", "Văn bản liên quan trong corpus."],
    ],
  },
};

const form = document.querySelector('#searchForm');
const question = document.querySelector('#question');
const result = document.querySelector('#result');
const answerText = document.querySelector('#answerText');
const sources = document.querySelector('#sources');

function search(value) {
  const text = value.trim();
  if (!text) return;
  const key = Object.keys(examples).find((item) => text.toLowerCase().includes(item));
  const data = examples[key] || {
    answer: 'Demo chưa có dữ liệu cho câu hỏi này. Trong hệ thống thật, retriever sẽ tìm các passage và document ID phù hợp trong corpus.',
    sources: [['—', 'Chưa có kết quả demo', 'Hãy thử một câu hỏi gợi ý bên dưới.']],
  };
  answerText.textContent = data.answer;
  sources.innerHTML = data.sources.map(([id, name, passage], index) => `
    <div class="source">
      <div class="source-top"><span class="source-id">document_id: ${id}</span><span class="source-score">#${index + 1}</span></div>
      <p class="source-name">${name}</p>
      <p>${passage}</p>
    </div>
  `).join('');
  result.classList.remove('is-hidden');
  result.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

form.addEventListener('submit', (event) => { event.preventDefault(); search(question.value); });
document.querySelectorAll('[data-question]').forEach((button) => {
  button.addEventListener('click', () => { question.value = button.dataset.question; search(question.value); });
});
