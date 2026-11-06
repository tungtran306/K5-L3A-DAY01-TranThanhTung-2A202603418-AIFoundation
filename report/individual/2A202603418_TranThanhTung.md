# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:*2A202603418*
- **Họ và tên:*Trần Thanh Tùng*
- **SEED** (in ở ô đầu notebook): 1403

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | [x] |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | [x] |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | [x] |

## 2. Kết quả chính

| Mô hình | F1 test | F1 SUB 40 | p95 (ms/tin) | Tốc độ (tin/s) | Chi phí ($/1k tin) |
|---|---|---|---|---|---|
| 1. BoW + LogReg (tự code) | ~0.94 | 0.947 | ~0.15 ms | >6000 | 0.00 $ |
| 2. BoW + MLP (deep learning) | ~0.96 | 0.974 | ~0.42 ms | >2000 | 0.00 $ |
| 3a. Encoder + kNN | ~0.91 | 0.923 | ~70.0 ms (vượt SLA) | ~14 | 0.00 $ |
| 3b. Encoder + LogReg | ~0.97 | 0.974 | ~42.0 ms | ~24 | 0.00 $ |
| 3c. Encoder + MLP head | ~0.97 | 0.974 | ~45.0 ms | ~22 | 0.00 $ |

## 3. Dự đoán và thực tế

**1. Dự đoán tradeoff (Độ trễ kNN trên biểu đồ Pareto)**
- **Bạn đoán gì:** Em đoán kNN (phương pháp 3a) không cần huấn luyện nên sẽ chạy rất nhanh và nằm trong nhóm tối ưu đạt chuẩn SLA.
- **Thực tế ra gì:** kNN là mô hình duy nhất vi phạm SLA với độ trễ p95 khoảng 77 ms (vượt xa ngưỡng 50 ms) và bị các mô hình khác lấn át hoàn toàn trên biểu đồ Pareto.
- **Vì sao lệch:** kNN chuyển toàn bộ gánh nặng tính toán sang lúc inference. Với mỗi tin nhắn mới, nó phải đo khoảng cách tới toàn bộ vector trong tập dữ liệu O(N), khiến độ trễ khi gọi từng tin nhắn bị đội lên rất cao.

**2. Dự đoán dl_relu (Ảnh hưởng của hàm kích hoạt phi tuyến trong MLP)**
- **Bạn đoán gì:** Em nghĩ mạng MLP có 64 nơ-ron ẩn thì dù bỏ ReLU vẫn sẽ phân loại tốt hơn mô hình Logistic Regression tuyến tính nhờ có nhiều tham số hơn.
- **Thực tế ra gì:** Khi bỏ ReLU, độ chính xác và F1 của MLP tụt xuống ngang bằng với mô hình Logistic Regression (Phần 1).
- **Vì sao lệch:** Tích của các phép biến đổi tuyến tính liên tiếp W2(W1*X + b1) + b2 suy cho cùng chỉ tương đương một phép biến đổi tuyến tính đơn W'*X + b'. Không có hàm phi tuyến như ReLU, mạng sâu bị thoái hoá hoàn toàn.

**3. Dự đoán ml_lr (Tốc độ hội tụ khi tăng Learning Rate)**
- **Bạn đoán gì:** Em đoán tăng tốc độ học lên cao (learning rate = 10.0) sẽ giúp mô hình Gradient Descent học nhanh tới đích và đạt F1 tối ưu sớm hơn.
- **Thực tế ra gì:** Hàm loss dao động dữ dội, không giảm được và F1 rơi xuống mức rất thấp (mô hình không thể hội tụ).
- **Vì sao lệch:** Bước nhảy gradient quá lớn làm trọng số w bị nhảy vọt qua điểm cực tiểu của hàm mất mát thay vì từ từ trượt xuống đáy.

## 4. Phân tích 50 case khó

- **Model tụt điểm nhiều nhất:** Mô hình **BoW + LogReg (tự code)** bị tụt điểm nặng nề nhất khi chuyển sang tập 50 case khó (F1 tụt sâu từ ~0.94 xuống dưới 0.60).
- **Nhóm case gây khó khăn nhất:**
  1. *Né từ khoá (obfuscation):* Kẻ gian cố tình viết lách ký tự như `C0ngr4tulat10ns`, `F R E E`, `c.a.s.h`.
  2. *Spam tự nhiên (natural spam):* Tin nhắn rác lừa đảo nhưng hành văn đời thường, không dùng các từ khoá quảng cáo truyền thống.
- **2 case cụ thể và cơ chế khiến model sai:**
  1. *Case né từ khoá (Spam bị đoán nhầm thành Ham):* Tin nhắn lừa đảo dùng từ biến dạng như `C0ngr4tulat10ns! You have won c.a.s.h`. Cơ chế sai: BoW chỉ so khớp chuỗi ký tự cứng theo từ vựng (token matching). Khi từ bị thay đổi ký tự, BoW coi đây là từ lạ (Out-Of-Vocabulary) và gán trọng số 0, dẫn tới việc bỏ lọt tin rác.
  2. *Case tưởng spam nhưng là tin thường (Ham bị đoán nhầm thành Spam):* Tin nhắn cá nhân bình thường nhưng chứa từ khoá quảng cáo: `I WON the prize at school today, call me back for FREE tickets`. Cơ chế sai: BoW không hiểu ngữ cảnh câu mà chỉ cộng dồn trọng số của các từ khoá đơn lẻ (`WON`, `FREE`), làm tổng xác suất bị đẩy vượt ngưỡng và chặn nhầm tin hợp lệ.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

- **Bài toán hệ thống:** 10 triệu SMS/ngày (~116 tin/giây trung bình), yêu cầu khắt khe về độ trễ p95 < 50 ms/tin.
- **Phương án lựa chọn:** Chọn **Mô hình 2 (BoW + MLP)** làm bộ lọc chính, kết hợp với **Kiến trúc Router lai (Hybrid)**.
- **Dẫn chứng số liệu:**
  - *Về độ trễ & thông lượng:* BoW + MLP có độ trễ p95 chỉ ~0.42 ms, thông lượng xử lý hơn 2.000 tin/giây, đáp ứng thừa yêu cầu SLA 50 ms. Trong khi đó, mô hình kNN vi phạm SLA (p95 ~77 ms), các mô hình Transformer quá sát ngưỡng (p95 ~42-45 ms) và chi phí tài nguyên GPU quá đắt đỏ.
  - *Về chất lượng:* BoW + MLP đạt F1 ~0.96 trên test thường và 0.974 trên tập SUB 40.
  - *Chi phí LLM:* Nếu đưa toàn bộ 10 triệu tin/ngày qua LLM (gpt-6-luna), chi phí tiền API tốn khoảng 10.50$ x 30 = 315 USD/tháng, chưa kể độ trễ mạng của LLM luôn ở mức hàng trăm ms (vi phạm SLA).
- **Thiết kế Router 2 cấp:**
  - *Tầng 1 (BoW + MLP):* Xử lý 100% lưu lượng tin nhắn với độ trễ cực nhanh (< 1 ms). Những tin nhắn có xác suất rõ ràng (p < 0.2 hoặc p > 0.8, chiếm khoảng 90% lưu lượng) sẽ được quyết định ngay.
  - *Tầng 2 (Mô hình ngữ nghĩa / LLM):* Chỉ khoảng 5-10% tin nhắn rơi vào vùng lưỡng lự (0.2 <= p <= 0.8) mới được chuyển lên tầng trên để phân tích sâu ngữ nghĩa.
- **Đánh đổi chấp nhận:** Chấp nhận hy sinh một phần nhỏ khả năng phân tích ngữ nghĩa sâu của 90% tin nhắn rõ ràng để đổi lấy thông lượng cực cao, độ trễ tức thì và chi phí phần cứng rẻ.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| **TODO 4.1 (API Call)** | Cách truyền tham số `max_completion_tokens` và tắt suy luận (`reasoning_effort="none"`) để tránh lỗi 400 trên dòng model GPT-6. | Hướng dẫn dùng `**provider_kwargs(max_tokens)` để tự động chuẩn hoá tham số theo từng provider và cách đo latency bằng `perf_counter()`. | Tự hoàn thiện hàm `call_openai`, căn chỉnh thụt lề chuẩn và đảm bảo trả về đúng tuple `(text, latency)`. |
| **TODO 4.2 & 4.3 (Prompt & Parser)** | Cách thiết kế prompt phân loại đóng khung tin nhắn và xử lý bẫy chuỗi phủ định `"not spam"` trong `parse_label`. | Gợi ý cấu trúc prompt kết thúc bằng `Label:`; cảnh báo bẫy chuỗi con `"not spam"` chứa từ `"spam"` cần ưu tiên kiểm tra trước để tránh gán nhầm nhãn 1. | Tự viết bộ bóc tách bằng regex `re.findall` để đọc nhãn an toàn, chạy kiểm tra đạt `✅ 4.2–4.3 prompt / parse_label`. |
| **TODO 4.4 (Batch & Chi phí)** | Công thức ước lượng token từ độ dài văn bản và cách tính tổng chi phí USD theo mảng giá `PRICE_PER_1M`. | Cung cấp công thức quy đổi xấp xỉ $\text{tokens} \approx \text{words} / 0.75$, hướng dẫn bọc lời gọi trong `retry_with_backoff` và thống kê `n_invalid`. | Tự hoàn thiện hàm `llm_classify_batch`, chạy qua bộ kiểm tra tự động đạt `✅ 4.1 + 4.4 call_openai / llm_classify_batch`. |
| **Bắt lỗi 3 & Phản ánh** | Phân tích rủi ro khi dùng toán tử tìm kiếm chuỗi con `in` để trích xuất nhãn phân loại từ câu trả lời của LLM. | Chỉ ra lỗi False Positive khi câu trả lời có tiền tố phủ định và giải thích vai trò của `temperature=0` để đảm bảo tính tất định khi phân loại. | Tự viết phần giải thích ngắn gọn cho Bắt lỗi 3 và câu hỏi phản ánh về cơ chế suy luận của LLM. |
**Điều em hiểu ra hôm nay:**
- Trong các bài toán sản phẩm thực tế, điểm F1 cao không phải là tất cả; độ trễ (latency SLA), thông lượng xử lý và chi phí tài nguyên phần cứng là những yếu tố sống còn quyết định mô hình có thể đưa vào phục vụ người dùng thật hay không.
- Các kỹ thuật vector hoá ma trận (NumPy/PyTorch) không chỉ giúp code gọn gàng hơn mà còn quyết định sự khác biệt giữa một mô hình phản hồi trong 0.2 ms so với việc bị nghẽn hệ thống nếu dùng vòng lặp thủ công.
