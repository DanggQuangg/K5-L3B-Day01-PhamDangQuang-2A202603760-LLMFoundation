# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603760
- **Họ và tên:** Phạm Đăng Quang 
- **SEED** (in ở ô đầu notebook):6130

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.961   0.718      0.822        0.857   
2. BoW + MLP (deep learning)         0.967   0.864      0.913        0.919   
3a. Encoder (MiniLM) + kNN           0.866   0.942      0.902        0.900   
3b. Encoder (MiniLM) + LogReg        0.931   0.913      0.922        0.947   
3c. Encoder (MiniLM) + MLP head      0.950   0.932      0.941        0.974   
4a. LLM nhỏ zero-shot                0.864   0.950      0.905        0.905   
4b. LLM lớn zero-shot                0.900   0.900      0.900        0.900   
4c. LLM nhỏ few-shot                 0.905   0.950      0.927        0.927   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            4.965      19283.778       0.050   
2. BoW + MLP (deep learning)         9.286      14322.373       0.279   
3a. Encoder (MiniLM) + kNN          84.426         39.844      41.582   
3b. Encoder (MiniLM) + LogReg       84.984         46.618      34.017   
3c. Encoder (MiniLM) + MLP head     85.956         47.176      27.089   
4a. LLM nhỏ zero-shot                0.000          0.191    2280.691   
4b. LLM lớn zero-shot                0.000          1.478     528.181   
4c. LLM nhỏ few-shot                 0.000          0.661     777.878   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.079         0.000  
2. BoW + MLP (deep learning)          2.469         0.000  
3a. Encoder (MiniLM) + kNN           61.216         0.000  
3b. Encoder (MiniLM) + LogReg        61.846         0.000  
3c. Encoder (MiniLM) + MLP head      39.761         0.000  
4a. LLM nhỏ zero-shot             16730.580         0.000  
4b. LLM lớn zero-shot              1385.550         0.024  
4c. LLM nhỏ few-shot               4433.691         0.000  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

1. Dự đoán ml_f1 (Mô hình Bag-of-Words + Logistic Regression):

Em đoán: F1 chỉ đạt khoảng ~25%, precision cao hơn recall vì nghĩ rằng tập dữ liệu mất cân bằng khiến mô hình đoán hầu hết là ham nên các chỉ số TP sẽ rất thấp.
Thực tế ra: F1 trên tập test đạt 0.822 (F1 trên SUB 40 đạt 0.857), Precision = 0.961 và Recall = 0.718.
Vì sao lệch: Bạn đã đánh giá quá thấp khả năng phân tách của mô hình tuyến tính đơn giản. Trong tin nhắn SMS thực tế, spam thường tập trung các từ khóa đặc trưng rất mạnh . Logistic regression học được trọng số lớn cho các từ này và phân loại chính xác phần lớn trường hợp phổ thông.

2. Dự đoán llm_models (So sánh mô hình LLM lớn, nhỏ và Embedding):

Em đoán: Mô hình có tham số lớn hơn sẽ có F1 cao hơn; LLM sẽ có chất lượng vượt trội và thắng mô hình Embedding.
Thực tế ra: LLM lớn (4b) có F1 (SUB 40) = 0.900, thậm chí không cao hơn LLM nhỏ (4a) có F1 = 0.905. Đặc biệt, mô hình 3c. Encoder (MiniLM) + MLP head đạt F1 lên tới 0.974 (trên SUB) và 0.941 (trên toàn bộ test), vượt trội hơn cả LLM.
Vì sao lệch: Bạn nghĩ LLM với hàng tỷ tham số suy luận luôn áp đảo các mô hình nhỏ. Tuy nhiên, trên bài toán phân loại nhị phân SMS có nhãn rõ ràng, Sentence Transformer (MiniLM) được tinh chỉnh biểu diễn câu tốt kết hợp đầu MLP được huấn luyện trực tiếp trên tập train bài toán; trong khi LLM zero-shot chỉ nhận diện qua prompt hướng dẫn chung mà không được tối ưu trọng số trên tập dữ liệu cụ thể.

3. Dự đoán hard_eval (Hiệu năng trên 50 case khó):

Em đoán: Accuracy của các mô hình còn khoảng 80%; nhóm câu không có từ spam trong BoW là khó nhất.
Thực tế ra: BoW tụt dốc không phanh xuống Accuracy chỉ còn 50% và F1 rơi thảm hại từ 0.857 xuống 0.286. Ngược lại, LLM nhỏ đạt Accuracy lên đến 96% và F1 đạt 0.964.
Vì sao lệch: Bạn đánh giá quá cao độ bền vững của BoW trước các đòn tấn công ngữ nghĩa và kỹ thuật lách luật viết lệch ký tự leetspeak, đảo từ. Cơ chế token rời rạc khiến BoW hoàn toàn yếu trước các từ biến thể, trong khi LLM hiểu sâu cấu trúc ngôn ngữ và ngữ cảnh đời thực nên giải mã các bẫy này rất dễ dàng.

## 4. Phân tích 50 case khó

- **Model tụt nhiều nhất:** `1. BoW + LogReg (tự code)` tụt nặng nề nhất khi F1 rơi từ **0.857** (tập thường SUB) xuống chỉ còn **0.286** trên 50 case khó (độ chính xác chỉ 50%, tương đương đoán ngẫu nhiên). Kế tiếp là `2. BoW + MLP` (F1 tụt từ 0.919 xuống 0.410).
- **Nhóm case model tụt nhiều nhất:**
  - Đối với nhóm BoW: Tụt nặng nhất ở nhóm **Né từ khoá (obfuscation)** và **Spam giọng tự nhiên / Thao túng**.
  - Đối với nhóm Embedding: Tụt nhiều nhất ở nhóm **Prompt injection** và **Thao túng / lừa đảo** (F1 tụt từ ~0.95 xuống ~0.58).
- **2 case cụ thể và cơ chế khiến model sai:**
  1. **Case #13 (Spam giọng tự nhiên — TẤT CẢ model đều đoán sai thành ham):**
     - *Nội dung:* `"Hope you're well! Quick reminder that your photos from the event are ready, view & order prints"`
     - *Nhãn thật:* `spam` (thực chất là tin nhắn quảng cáo / chèo kéo in ảnh thương mại chưa xin phép).
     - *Cơ chế sai:* Tin nhắn hoàn toàn không chứa từ ngữ spam tiêu chuẩn nào ("free", "win", "urgent", "credit"), văn phong rất lịch sự, mở đầu bằng lời chào cá nhân thân thiện và đóng vai trò như một lời nhắc nhở sự kiện đời thường. BoW coi toàn bộ token là từ ngữ giao tiếp thông thường; Embedding gom cụm tin này vào không gian văn bản sự kiện/bạn bè; và ngay cả LLM cũng bị lừa bởi câu chữ tự nhiên không có dấu hiệu đe dọa hay ép buộc.
  2. **Case #32 (Né từ khoá / Leetspeak):**
     - *Nội dung:* `"Fr€€ ringt0nes 4 ur m0bile!!! txt T0NE to 85555 now"`
     - *Nhãn thật:* `spam`.
     - *Cơ chế sai:* Người gửi sử dụng ký tự thay thế để lách bộ lọc (`Fr€€`, `ringt0nes`, `4`, `ur`, `m0bile`). Đối với BoW, từ điển chỉ lưu các token nguyên bản như "free", "tones", "mobile". Các từ bị biến tấu trở thành từ ngoài từ vựng (OOV - Out of Vocabulary) hoặc tạo vector toàn số 0. Do không kích hoạt được bất kỳ trọng số spam nào, BoW tự động kết luận tin nhắn là `ham` theo xác suất nền.

## 5. Thử thách cá nhân

- **SEED:** 6130 → `MY_CHALLENGE = 6130 % 6 = 4` (tương ứng **Đề #5**).
- **Đề bài:** *"Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin."*
- **Cách làm:**
  Thiết kế 4 trường hợp biên để thử thách giới hạn của biểu diễn Bag-of-Words và Sentence Embedding:
  1. *Spam 1 (Né từ khóa quà tặng):* `"C0ngr4ts! U won a fr33 gift c4rd, cl1ck h3r3 to r3d33m: gift-now.com"` (nhãn thật: SPAM)
  2. *Spam 2 (Né từ khóa vay nợ):* `"G3t your cash 1oan appr0ved now, n0 cred1t check required. Reply YES"` (nhãn thật: SPAM)
  3. *Ham 1 (Chứa từ 'free' rủ đi cafe):* `"Are you free for lunch today? Let's grab some coffee this afternoon"` (nhãn thật: HAM)
  4. *Ham 2 (Chứa từ 'win' kể chuyện bóng đá):* `"Our football team had a great win yesterday! Let's celebrate tonight"` (nhãn thật: HAM)
  Chạy qua 2 mô hình `1. BoW + LogReg` và `3b. Encoder (MiniLM) + LogReg` để so sánh nhãn và xác suất đầu ra.

- **Kết quả:**
  - *Tin 1:* BoW đoán **HAM** (p=11.2% - SAI). Embedding + LR đoán **SPAM** (p=99.4% - ĐÚNG).
  - *Tin 2:* BoW đoán **HAM** (p=13.0% - SAI). Embedding + LR đoán **HAM** (p=38.4% - SAI).
  - *Tin 3:* BoW đoán **HAM** (p=3.7% - ĐÚNG). Embedding + LR đoán **HAM** (p=1.8% - ĐÚNG).
  - *Tin 4:* BoW đoán **HAM** (p=11.1% - ĐÚNG). Embedding + LR đoán **HAM** (p=0.2% - ĐÚNG).

- **Giải thích cơ chế:**
  - *Tin 1:* BoW thất bại hoàn toàn vì các biến thể leetspeak (`C0ngr4ts`, `fr33`, `c4rd`, `cl1ck`) biến thành các từ nằm ngoài từ điển (OOV). Ngược lại, Sentence Transformer dùng bộ tách từ Subword và ánh xạ ngữ nghĩa toàn câu nên vẫn nhận ra cấu trúc lừa đảo quà tặng quen thuộc, cho độ tự tin spam tới 99.4%.
  - *Tin 2:* Kỹ thuật thay thế số (`1oan`, `n0 cred1t`) không chỉ vô hiệu hóa BoW mà còn làm biến dạng ngữ nghĩa vector embedding ở câu quá ngắn, khiến cả hai mô hình không đủ tự tin để vượt qua ngưỡng kích hoạt 0.5.
  - *Tin 3 & 4:* Cả hai mô hình đều xử lý tốt: BoW nhờ các từ đời thường áp đảo trọng số ("lunch", "coffee", "football team", "celebrate") nên kéo xác suất spam xuống; còn Embedding hiểu trọn vẹn ngữ cảnh giao lưu xã hội và chiến thắng thể thao thay vì bị đánh lừa bởi từ đơn lẻ.

## 6. Phương án cho sản phẩm thật

Với yêu cầu hệ thống nhà mạng xử lý **10 triệu SMS/ngày** (trung bình ~116 tin/giây, cao điểm có thể lên đến 300–500 tin/giây) và cam kết SLA **p95 < 50 ms/tin**:

- **Phương án lựa chọn:**
  - **Mô hình cốt lõi triển khai trực tiếp:** Chọn **`3c. Encoder (MiniLM) + MLP head`** (hoặc `3b. Encoder (MiniLM) + LogReg`).
  - **Dẫn chứng số liệu từ thực nghiệm:**
    - **Độ trễ:** Đạt chuẩn SLA xuất sắc với **p95 = 39.76 ms** (< 50 ms) và **p50 = 27.09 ms**.
    - **Chất lượng:** Đạt độ chính xác cao nhất toàn hệ thống với **F1 (test) = 0.941** và **F1 (SUB 40) = 0.974** (vượt xa mọi mô hình BoW và cả 3 mô hình LLM).
    - **Chi phí vận hành:** **0.000 USD** tiền API (chạy suy luận on-premise trên máy chủ nội bộ).
    - **Thông lượng:** Đạt **47.18 tin/s** trên 1 luồng Colab CPU đơn lẻ. Khi triển khai production chỉ cần một cụm nhỏ (vài worker có GPU như NVIDIA T4/L4 hoặc tối ưu ONNX Runtime) là dễ dàng gánh trọn vẹn tải 10 triệu SMS/ngày.
  - **Thiết kế tối ưu thực tế (Kiến trúc Router 2 tầng):**
    - *Tầng 1 (Bộ lọc thô BoW):* Dùng `1. BoW + LogReg` (p95 siêu tốc **0.079 ms**, thông lượng > 19.000 tin/s) để quét ngay lập tức. Nếu $p < 0.1$ (chắc chắn ham) hoặc $p > 0.9$ (chắc chắn spam) thì trả về kết quả ngay (giải quyết ~80-85% lưu lượng).
    - *Tầng 2 (Bộ lọc tinh Encoder):* Với các tin nhắn mập mờ nằm trong vùng $p \in [0.1, 0.9]$, đẩy qua `3c. Encoder + MLP head` (p95 ~ 39.8 ms) để thẩm định ngữ nghĩa sâu.
  - **Lý do loại bỏ hoàn toàn LLM:**
    - LLM nhỏ zero-shot có p95 lên tới **16.730 ms** (chậm gấp 334 lần quy định SLA).
    - LLM lớn có p95 là **1.385 ms** (chậm gấp gần 28 lần quy định SLA), đồng thời tiêu tốn **0.024 USD / 1k tin**, tương đương **~240 USD/ngày (~7.200 USD/tháng)** nếu áp dụng cho 10 triệu tin/ngày.
- **Chấp nhận đánh đổi (Trade-off):**
  - Chấp nhận tốn thêm tài nguyên máy chủ (độ trễ ~40 ms so với 0.08 ms của BoW) để đổi lấy F1 cao và khả năng chống chịu lách luật ngữ nghĩa tốt hơn nhiều so với BoW.
  - Chấp nhận khả năng phòng thủ prompt injection ở mức vừa phải (thực nghiệm 50 case khó đạt 62%) để đảm bảo 100% về độ trễ và chi phí khả thi trong thực tế.

## 7. AI log và điều đã học

### Bảng nhật ký dùng AI:
| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| 
| Phần 5 (Trade-off) | Hỏi về nguyên lý đường Pareto (Pareto frontier) và cách đọc biểu đồ đánh đổi giữa độ trễ và tỷ lệ lỗi | Giải thích các điểm nằm trên đường biên là tối ưu (không điểm nào nhanh hơn mà ít lỗi hơn), các điểm khác là bị lấn át | Đối chiếu trực tiếp với biểu đồ scatter plot của notebook, nhận thấy toàn bộ cụm LLM bị lấn át bởi mô hình Encoder |

### Điều bạn hiểu ra hôm nay mà trước đó chưa hiểu:
Trước buổi lab này, em từng mặc định rằng LLM luôn là đỉnh cao công nghệ và áp đảo mọi bài toán NLP, mô hình càng nhiều tham số thì hiệu quả phân loại càng vượt trội. Nhưng qua thực nghiệm trực tiếp hôm nay, em nhận ra rằng với các bài toán phân loại chuẩn mực đã có dữ liệu gán nhãn, một mô hình Sentence Transformer nhỏ gọn kết hợp đầu MLP cho độ chính xác cao hơn, chạy nhanh hơn hàng trăm lần và tiết kiệm hoàn toàn chi phí so với việc gọi API LLM. LLM thực chất chỉ phát huy thế mạnh độc tôn ở giai đoạn ban đầu khi hoàn toàn không có dữ liệu gán nhãn hoặc làm tầng phòng thủ phụ đối đầu với các cuộc tấn công ngữ nghĩa và lừa đảo tinh vi.