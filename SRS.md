# BẢN ĐẶC TẢ YÊU CẦU PHẦN MỀM (SOFTWARE REQUIREMENTS SPECIFICATION - SRS)

## HỆ THỐNG WEB AI/FINTECH DỰ BÁO MỨC ĐỘ TÁC ĐỘNG CỦA TIN TỨC TÀI CHÍNH ĐẾN THỊ TRƯỜNG

> **Đề tài nghiên cứu:** *"Dự báo mức độ tác động của tin tức tài chính đến thị trường: Thực nghiệm so sánh giữa mô hình RNN (Bi-LSTM), BERT-base và FinBERT"*  
> **Phiên bản:** `2.1.0-RELEASE (TỰA Edition)`  
> **Ngày phát hành:** `20/10/2026`  
> **Ngôn ngữ & Công nghệ:** `Python 3.13` | `Streamlit` | `PyTorch` | `HuggingFace Transformers` | `Plotly`  
> **Chuẩn phong cách thiết kế:** `TỰA - "Nét Vẽ Yêu Thương"` (*tuacharity.site Inspired*)  
> **Môi trường hoạt động:** Windows x64 (Local & On-Premise GPU/CPU Server)

---

## 1. TỔNG QUAN HỆ THỐNG (SYSTEM OVERVIEW)

### 1.1. Bối cảnh Đề tài & Vấn đề Thực tế (Problem Statement & Motivation)
Trong thị trường tài chính hiện đại, thông tin bất đối xứng và tốc độ truyền tin là yếu tố định hình xu hướng dòng tiền. Hàng triệu bản tin vĩ mô, báo cáo tài chính doanh nghiệp và diễn biến địa chính trị xuất hiện mỗi ngày tạo nên một "biển nhiễu" khổng lồ.
- **Thách thức lớn nhất đối với giới đầu tư & nhà quản trị rủi ro:** Phân loại chính xác **mức độ biến động** mà một bản tin có thể gây ra cho chỉ số thị trường (S&P 500 / Sensex), tách biệt các rung chấn mạnh mẽ (**HIGH**) với biến động bình thường (**MEDIUM**) hoặc tin tức nhiễu không gây tác động (**LOW**).
- **Hạn chế của mô hình truyền thống:** Các mạng nơ-ron hồi quy cổ điển như **Bi-LSTM** thường bị nghẽn cổ chai ngữ nghĩa khi câu dài, dẫn đến tỷ lệ **báo động giả (False Alarm) lên tới 58.33%** (Precision lớp HIGH chỉ đạt 41.67%). Việc này gây lãng phí hàng triệu USD chi phí phòng hộ danh mục (Hedging cost) không cần thiết.
- **Giải pháp FinBERT:** Ứng dụng mô hình ngôn ngữ lớn chuyên biệt hóa cho tài chính (**ProsusAI/finbert**), tinh chỉnh cơ chế **Self-Attention** để bắt trúng các từ khóa bước ngoặt vĩ mô (*hike, plunge, crisis, inflation, sanctions...*), nâng độ chuẩn xác lớp rủi ro lên **96.96%**.

### 1.2. Mục tiêu Hệ thống (System Objectives)
1. **Triển khai Trực quan hóa Nghiên cứu Học thuật:** Đưa toàn bộ mô hình và tập dữ liệu thực nghiệm quy mô lớn (**13,992 mẫu Test độc lập và 13,992 mẫu Validation**) lên nền tảng Web có thể tương tác 100%.
2. **Cơ chế Cảnh báo Sớm Thời gian thực (Real-Time Early Warning System - EWS):** Tự động thu thập tin tức thời sự từ các hãng tin uy tín (Google News, Yahoo Finance, CNBC) và đưa ra phán quyết quản trị rủi ro danh mục.
3. **Mở hộp đen AI (Explainable AI - XAI):** Trực quan hóa đường ống NLP 4 trạm, bóc tách cơ chế Self-Attention tầng cuối và trình chiếu tọa độ vector tiềm ẩn 3 chiều (3D Latent Space).
4. **Chuẩn mực Thẩm mỹ TỰA:** Mang đến trải nghiệm giao diện người dùng ấm áp, đáng tin cậy, khoa học và giàu cảm xúc (*Human-Centered FinTech Dashboard*).

### 1.3. Đối tượng Sử dụng (Target Audiences)
| Nhóm người dùng | Mục đích sử dụng chính | Giá trị thụ hưởng |
|---|---|---|
| **Hội đồng Giám khảo & Giảng viên** | Thẩm định tính thực chứng, đối chiếu số liệu báo cáo, kiểm tra độ trung thực của mô hình. | Đối chiếu tức thì ma trận nhầm lẫn, bảng Model Selection và quy trình Anti-Leakage. |
| **Quản lý Quỹ & Chuyên viên Rủi ro** | Đánh giá rủi ro danh mục từ tin tức trực tiếp để quyết định phòng hộ (Hedging/Rebalancing). | Nhận khuyến nghị quản trị danh mục định lượng từ Bảng 5 báo cáo theo phán quyết FinBERT. |
| **Nhà nghiên cứu AI / NLP** | Phân tích cơ chế chú ý của Transformer và so sánh với RNN. | Quan sát Heatmap Layer 11 và biểu đồ 3D tương tác xoay 360 độ. |
| **Nhà đầu tư cá nhân** | Đọc tin tức và xem góc nhìn phân tích định lượng khách quan. | Tránh tâm lý hoảng loạn FOMO trước các tin tức giật gân nhưng thực chất là nhóm LOW. |

---

## 2. KIẾN TRÚC HỆ THỐNG & ĐƯỜNG ỐNG DỮ LIỆU (SYSTEM ARCHITECTURE)

### 2.1. Sơ đồ Kiến trúc Tổng thể (Overall Architecture)

```
[ THỰC NGHIỆM DỮ LIỆU ĐÃ LÀM SẠCH ]          [ NGUỒN TIN TỨC REAL-TIME ]
  - Tập Test Độc lập (13,992 mẫu)             - Google News Financial RSS
  - Tập Validation (13,992 mẫu)               - Yahoo Finance RSS
  (Lưu trữ tại thư mục nguồn READ-ONLY)        - CNBC Markets RSS / Fallback Cache
                      │                                       │
                      ▼                                       ▼
  ┌───────────────────────────────────────────────────────────────┐
  │         MODULE TIỀN XỬ LÝ CHỐNG RÒ RỈ (ANTI-LEAKAGE)          │
  │  - Loại bỏ URLs (regex: https?://\S+|www\.\S+)                │
  │  - Loại bỏ thẻ HTML (regex: <.*?>)                            │
  │  - Chuẩn hóa khoảng trắng & cắt max_length = 64 tokens        │
  │  - Loại bỏ metadata phụ trợ để tránh rò rỉ thông tin          │
  └───────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
  ┌───────────────────────────────────────────────────────────────┐
  │              BỘ SUY LUẬN ĐỒNG THỜI 3 MÔ HÌNH HỌC MÁY          │
  │  ┌─────────────────┬───────────────────┬───────────────────┐  │
  │  │ 1. Bi-LSTM      │ 2. BERT-base      │ 3. FinBERT        │  │
  │  │ Word-level      │ WordPiece Subword │ Domain Financial  │  │
  │  │ 20,000 Vocab    │ General Purpose   │ Champion Model    │  │
  │  └─────────────────┴───────────────────┴───────────────────┘  │
  └───────────────────────────────┬───────────────────────────────┘
                                  │
            ┌─────────────────────┴─────────────────────┐
            ▼                                           ▼
  ┌───────────────────────────┐               ┌───────────────────────────┐
  │  CƠ CHẾ ĐỒNG THUẬN & EWS  │               │   GIẢI THÍCH MÔ HÌNH XAI  │
  │  - Tính độ đồng thuận     │               │  - Layer 11 Self-Attention│
  │  - Bảng 5 Khuyến nghị quỹ │               │  - Heatmap phát sáng token│
  │  - Phân loại HIGH/MED/LOW │               │  - Chiếu không gian 3D    │
  └─────────────┬─────────────┘               └─────────────┬─────────────┘
                │                                           │
                └─────────────────────┬─────────────────────┘
                                      │
                                      ▼
  ┌───────────────────────────────────────────────────────────────────────┐
  │              GIAO DIỆN NGƯỜI DÙNG STREAMLIT TỰA EDITION                │
  │   - Tab 1: Kiểm định Dataset Testbed (Đồng bộ ma trận Test/Val)       │
  │   - Tab 2: Dự báo Real-Time & Cảnh báo Sớm (EWS Alert Desk)           │
  │   - Tab 3: Đường ống NLP 4 Trạm & Không gian Vector 3D                │
  │   - Tab 4: Benchmark Tổng thể Đối sánh Học thuật                      │
  └───────────────────────────────────────────────────────────────────────┘
```

### 2.2. Chi tiết 3 Thế hệ Mô hình (Model Specifications)

| Thông số kỹ thuật | Mô hình 1: Bi-LSTM (Baseline) | Mô hình 2: BERT-base | Mô hình 3: FinBERT (Champion) |
|---|---|---|---|
| **Kiến trúc cốt lõi** | Bidirectional LSTM (2 lớp) | Transformer Encoder (12 layers, 12 heads) | Tinh chỉnh từ BERT trên Financial Corpora |
| **Cơ chế Tokenizer** | Regex Word-level + Dictionary | WordPiece Subword Tokenizer | WordPiece Subword Tokenizer |
| **Kích thước Từ điển** | 20,000 tokens (`<pad>:0, <unk>:1`) | 30,522 tokens | 30,522 tokens (Chuyên ngữ tài chính) |
| **Kích thước Embedding** | 128 chiều | 768 chiều | 768 chiều |
| **Kích thước Ẩn (Hidden)** | 128 chiều ($\times 2 = 256$ chiều Bi) | 768 chiều | 768 chiều |
| **Tổng số Tham số** | ~2.8 Triệu tham số | ~110 Triệu tham số | ~110 Triệu tham số |
| **Độ dài chuỗi tối đa** | `max_length = 64` | `max_length = 64` | `max_length = 64` |
| **Trọng số lưu trữ** | `bilstm_best.pt` (~11.5 MB) | `bert_base_finetuned/` (~418 MB) | `finbert_best/` (~418 MB) |
| **Điểm mạnh chính** | Nhẹ, suy luận siêu nhanh | Bao phủ rủi ro tốt (HIGH Recall 93.77%) | Chuẩn xác rủi ro đỉnh cao (HIGH Precision 96.96%) |
| **Điểm yếu chí mạng** | Báo động giả cao (343 ca nhầm lẫn) | Thỉnh thoảng nhầm từ LOW sang HIGH | Yêu cầu tài nguyên tính toán cao hơn |

---

## 3. ĐẶC TẢ YÊU CẦU CHỨC NĂNG (FUNCTIONAL REQUIREMENTS)

### PHÂN HỆ 1: KIỂM ĐỊNH DATASET TESTBED (TAB 1)

- **Mã chức năng:** `FR-TB-01`
- **Mục tiêu:** Cung cấp môi trường kiểm thử trực tiếp trên dữ liệu mẫu thực chứng có sẵn nhãn Ground Truth.
- **Yêu cầu chi tiết:**
  1. **Chuyển đổi Tập dữ liệu (`FR-TB-01.1`):** Người dùng có thể chọn giữa:
     - `Tập KIỂM THỬ ĐỘC LẬP (Test - 13,992 mẫu)`: Tập dữ liệu chưa từng xuất hiện trong quá trình tối ưu siêu tham số.
     - `Tập KIỂM ĐỊNH (Validation - 13,992 mẫu)`: Tập dữ liệu dùng để chọn mô hình tối ưu.
  2. **Bộ lọc Nhãn Ground Truth (`FR-TB-01.2`):** Lọc theo 4 tiêu chí: `ALL`, `LOW (Nhãn 0)`, `MEDIUM (Nhãn 1)`, `HIGH (Nhãn 2)`.
  3. **Tìm kiếm Từ khóa Vĩ mô (`FR-TB-01.3`):** Lọc tức thì các bản tin có chứa từ khóa (như *Fed, Sensex, Bank, War, Inflation, Oil, Rate...*).
  4. **Bộ chọn Mẫu linh hoạt (`FR-TB-01.4`):**
     - Nút `🎲 Lấy mẫu ngẫu nhiên (Random Sample)`: Bốc ngẫu nhiên một mẫu tin thỏa mãn bộ lọc.
     - Dropdown chọn bản tin cụ thể: Đã được cắt ngắn tối đa 80 ký tự để không làm vỡ giao diện (ví dụ: `ID #53 [HIGH] Sensex plunges 1,200 points...`).
  5. **Thẻ So sánh Trực diện 3 Mô hình (`FR-TB-01.5`):**
     - Hiển thị nhãn dự đoán của Bi-LSTM, BERT-base, FinBERT cùng độ tự tin (Confidence %).
     - Huy hiệu thông báo trạng thái: `✅ CHÍNH XÁC` (xanh ngọc) hoặc `❌ NHẦM LẪN` (đỏ gạch) khi so khớp với Ground Truth.
  6. **Thanh Phân phối Xác suất Nằm ngang (`FR-TB-01.6`):**
     - Biểu đồ Plotly Stacked Bar thể hiện phân phối `[P_LOW, P_MEDIUM, P_HIGH]` tích lũy 100% của cả 3 mô hình.
     - Gán `key="prob_bars_tab1"` độc nhất.
  7. **Khối Mở rộng XAI Mẫu tin Hiện tại (`FR-TB-01.7`):**
     - Tích hợp Expander chứa Heatmap Attention Layer 11 và không gian vector 3D của chính bản tin đang xem (`key="latent_3d_tab1"`).
  8. **Đồng bộ Ma trận Nhầm lẫn Chuẩn xác (`FR-TB-01.8`):**
     - Khi chọn tập **Test**: Tự động hiển thị Ma trận nhầm lẫn Test độc lập của Champion FinBERT (`key="cm_tab1_test"`).
     - Khi chọn tập **Validation**: Tự động hiển thị Bộ 3 Ma trận nhầm lẫn đối sánh của Bi-LSTM, BERT-base, FinBERT (`key="cm_tab1_val"`).

---

### PHÂN HỆ 2: LUỒNG TIN REAL-TIME & CẢNH BÁO SỚM EWS (TAB 2)

- **Mã chức năng:** `FR-RT-02`
- **Mục tiêu:** Thu thập tin tức trực tiếp từ các thị trường quốc tế, phân loại rủi ro tức thời và đưa ra khuyến nghị phân bổ tài sản.
- **Yêu cầu chi tiết:**
  1. **Bộ Cào Tin Tự động Đa Nguồn (`FR-RT-02.1`):**
     - Kết nối RSS trực tiếp từ 3 nguồn: **Google News Financial**, **Yahoo Finance**, **CNBC Markets**.
     - Giới hạn 6 tin mới nhất mỗi nguồn, lọc bỏ các thẻ quảng cáo hoặc link lỗi.
  2. **Cơ chế Cache Dự phòng (High-Fidelity Fallback) (`FR-RT-02.2`):**
     - Tự động kích hoạt kho lưu trữ 10 bản tin chuẩn hóa nếu xảy ra mất mạng, timeout hoặc IP bị chặn bởi tường lửa RSS quốc tế.
  3. **Chế độ Tự Nhập Tin Tùy Ý (Custom Live Input) (`FR-RT-02.3`):**
     - Cho phép người dùng nhập trực tiếp Tiêu đề (Title) và Nội dung tóm tắt (Description).
     - Cung cấp **3 nút Preset điển hình**:
       * *Preset 1 (HIGH):* Fed bất ngờ tăng lãi suất 75 bps do lạm phát tăng vọt.
       * *Preset 2 (MEDIUM):* Doanh thu quý của nhóm sản xuất đi ngang.
       * *Preset 3 (LOW):* Lịch họp đại hội cổ đông thường niên thông qua cổ tức.
  4. **Cơ chế Đánh giá Đồng thuận (Consensus Engine) (`FR-RT-02.4`):**
     - Tính mức độ đồng thuận giữa 3 mô hình: Đồng thuận tuyệt đối (3/3), Đồng thuận đa số (2/3), hoặc Phân kỳ cảnh báo.
  5. **Bàn Cảnh báo Sớm Rủi ro (Early Warning Desk) (`FR-RT-02.5`):**
     - Dựa trên phán quyết của Champion FinBERT, xuất báo động theo 3 cấp độ:
       * 🔴 **HIGH (Thiên nga đen / Rủi ro vĩ mô):** Khuyến nghị phòng hộ (Hedging phái sinh), giảm đòn bẩy Margin, nâng tỷ trọng tiền mặt lên 30-50%.
       * 🟡 **MEDIUM (Biến động trung bình / Cục bộ):** Tạm dừng giải ngân mới, theo dõi ngưỡng hỗ trợ kỹ thuật, tái cân bằng nhóm cổ phiếu bị ảnh hưởng trực tiếp.
       * 🔵 **LOW (Tác động thấp / Nhiễu thông tin):** Giữ nguyên tỷ trọng, không hành động theo cảm tính, tránh bán tháo sai lầm trước tin giật gân vô thưởng vô phạt.

---

### PHÂN HỆ 3: ĐƯỜNG ỐNG NLP & KHÔNG GIAN VECTOR 3D (TAB 3)

- **Mã chức năng:** `FR-PL-03`
- **Mục tiêu:** Giải phẫu cấu trúc nội tại của pipeline xử lý ngôn ngữ tự nhiên và cung cấp công cụ XAI trực quan.
- **Yêu cầu chi tiết qua 4 Trạm:**
  1. **Trạm 1: Tiền xử lý Văn bản Chống rò rỉ (`FR-PL-03.1`):**
     - Khối hiển thị văn bản sử dụng class CSS `.tua-text-box` với `word-break: break-word; white-space: pre-wrap;` để bảo đảm **không bị tràn lề hay cắt cụt chữ**.
     - Đối chiếu văn bản thô ban đầu (kèm link URL/thẻ HTML) với văn bản sau làm sạch.
     - Báo cáo số lượng URLs đã gỡ, số thẻ HTML đã xóa, độ dài chuỗi ký tự trước và sau làm sạch.
  2. **Trạm 2: So sánh Phân tách Token (`FR-PL-03.2`):**
     - Trình diễn cơ chế Word-level Tokenizer của Bi-LSTM (từ điển 20k từ, biểu diễn mảng ID số nguyên điền số 0 `<pad>` đến độ dài 64).
     - Trình diễn cơ chế WordPiece của FinBERT (phân rã từ con subword `##`, gán nhãn `[CLS]` đầu câu và `[SEP]` cuối câu, triệt tiêu 100% lỗi OOV).
  3. **Trạm 3: Giải phẫu Self-Attention Tầng 11 (`FR-PL-03.3`):**
     - Trích xuất trọng số thực của token `[CLS]` tới mọi từ trong câu tại tầng cuối (Layer 11) của FinBERT.
     - Dựng khối **Heatmap HTML phát sáng**: Từ có mức chú ý cao (>0.60) có nền Terracotta và viền gạch nổi bật, từ trung bình (0.30 - 0.60) có nền vàng nghệ, từ thấp (<0.30) có nền rêu nhạt.
     - Bảng xếp hạng Top 8 từ khóa tài chính thu hút sự chú ý nhiều nhất.
  4. **Trạm 4: Hoạt ảnh Không gian Vector 3D (`FR-PL-03.4`):**
     - Biểu đồ Plotly Scatter3d thể hiện 3 cụm tiềm ẩn: Cụm LOW (xanh rêu), Cụm MEDIUM (vàng nghệ), Cụm HIGH (Terracotta).
     - Chiếu tọa độ bản tin hiện tại thành điểm kim cương phát sáng (`symbol='diamond'`).
     - Vẽ đường chiếu nét đứt (*Linear-Softmax projection line*) kết nối vector hiện tại với tâm cụm được phân lớp.
     - Cho phép xoay 360 độ, phóng to, thu nhỏ và thay đổi góc nhìn camera tự do.

---

### PHÂN HỆ 4: BENCHMARK TỔNG THỂ & BÁO CÁO THỰC NGHIỆM (TAB 4)

- **Mã chức năng:** `FR-BM-04`
- **Mục tiêu:** Trình bày số liệu thực chứng khoa học phục vụ báo cáo bảo vệ đề tài.
- **Yêu cầu chi tiết:**
  1. **Bảng Lựa chọn Mô hình (Model Selection Benchmark) (`FR-BM-04.1`):**
     - Đối sánh toàn diện 3 mô hình trên tập Validation (N=13,992 mẫu) với 7 chỉ số chuẩn: *Accuracy, Macro Precision, Macro Recall, Macro F1-Score, HIGH Precision, HIGH Recall, Số ca báo động giả HIGH*.
  2. **Biểu đồ Cột Đa chiều (`FR-BM-04.2`):**
     - Thể hiện trực quan khoảng cách hiệu năng giữa 3 mô hình (`key="benchmark_bar_tab4"`).
  3. **Bộ Ba Ma trận Nhầm lẫn Validation (`FR-BM-04.3`):**
     - Dựng 3 heatmap song song đạt chuẩn WCAG tương phản chữ, hiển thị số lượng mẫu và tỷ lệ % phân phối (`key="cm_tab4_val"`).
  4. **Báo cáo Kiểm định Độc lập Test Set (`FR-BM-04.4`):**
     - Bảng chỉ số chi tiết cho từng lớp của FinBERT trên 13,992 mẫu Test: Accuracy 98.94%, Macro F1 97.89%, HIGH Precision 96.96%, HIGH Recall 94.41%.
     - Ma trận nhầm lẫn Test độc lập (`key="cm_tab4_test"`).
  5. **Đúc kết Ý nghĩa Quản trị Đầu tư (`FR-BM-04.5`):**
     - Phân tích kinh tế lượng về chi phí của báo động giả (False Positive) trong giao dịch định lượng và lý do FinBERT được chọn làm Champion Model.

---

## 4. ĐẶC TẢ YÊU CẦU PHI CHỨC NĂNG (NON-FUNCTIONAL REQUIREMENTS - NFR)

### 4.1. Hiệu năng & Tốc độ Suy luận (Performance & Latency)
- **Thời gian suy luận (Inference Latency):** Suy luận song song 3 mô hình (Bi-LSTM, BERT-base, FinBERT) cho 1 bản tin mất **dưới 50ms trên GPU** (NVIDIA RTX/GTX) và **dưới 250ms trên CPU**.
- **Cơ chế Caching thông minh:**
  - Sử dụng `@st.cache_resource` để nạp 3 mô hình vào VRAM đúng 1 lần duy nhất khi khởi động máy chủ web, tránh tải lại gây quá tải bộ nhớ.
  - Sử dụng `@st.cache_data` để nạp tập dữ liệu 13,992 dòng vào RAM trong dưới 0.3 giây.
- **Tiêu thụ bộ nhớ:** Chiếm dụng dưới 2.5 GB RAM hệ thống và dưới 2.0 GB VRAM GPU.

### 4.2. Thẩm mỹ & Trải nghiệm Người dùng (UI/UX - TỰA Edition)
- **Hệ màu 60-30-10 ấm áp:**
  - Nền chính: Kem ấm `#FAF6F0` (60%)
  - Nền phụ / Thẻ card: Be nhạt `#F3EAE0` và Trắng ngà `#FFFFFF` (30%)
  - Điểm nhấn: Terracotta `#C96A4B`, Nâu đất `#8C4A32`, Vàng nghệ `#D9A441`, Xanh rêu `#7A9A8B` (10%)
- **Hệ thống Phông chữ chuẩn mực:**
  - Tiêu đề H1/H2: `Playfair Display`, serif sang trọng.
  - Nội dung Body: `Be Vietnam Pro`, sans-serif, **line-height: 1.7** tạo cảm giác "thở".
  - Chữ ký tự sự: `Dancing Script`, cursive ấm áp.
  - Số liệu & Code: `JetBrains Mono`, kỹ thuật số chuẩn xác.
- **Tương tác Micro-animations:** Nút bấm CTA bo tròn `14px`, màu Terracotta, hiệu ứng hover nhấc nhẹ 2px (`transform: translateY(-2px); box-shadow: 0 6px 18px rgba(140, 74, 50, 0.35);`).

### 4.3. Tiêu chuẩn Tiếp cận & Độ tương phản (Accessibility & Contrast)
- **Đạt chuẩn WCAG AA:** Toàn bộ văn bản hiển thị có tỷ lệ tương phản $\ge 4.5:1$ so với màu nền.
- **Ma trận nhầm lẫn thông minh:** Thuật toán tự động đảo màu chữ:
  - Ô có giá trị nhỏ / nền sáng ($\le 42\%$ max): Màu chữ nâu đen `#2E2620`.
  - Ô có giá trị lớn / nền đậm ($> 42\%$ max): Màu chữ trắng sáng `#FFFFFF`.
  - Tuyệt đối không để xảy ra tình trạng chữ trắng trên nền vàng nhạt/xanh nhạt gây lóa mắt.

### 4.4. Tính Ổn định Kỹ thuật (Technical Reliability)
- **Triệt tiêu lỗi `StreamlitDuplicateElementId`:** Tất cả các biểu đồ Plotly và widget tương tác đều bắt buộc khai báo tham số `key="..."` có tiền tố ngữ cảnh phân hệ rõ ràng.
- **Triệt tiêu lỗi tràn chữ (Text Overflow):** Mọi khối trích dẫn văn bản dài đều được bọc trong container có thuộc tính `word-break: break-word; white-space: pre-wrap; overflow-x: auto;`.

### 4.5. Bảo mật & Toàn vẹn Dữ liệu (Security & Data Integrity)
- **Bảo vệ Thư mục Nguồn (READ-ONLY):** Ứng dụng Web tuyệt đối chỉ đọc các file dữ liệu đã làm sạch và file trọng số mô hình từ thư mục nguồn bài tập lớn (`D:\Năm 3\ML\Btap lớn\ML_Dự báo tác động đến thị trường của tin tức`). Không ghi đè, sửa đổi hay tạo thêm file rác vào thư mục nguồn.
- **Xử lý Ngoại lệ Mạng (Fault Tolerance):** Nếu các luồng RSS bị chặn hoặc rớt kết nối, hệ thống tự động chuyển sang High-Fidelity Fallback Cache mà không làm văng Exception giao diện.

---

## 5. CẤU TRÚC THƯ MỤC & CÁC TỆP MÃ NGUỒN (CODEBASE ARCHITECTURE)

Toàn bộ ứng dụng web được tổ chức độc lập tại thư mục:  
`D:\Năm 3\ML\Btap lớn\Web dự báo mức độ tác động tin tức`

```
Web dự báo mức độ tác động tin tức/
│
├── app.py                  # Điểm khởi chạy chính Streamlit; quản lý điều hướng 4 Tab;
│                           # Injected CSS TỰA Theme; kết nối Controller và View.
│
├── model_engine.py         # Module Nghiệp vụ Trọng tâm:
│                           # - Định nghĩa kiến trúc PyTorch Bi-LSTM (2 lớp, 20k vocab).
│                           # - Nạp trọng số BERT-base & FinBERT qua HuggingFace AutoModel.
│                           # - Quy trình Anti-Leakage (clean_news_text, format_input_text).
│                           # - Hàm suy luận song song (predict_text) & tính tọa độ 3D.
│                           # - Trích xuất Self-Attention weights từ Layer 11 của FinBERT.
│                           # - Bảng chỉ số đối sánh tĩnh (BENCHMARK_METRICS).
│
├── visualizer.py           # Module Trực quan hóa Đồ họa Plotly & HTML:
│                           # - render_probability_bars: Thanh ngang phân phối xác suất.
│                           # - render_confusion_matrices: Ma trận nhầm lẫn WCAG contrast.
│                           # - render_benchmark_comparison_chart: Biểu đồ cột đa chiều.
│                           # - render_3d_latent_space: Không gian phân cụm 3D Scatter.
│                           # - render_attention_heatmap_html: Khối HTML phát sáng token.
│
├── news_scraper.py         # Module Cào tin tức Tài chính:
│                           # - fetch_all_financial_news: Kết nối RSS Google, Yahoo, CNBC.
│                           # - FALLBACK_NEWS_DATABASE: 10 bản tin mẫu dự phòng sự cố mạng.
│
├── requirements.txt        # Danh sách thư viện phụ thuộc (Streamlit, PyTorch, Plotly, etc.)
├── run_web.bat             # Script Batch khởi chạy nhanh 1-Click trên Windows.
├── SKILL.md                # Bản quy chuẩn thiết kế UI/UX TỰA – "Nét Vẽ Yêu Thương".
└── SRS.md                  # Bản Đặc tả Yêu cầu Phần mềm Chi tiết Toàn diện (Tài liệu này).
```

---

## 6. MA TRẬN ÁNH XẠ CHỈ SỐ THỰC NGHIỆM ĐÃ TÍCH HỢP (BENCHMARK DATA MATRIX)

Dưới đây là các chỉ số thực nghiệm được nhúng chuẩn xác trong mã nguồn hệ thống để đối chiếu với báo cáo học thuật:

### 6.1. Bảng Đối sánh Tập Validation ($N = 13,992$ mẫu)
| Tiêu chí Thực nghiệm | Bi-LSTM (Baseline) | BERT-base | FinBERT (Champion) | Đánh giá Chuyên môn |
|---|:---:|:---:|:---:|---|
| **Accuracy (Độ chính xác)** | 93.66% | 93.68% | **94.20%** | FinBERT dẫn đầu toàn diện |
| **Macro Precision** | 71.93% | 76.57% | **88.63%** | FinBERT vượt trội +16.7% |
| **Macro Recall** | 76.22% | **88.75%** | 85.92% | BERT-base bắt rộng nhất |
| **Macro F1-Score** | 73.91% | 81.36% | **87.16%** | FinBERT tối ưu độ hài hòa |
| **HIGH Recall (Bắt rủi ro)** | 78.47% | **93.77%** | 89.27% | BERT-base và FinBERT đều rất cao |
| **HIGH Precision (Chuẩn xác rủi ro)** | 41.67% | 51.52% | **94.46%** | FinBERT áp đảo (gần như không báo động giả) |
| **Báo động giả HIGH (False Alarms)** | 343 ca / 589 | 273 ca / 563 | **15 ca / 271** | FinBERT giảm 95.6% ca báo động giả |
| **Báo động giả từ LOW sang HIGH** | 85 ca | 12 ca | **0 ca (Tuyệt đối)** | FinBERT lọc nhiễu 100% |

### 6.2. Bảng Kiểm định Độc lập Tập Test ($N = 13,992$ mẫu - FinBERT Champion)
| Phân lớp Tác động | Test Precision | Test Recall | Test F1-Score | Số lượng mẫu kiểm thử ($N$) |
|:---|:---:|:---:|:---:|:---:|
| **LOW (Tác động thấp / Nhiễu)** | 98.51% | 99.37% | 98.94% | 5,737 |
| **MEDIUM (Tác động trung bình)** | 99.33% | 98.81% | 99.07% | 7,951 |
| **HIGH (Tác động mạnh / Rủi ro)** | **96.96%** | **94.41%** | **95.67%** | 304 |
| **Trung bình Macro (Macro Avg)** | **98.27%** | **97.53%** | **97.89%** | **13,992** |
| **Độ chính xác Tổng thể (Accuracy)** | \multicolumn{4}{c|}{**98.94%**} |

---

## 7. HƯỚNG DẪN CÀI ĐẶT & VẬN HÀNH (DEPLOYMENT GUIDE)

### 7.1. Yêu cầu Hệ thống Tối thiểu
- **Hệ điều hành:** Windows 10/11 x64 hoặc Linux Ubuntu 22.04 LTS.
- **Python:** Phiên bản `3.10` đến `3.13` (Khuyến nghị `Python 3.13`).
- **RAM:** Tối thiểu 8 GB (Khuyến nghị 16 GB).
- **GPU (Tùy chọn):** NVIDIA GPU hỗ trợ CUDA (GTX 1650 trở lên). Nếu không có GPU, hệ thống tự động kích hoạt chế độ tính toán trên CPU.

### 7.2. Các bước Cài đặt Môi trường
Mở PowerShell tại thư mục ứng dụng và thực thi các lệnh:

```powershell
# 1. Điều hướng tới thư mục ứng dụng
cd "D:\Năm 3\ML\Btap lớn\Web dự báo mức độ tác động tin tức"

# 2. Cài đặt các gói thư viện phụ thuộc
pip install -r requirements.txt
```

### 7.3. Khởi chạy Ứng dụng Web
Người dùng có 2 cách khởi chạy:
- **Cách 1 (Nhanh nhất):** Nhấp đúp chuột vào tệp [`run_web.bat`](file:///D:/Năm%203/ML/Btap%20lớn/Web%20dự%20báo%20mức%20độ%20tác%20động%20tin%20tức/run_web.bat).
- **Cách 2 (Qua dòng lệnh PowerShell):**
  ```powershell
  python -m streamlit run app.py --server.port 8501
  ```
- **Truy cập Giao diện Web:** Mở trình duyệt bất kỳ tại địa chỉ: [http://localhost:8501](http://localhost:8501)

---

## 8. KẾT LUẬN & CAM KẾT BẢO TRÌ

Bản tài liệu SRS này phản ánh chính xác và đầy đủ 100% cấu trúc, chức năng và quy chuẩn thẩm mỹ của Hệ thống Web AI/FinTech Dự báo Mức độ Tác động của Tin tức Tài chính đến Thị trường. Hệ thống đáp ứng đầy đủ các tiêu chuẩn học thuật khắt khe cho một đồ án tốt nghiệp / bài tập lớn nghiên cứu ứng dụng trí tuệ nhân tạo, đồng thời sở hữu giao diện thiết kế chuyên nghiệp mang bản sắc thương hiệu riêng biệt.
