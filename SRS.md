# BẢN ĐẶC TẢ YÊU CẦU PHẦN MỀM (SOFTWARE REQUIREMENTS SPECIFICATION - SRS)

## HỆ THỐNG WEB AI/FINTECH DỰ BÁO MỨC ĐỘ TÁC ĐỘNG CỦA TIN TỨC TÀI CHÍNH ĐẾN THỊ TRƯỜNG CHỨNG KHOÁN
### Financial News Market Impact Forecasting Web Application (NLP & Deep Learning)

> **Đề tài nghiên cứu:** *"Dự báo mức độ tác động của tin tức tài chính đến thị trường: Thực nghiệm so sánh giữa mô hình RNN (Bi-LSTM), BERT-base và FinBERT"*  
> **Phiên bản tài liệu:** `3.0.0-ACADEMIC FINTECH RELEASE`  
> **Ngày cập nhật:** `20/10/2026`  
> **Ngôn ngữ & Công nghệ:** `Python 3.10+ / 3.13` | `Streamlit` | `PyTorch` | `HuggingFace Transformers` | `Plotly` | `Git LFS`  
> **Hệ thống Thiết kế UI/UX:** `Academic FinTech Design System` (Bảng màu Kem ấm `#FAF6F0`, Terracotta `#C96A4B`, Than nâu `#2E2620`, Olive `#606C38`)  
> **Môi trường hoạt động:** Windows x64 (Local, On-Premise CPU/GPU & Cloud Deployment)  
> **Kho mã nguồn GitHub:** [https://github.com/DonnieNguyen43/D-B-O-T-C-NG-C-A-TIN-T-C-T-I-CH-NH-N-TH-TR-NG](https://github.com/DonnieNguyen43/D-B-O-T-C-NG-C-A-TIN-T-C-T-I-CH-NH-N-TH-TR-NG)

---

## 📌 QUY TẮC BẢO TRÌ & ĐỒNG BỘ SRS (SRS SYNCHRONIZATION MANDATE)

> [!IMPORTANT]
> **NGUYÊN TẮC BẮT BUỘC DÀNH CHO TOÀN BỘ ĐỘI NGŨ PHÁT TRIỂN & AI AGENT:**  
> 1. Mọi quyết định sửa đổi mã nguồn (từ kiến trúc, thuật toán, luồng dữ liệu, UI/UX đến từng thẻ widget Streamlit) đều **bắt buộc phải căn cứ trực tiếp trên Bản đặc tả SRS này**.
> 2. Mỗi khi có bất kỳ sự thay đổi hoặc tinh chỉnh nào trong mã nguồn (`app.py`, `model_engine.py`, `visualizer.py`, `news_scraper.py`), **phải đồng thời cập nhật tức thì vào tài liệu SRS.md** để đảm bảo tài liệu luôn là nguồn sự thật duy nhất (Single Source of Truth - SSOT).

---

## 1. TỔNG QUAN HỆ THỐNG (SYSTEM OVERVIEW)

### 1.1. Bối cảnh Đề tài & Vấn đề Thực tế (Problem Statement & Motivation)
Trong thị trường tài chính hiện đại, thông tin bất đối xứng và tốc độ truyền tin là yếu tố cốt lõi định hình hành vi dòng tiền và sự biến động của giá cổ phiếu. Hàng triệu bản tin vĩ mô, báo cáo tài chính doanh nghiệp và diễn biến địa chính trị xuất hiện mỗi ngày tạo nên một "biển nhiễu" khổng lồ.
- **Thách thức lớn nhất đối với giới đầu tư & nhà quản trị rủi ro:** Phân loại chính xác **mức độ biến động** mà một bản tin có thể gây ra cho chỉ số thị trường (S&P 500 / Sensex), tách biệt các rung chấn mạnh mẽ (**HIGH**) với biến động bình thường (**MEDIUM**) hoặc tin tức nhiễu không gây tác động (**LOW**).
- **Hạn chế của mô hình truyền thống:** Các mạng nơ-ron hồi quy cổ điển như **Bi-LSTM** thường bị nghẽn cổ chai ngữ cảnh dài, bỏ sót mối liên kết liên câu và nhầm lẫn các thuật ngữ tài chính chuyên sâu.
- **Giải pháp Transformer & FinBERT:** Tận dụng cơ chế Multi-Head Self-Attention của các mô hình ngôn ngữ lớn (LLMs). Đặc biệt, mô hình **FinBERT** (được tiền huấn luyện trên hàng tỷ token tài chính từ Financial PhraseBank, Reuters, SEC filings) mang lại khả năng phân loại vượt trội và triệt tiêu tối đa các ca báo động giả.

### 1.2. Mục tiêu Hệ thống (System Objectives)
1. **Xây dựng Nền tảng Đánh giá Thực nghiệm Trực quan:** Cung cấp môi trường so sánh đa chiều giữa 3 thế hệ mô hình: **Bi-LSTM** (Baseline), **BERT-base** (General NLP) và **FinBERT** (Domain Champion).
2. **Minh bạch hóa Trí tuệ Nhân tạo (Explainable AI - XAI):** Bóc tách lớp Self-Attention Layer 11 của Transformer và trực quan hóa không gian vector tiềm ẩn 3D (3D Latent Embedding Space) để giải thích cơ chế ra quyết định của mô hình.
3. **Cảnh báo Sớm Dòng tin Thời gian Thực (Real-Time Inference):** Tự động thu thập các bản tin tài chính trực tiếp từ các luồng RSS uy tín quốc tế, dự báo nhanh mức độ tác động và hỗ trợ các nhà phân tích đưa ra phản ứng kịp thời.
4. **Chuẩn mực Hóa Giao diện Học thuật FinTech:** Thiết kế giao diện theo phong cách học thuật hiện đại, sang trọng, màu sắc trung tính ấm áp, đạt chuẩn dễ đọc WCAG AA, không dùng từ ngữ cảm tính hay biểu tượng không phù hợp.

### 1.3. Đối tượng Sử dụng (Target Audiences)
- **Hội đồng Khoa học & Giảng viên Hướng dẫn:** Kiểm tra tính toàn vẹn của kết quả nghiên cứu, đối chiếu số liệu thực nghiệm trên tập Test/Validation và đánh giá mã nguồn.
- **Chuyên viên Phân tích Định lượng (Quant Analysts):** Trực quan hóa phân phối xác suất, độ tự tin và ma trận nhầm lẫn để kiểm định khả năng phòng hộ rủi ro.
- **Nhà Quản trị Danh mục (Portfolio Managers):** Sử dụng phân hệ thời gian thực để phân loại xung lực tin tức vĩ mô nhằm bảo vệ danh mục đầu tư.

---

## 2. KIẾN TRÚC HỆ THỐNG & ĐƯỜNG ỐNG DỮ LIỆU (SYSTEM ARCHITECTURE)

### 2.1. Sơ đồ Luồng Dữ liệu Tổng thể
```
[Nguồn Dữ liệu]
  ├── Tập dữ liệu làm sạch: data/processed/ (train.csv, val.csv, test.csv, stock_news_clean.csv)
  └── Luồng tin tức trực tiếp: RSS Feeds (Yahoo Finance, Reuters, MarketWatch, CNBC)
        │
        ▼
[Tiền xử lý NLP Chống rò rỉ (Anti-Leakage Pipeline)]
  ├── Lowercase ➔ Bỏ URLs / HTML Tags ➔ Chuẩn hóa Ký tự tiền tệ & Số liệu
  ├── Bi-LSTM Tokenizer (Từ điển 20.000 từ, độ dài cố định 64 tokens, Pad 0)
  └── WordPiece Tokenizer (FinBERT / BERT-base, 64 tokens, subword ##, CLS/SEP)
        │
        ▼
[Bộ Ba Mô hình Học sâu (Parallel Inference Engine)]
  ├── Mô hình 1: Bi-LSTM (2 lớp ẩn 128 chiều, Dropout 0.3, Linear Classifier)
  ├── Mô hình 2: BERT-base-uncased (12 lớp, 768 chiều, 12 attention heads)
  └── Mô hình 3: FinBERT Champion (Fine-tuned Sequence Classification, 3 Classes)
        │
        ▼
[Động cơ XAI & Hậu xử lý (Post-Processing & Explainability)]
  ├── Trích xuất Softmax Probabilities: [P_LOW, P_MEDIUM, P_HIGH]
  ├── Tính toán Delta Gap & Đồng hồ đo Độ tự tin (Confidence Gauge)
  ├── Bóc tách Ma trận Chú ý Lớp cuối (Layer 11 Multi-Head Attention Weights)
  └── Giảm chiều Vector ẩn 768D ➔ Tọa độ 3D Không gian Tiềm ẩn (PCA)
        │
        ▼
[Giao diện Trực quan hóa Streamlit Dashboard]
  ├── Tab 1: Thực nghiệm Dataset độc lập (Testbed & Model Comparison)
  ├── Tab 2: Suy luận Real-time & Cảnh báo sớm (Live RSS & Instant AI Verdict)
  ├── Tab 3: Tiền xử lý NLP & Không gian 3D (Text Cleaning, Tokens & 3D PCA Space)
  └── Tab 4: Đối sánh Benchmark & Báo cáo kỹ thuật (Scientific Tables & Metrics)
```

### 2.2. Chi tiết 3 Thế hệ Mô hình (Model Specifications)
| Thông số Kỹ thuật | Bi-LSTM (Baseline) | BERT-base-uncased | FinBERT (Champion) |
|---|---|---|---|
| **Họ kiến trúc** | Recurrent Neural Network (RNN) | Transformer Encoder (General) | Transformer Encoder (Financial Specialized) |
| **Trọng số nền** | Khởi tạo ngẫu nhiên + Embedding | `bert-base-uncased` (Google) | `ProsusAI/finbert` (Hugging Face) |
| **Kích thước Từ vựng** | 20.000 từ (Tần số xuất hiện $\ge 2$) | 30.522 subwords | 30.522 subwords (Financial domain tuned) |
| **Số chiều Vector ẩn ($d_{model}$)** | 128 chiều ($2 	imes 64$ chiều 2 hướng) | 768 chiều | 768 chiều |
| **Số tầng kiến trúc** | 2 lớp Bi-LSTM xếp chồng | 12 Hidden Layers | 12 Hidden Layers |
| **Cơ chế Chú ý (Attention)** | Không có | 12 Heads / Layer | 12 Heads / Layer (Layer 11 XAI trích xuất) |
| **Dung lượng tệp trọng số** | **12.35 MB** (`best_bilstm.pt`) | **417.67 MB** (`model.safetensors` qua Git LFS) | **417.67 MB** (`model.safetensors` qua Git LFS) |
| **Thời gian suy luận (CPU/GPU)**| **~15 ms** | ~45 ms | ~48 ms |

---

## 3. ĐẶC TẢ YÊU CẦU CHỨC NĂNG (FUNCTIONAL REQUIREMENTS)

### 3.1. Phân hệ Đầu trang: Hero Section & Thống kê Tổng quan
- **Mã chức năng:** `FR-HERO-01`
- **Yêu cầu chi tiết:**
  1. **Logo Định chế Kỹ thuật số (Vector SVG):** Vẽ trực tiếp bằng SVG mã nguồn sạch, chứa biểu tượng biểu đồ tài chính, nốt nơ-ron học máy và hiệu ứng viền sáng nhẹ, kích thước `56x56 px`.
  2. **Tiêu đề Nghiên cứu Chuẩn mực:** Tiêu đề cấp 1 `HỆ THỐNG DỰ BÁO MỨC ĐỘ TÁC ĐỘNG CỦA TIN TỨC TÀI CHÍNH ĐẾN THỊ TRƯỜNG CHỨNG KHOÁN`.
  3. **Huy hiệu Thống kê Học thuật (Institutional KPI Badges):**
     - `2.169 Phiên giao dịch S&P 500` (Giai đoạn 2016 – 2026).
     - `139.919 Bản tin tài chính vĩ mô` đã được làm sạch chuẩn Anti-Leakage.
     - `3 Kiến trúc mô hình chuyên sâu`: Bi-LSTM · BERT-base · FinBERT (Champion).

### 3.2. Phân hệ Thanh Điều hướng Tabs: Minimalist Centered Bold Navigation
- **Mã chức năng:** `FR-NAV-02`
- **Yêu cầu chi tiết:**
  1. **Bố cục Canh giữa (Centered Alignment):** Toàn bộ thanh tab được căn giữa tuyệt đối theo chiều ngang trang (`justify-content: center !important; margin: 10px auto 26px auto; width: 100%`).
  2. **Phong cách Chọn In Đậm (Bold Active Tab - Chuẩn Ảnh 1):**
     - Tab đang được chọn (Active Tab): Hiển thị chữ **in đậm** (`font-weight: 800 !important; color: #2E2620 !important;`).
     - Tab chưa chọn (Inactive Tabs): Hiển thị chữ thường (`font-weight: 500 !important; color: #554E48 !important;`). Hiệu ứng hover chuyển màu sang `#2E2620`.
  3. **Triệt tiêu Viền Gạch chân Màu cam & Nền Hộp thô cứng:**
     - Ẩn hoàn toàn `div[data-baseweb="tab-highlight"]` và `div[data-baseweb="tab-border"]`.
     - Nền tab trong suốt (`background: transparent !important; border: none !important; box-shadow: none !important;`).
     - Đường kẻ phân tách trên và dưới mỏng nhẹ thanh lịch (`border-top: 1px solid rgba(217, 205, 193, 0.5); border-bottom: 1px solid rgba(217, 205, 193, 0.5)`).
  4. **Quy chuẩn Tên Tab (Title Case, Không dùng số La Mã):**
     - Tab 1: `Thực nghiệm Dataset độc lập`
     - Tab 2: `Suy luận Real-time & Cảnh báo sớm`
     - Tab 3: `Tiền xử lý NLP & Không gian 3D`
     - Tab 4: `Đối sánh Benchmark & Báo cáo kỹ thuật`

### 3.3. PHÂN HỆ 1: THỰC NGHIỆM DATASET ĐỘC LẬP (TAB 1)
- **Mã chức năng:** `FR-TB-01`
- **Yêu cầu chi tiết:**
  1. **Chuyển đổi Tập dữ liệu (`FR-TB-01.1`):** Lựa chọn giữa tập Kiểm thử độc lập (`test.csv`) hoặc tập Kiểm định (`val.csv`).
  2. **Bộ lọc Nhãn Ground Truth (`FR-TB-01.2`):** Lọc theo 4 tiêu chí: `ALL`, `LOW (0)`, `MEDIUM (1)`, `HIGH (2)`.
  3. **Tìm kiếm Từ khóa Vĩ mô (`FR-TB-01.3`):** Lọc nhanh bản tin theo các từ khóa như *Fed, Inflation, Rate, Earnings, Bank, Crisis...*
  4. **Bộ chọn Mẫu & Lấy mẫu Ngẫu nhiên (`FR-TB-01.4`):** Nút bốc mẫu ngẫu nhiên và dropdown chọn bản tin (giới hạn 80 ký tự tránh tràn lề).
  5. **Thẻ So sánh Trực diện 3 Mô hình (`FR-TB-01.5`):**
     - Hiển thị nhãn dự đoán (`LOW`, `MEDIUM`, `HIGH`) và độ tự tin của Bi-LSTM, BERT-base, FinBERT.
     - So khớp tức thì với nhãn Ground Truth (`CHÍNH XÁC` hoặc `NHẦM LẪN`).
     - Đo lường khoảng cách chênh lệch xác suất (**Delta Gap**).
     - **Thẻ Champion Model (FinBERT):** Tự động áp dụng logic hiển thị động độ chuẩn xác thực nghiệm (Precision) tương ứng với nhãn dự đoán (lấy chuẩn từ Bảng 4 báo cáo):
       * Nếu dự đoán **HIGH**: `Độ chuẩn xác thực nghiệm (HIGH Precision): 96.96% trên tập Test (94.46% trên tập Val)`.
       * Nếu dự đoán **MEDIUM**: `Độ chuẩn xác thực nghiệm (MEDIUM Precision): 99.33% trên tập Test (99.17% trên tập Val)`.
       * Nếu dự đoán **LOW**: `Độ chuẩn xác thực nghiệm (LOW Precision): 98.51% trên tập Test (98.80% trên tập Val)`.
     - **Thẻ Bi-LSTM Baseline:** Hiển thị rõ ràng ngữ cảnh học thuật: *"Tạo ra 343 ca báo động giả trên 588 lần cảnh báo biến động mạnh (tập Validation N = 13.992 mẫu, HIGH Precision chỉ đạt 41.67%)"*.
  6. **Đồng hồ Đo Áp lực Tác động (Confidence Gauge) (`FR-TB-01.6`):**
     - Đồng hồ hình bán nguyệt trực quan hóa mức độ tự tin phân loại của FinBERT với các dải màu học thuật chuẩn mực.
  7. **Biểu đồ Phân phối Xác suất Nằm ngang (Stacked Bar Chart) (`FR-TB-01.7`):**
     - Hiển thị tỷ lệ `[P_LOW, P_MEDIUM, P_HIGH]` tích lũy 100% của 3 mô hình.
     - **Tách rời chú thích (Decoupled HTML Legend Badges):** Ẩn chú thích mặc định của Plotly để tránh lỗi va chạm/cắt cụt nhãn chữ; thay bằng các huy hiệu chú giải HTML cố định phía trên biểu đồ.
  8. **Ma trận Trọng số Chú ý (Self-Attention Layer 11) (`FR-TB-01.8`):**
     - Trích xuất Attention Weights từ FinBERT để làm nổi bật các từ khóa tài chính mang tính quyết định.
  9. **Hiển thị Ma trận Nhầm lẫn Đồng bộ (`FR-TB-01.9`):**
     - Tự động hiển thị Ma trận nhầm lẫn Test độc lập khi xem tập Test, hoặc bộ 3 Ma trận nhầm lẫn khi xem tập Validation.
  10. **Tối ưu Bố cục Không gian 3D:** Đã dời hoàn toàn biểu đồ 3D sang Tab 3 để tránh hiển thị trùng lặp, giải tỏa mật độ thị giác cho Tab 1.

### 3.4. PHÂN HỆ 2: SUY LUẬN REAL-TIME & CẢNH BÁO SỚM (TAB 2)
- **Mã chức năng:** `FR-RT-02`
- **Mục tiêu:** Quy trình tinh gọn, tập trung vào việc lấy tin tức thời sự thị trường trực tiếp và phán đoán nhanh mức độ tác động.
- **Yêu cầu chi tiết:**
  1. **Bộ Cào Tin Tự động RSS Đa kênh (`FR-RT-02.1`):**
     - Kết nối trực tiếp đến các nguồn tin tài chính quốc tế uy tín (Yahoo Finance, Reuters, MarketWatch, Investing.com).
     - Thu thập các bản tin vừa phát sinh kèm thời gian xuất bản.
  2. **Kho Lưu trữ Dự phòng (High-Fidelity Fallback) (`FR-RT-02.2`):**
     - Tự động kích hoạt cơ sở dữ liệu tin tức mẫu chuẩn hóa nếu mất kết nối mạng hoặc bị giới hạn bởi tường lửa RSS.
  3. **Bộ Chọn & Duyệt Tin Trực quan (`FR-RT-02.3`):**
     - Giao diện cho phép chọn các tin tức vừa cào về để xem nội dung và chuyển đổi linh hoạt.
     - Cho phép nhập tin tức tùy biến hoặc sử dụng 3 kịch bản thị trường mẫu điển hình (*Fed tăng lãi suất, Doanh nghiệp đi ngang, Đại hội cổ đông*).
  4. **Phán đoán Nhanh Mức độ Tác động (`FR-RT-02.4`):**
     - Đưa bản tin qua bộ ba mô hình để đưa ra dự báo tác động tức thời.
     - Hiển thị rõ ràng cấp độ tác động, phân phối xác suất và đồng hồ tự tin mà không bị rườm rà.

### 3.5. PHÂN HỆ 3: TIỀN XỬ LÝ NLP & KHÔNG GIAN 3D (TAB 3)
- **Mã chức năng:** `FR-PL-03`
- **Mục tiêu:** Trình diễn minh bạch quy trình tiền xử lý ngôn ngữ tự nhiên và trực quan hóa không gian biểu diễn ẩn đa chiều.
- **Yêu cầu chi tiết:**
  1. **Đường ống Tiền xử lý Văn bản Chuẩn hóa (`FR-PL-03.1`):**
     - Trình bày trực diện 6 bước làm sạch: Viết thường (Lowercasing), Gỡ bỏ liên kết URL & thẻ HTML, Chuẩn hóa ký hiệu tài chính, Xóa khoảng trắng thừa, Tokenization.
     - Đối chiếu văn bản gốc và văn bản sau làm sạch trong các khung viền chống tràn chữ.
  2. **So sánh Phân tách Từ vựng & Tokenization (`FR-PL-03.2`):**
     - Phân tách Word-level Tokenizer (Bi-LSTM, từ điển 20.000 từ, padding 0).
     - Phân tách WordPiece Tokenizer (FinBERT, subword `##`, `[CLS]`, `[SEP]`).
  3. **Bản đồ Chú ý Self-Attention Tầng 11 & Khắc phục Lỗi Attention Sink (`FR-PL-03.3`):**
     - **Làm sạch văn bản đầu vào trước khi trích xuất:** Cho văn bản chạy qua hàm `clean_news_text()` để loại bỏ hoàn toàn link URL và thẻ HTML, tuân thủ nghiêm ngặt công thức (1) trong Báo cáo: $X = clean(title) + ". " + clean(description)$.
     - **Trích xuất ma trận Attention chuẩn:** Trích xuất ma trận ở Layer 11 từ token `[CLS]`: `cls_attn = outputs.attentions[11][0, :, 0, :]` (kích thước `[num_heads=12, seq_len]`).
     - **BẮT BUỘC tính trung bình cộng (MEAN) qua 12 heads:** `mean_attn = cls_attn.mean(dim=0).cpu().numpy()`. Tuyệt đối không dùng `torch.max()`.
     - **Gom nhóm Subwords (WordPiece Aggregation):** Gộp các từ con bị phân tách (`crash` + `##es` -> `crashes`) và lấy trung bình trọng số.
     - **Lọc sạch Dấu câu, Special Tokens & Rò rỉ Web khỏi Bảng Xếp hạng:**
       * Thiết lập tập loại trừ: `exclude_tokens = {'[cls]', '[sep]', '[pad]', ',', '.', ';', ':', '!', '?', '-', '--', '/', '\', 'http', 'https', 'reuters', 'com', 'www', 'rt'}`.
       * Bảng "Top Từ Khóa Tập Trung Chú Ý" chỉ giữ lại các token thuộc bảng chữ cái (`any(c.isalpha())` và không nằm trong `exclude_tokens`).
     - **Chuẩn hóa điểm số Min-Max Scaling:** Thực hiện Min-Max scaling trên các từ vựng thực tế để đưa dải điểm về trực quan `[0.0 - 1.0]`, giúp các từ vựng tác động mạnh (*crashes, tensions, crude, sensex*) được tô màu cam/đỏ nổi bật, còn dấu câu giữ điểm thấp (0.05) và màu xám nhạt tự nhiên trên Heatmap.
         * **Cơ chế Phòng vệ 2 Lớp (Two-tier Defense):** Áp dụng đồng bộ cả ở backend pipeline (`model_engine.predict_text`) và frontend rendering (`visualizer.render_attention_heatmap_html`), đảm bảo tuyệt đối không có hiện tượng dấu câu/ký tự đặc biệt hiển thị màu cam/đỏ hay vượt ngưỡng 0.05 trên khối Heatmap trực quan.
  4. **Không gian Tiềm ẩn 3D Tương phản Cao (3D Latent Space Visualization) (`FR-PL-03.4`):**
     - Vector ẩn 768 chiều từ FinBERT được giảm chiều qua PCA xuống không gian 3 chiều ($X, Y, Z$).
     - Hiển thị 3 cụm tác động: `LOW` (Olive Green `#606C38`), `MEDIUM` (Vàng nghệ `#D9A441`), `HIGH` (Terracotta `#C96A4B`).
     - Bản tin hiện tại được chiếu thành điểm hình kim cương nổi bật (`symbol='diamond'`).
     - **Độ tương phản lưới tọa độ cải tiến (Contrast Gridlines):** Các đường lưới không gian 3D được thiết lập màu xanh đá đậm (`gridcolor='#7A8B99'`, `linecolor='#334155'`) đảm bảo hiển thị sắc nét trên nền sáng kem, khắc phục hoàn toàn hiện tượng đường kẻ mờ nhạt khó nhìn.
     - Hỗ trợ xoay 360 độ, thu phóng và điều khiển camera linh hoạt.

### 3.6. PHÂN HỆ 4: ĐỐI SÁNH BENCHMARK & BÁO CÁO KỸ THUẬT (TAB 4)
- **Mã chức năng:** `FR-BM-04`
- **Mục tiêu:** Cung cấp báo cáo số liệu thực chứng khoa học phục vụ bảo vệ đề tài.
- **Yêu cầu chi tiết:**
  1. **Bảng Đối sánh Toàn diện trên Tập Validation ($N = 13.992$ mẫu) (`FR-BM-04.1`):**
     - Đánh giá đầy đủ: Accuracy, Macro Precision, Macro Recall, Macro F1-Score, HIGH Precision, HIGH Recall, Số ca báo động giả.
  2. **Biểu đồ Cột Hiệu năng Đa chiều (`FR-BM-04.2`):**
     - Biểu đồ Plotly đối sánh trực quan khoảng cách hiệu năng giữa Bi-LSTM, BERT-base và FinBERT.
  3. **Bộ Ba Ma trận Nhầm lẫn Validation (`FR-BM-04.3`):**
     - 3 ma trận nhầm lẫn cạnh nhau đạt chuẩn tương phản chữ WCAG AA (tự động đảo màu chữ đen/trắng theo độ đậm nhạt của ô).
  4. **Báo cáo Kiểm định Độc lập trên Tập Test ($N = 13.992$ mẫu) (`FR-BM-04.4`):**
     - Bảng chỉ số chi tiết cho từng lớp của FinBERT: Accuracy 98.94%, Macro F1 97.89%, HIGH Precision 96.96%, HIGH Recall 94.41%.
  5. **Ý nghĩa Quản trị Rủi ro Đầu tư (`FR-BM-04.5`):**
     - Phân tích chi phí kinh tế lượng của các ca báo động giả và luận cứ chọn FinBERT làm mô hình Champion.

---

## 4. ĐẶC TẢ YÊU CẦU PHI CHỨC NĂNG (NON-FUNCTIONAL REQUIREMENTS - NFR)

### 4.1. Hiệu năng & Tốc độ Suy luận (Performance & Latency)
- **Thời gian suy luận (Inference Latency):** Suy luận song song cả 3 mô hình cho 1 bản tin đạt **dưới 50ms trên GPU** và **dưới 250ms trên CPU**.
- **Cơ chế Caching Tối ưu:**
  - Sử dụng `@st.cache_resource` để nạp 3 mô hình vào VRAM đúng 1 lần duy nhất khi khởi động máy chủ web.
  - Sử dụng `@st.cache_data` để nạp tập dữ liệu kiểm thử vào RAM trong dưới 0.3 giây.
- **Tiêu thụ Bộ nhớ:** Dưới 2.5 GB RAM hệ thống và dưới 2.0 GB VRAM GPU.

### 4.2. Thẩm mỹ & Trải nghiệm Người dùng (UI/UX Standards)
- **Hệ màu Trung tính Ấm áp (Academic FinTech Palette):**
  - Nền chính: Kem ấm `#FAF6F0` (60%)
  - Nền phụ & Thẻ Card: Trắng ngà `#FFFFFF` và Be nhạt `#F3EAE0` (30%)
  - Màu điểm nhấn: Terracotta `#C96A4B`, Nâu than `#2E2620`, Olive nhạt `#606C38`, Vàng nghệ `#D9A441` (10%)
- **Typography & Font chữ:**
  - Tiêu đề H1/H2: `Playfair Display`, serif sang trọng.
  - Nội dung: `Be Vietnam Pro`, sans-serif, `line-height: 1.6` tạo nhịp điệu đọc thoáng đãng.
  - Số liệu định lượng: `JetBrains Mono` / `Segoe UI Mono`.
- **Cân bằng Chiều cao Khối thẻ (Flexbox Stretch):**
  - Mọi thẻ chỉ số thống kê (`.academic-stat-card`, `.kpi-card`) trong cùng một hàng đều được tự động co giãn bằng chiều cao thông qua Streamlit flex column rules (`display: flex; flex-direction: column; justify-content: space-between; height: 100%`).
- **Triệt tiêu Hoàn toàn Lỗi "Chữ Trắng Nền Trắng" (White-on-White Bug):**
  - Toàn bộ các widget Streamlit (`st.selectbox`, `st.text_input`, `st.radio`, `st.slider`) đều được can thiệp CSS override ép màu chữ luôn là nâu đen than `#2E2620` trên nền sáng `#FFFFFF` hoặc `#FAF6F0`.

### 4.3. Tiêu chuẩn Tiếp cận & Độ tương phản (Accessibility & Contrast)
- **Đạt chuẩn WCAG AA:** Toàn bộ văn bản hiển thị có tỷ lệ tương phản $\ge 4.5:1$ so với màu nền.
- **Tự động Đảo Màu Chữ Ma trận:** Giá trị nhỏ nền sáng $\le 42\%$ dùng chữ `#2E2620`; giá trị lớn nền đậm $> 42\%$ dùng chữ `#FFFFFF`.
- **Đường Lưới Không gian 3D Tương phản Cao:** Trục tọa độ và lưới trong đồ thị 3D sử dụng màu xám xanh đậm (`#7A8B99` và `#334155`), loại bỏ triệt để hiện tượng đường kẻ trắng chìm vào nền kem.

### 4.4. Tính Ổn định Kỹ thuật (Technical Reliability)
- **Độc nhất Khóa Phần tử (Unique Key Enforcement):** Mọi biểu đồ Plotly và widget tương tác đều có tham số `key="..."` với tiền tố phân hệ rõ ràng (ví dụ: `prob_bars_tab1`, `gauge_tab1`, `cm_tab1_test`), ngăn chặn 100% lỗi `StreamlitDuplicateElementId`.
- **Chống Tràn Chữ (Text Overflow Prevention):** Mọi vùng hiển thị văn bản đều có thuộc tính CSS `word-break: break-word; white-space: pre-wrap; overflow-x: auto;`.

### 4.5. Cơ chế Nạp Đường dẫn Cục bộ Ưu tiên (Local-First Path Resolution)
- Hàm `get_source_dir()` trong `model_engine.py` luôn ưu tiên kiểm tra thư mục `models/` và `data/` nằm ngay trong thư mục ứng dụng hiện tại (`os.path.dirname(__file__)`).
- Nếu không tìm thấy, hệ thống mới kích hoạt đường dẫn dự phòng đến thư mục nguồn ban đầu. Cơ chế này đảm bảo khi repository được clone về bất kỳ máy tính nào cũng có thể chạy độc lập ngay lập tức mà không bị lỗi đường dẫn cố định.

---

## 5. CẤU TRÚC THƯ MỤC & CÁC TỆP MÃ NGUỒN (CODEBASE ARCHITECTURE)

Kho mã nguồn ứng dụng web được tổ chức tự chứa (self-contained) như sau:

```text
Web dự báo mức độ tác động tin tức/
│
├── .gitattributes          # Cấu hình Git LFS theo dõi 2 file trọng số model.safetensors
├── .gitignore              # Bỏ qua __pycache__, bytecode, môi trường ảo và tệp tạm
├── .streamlit/
│   └── config.toml         # Cấu hình chủ đề giao diện Streamlit (Light FinTech Theme)
│
├── data/                   # Dữ liệu phân tách phục vụ thực nghiệm & kiểm thử
│   ├── processed/          # Tập train.csv, val.csv, test.csv, stock_news_processed_clean.csv
│   └── raw/                # Tập dữ liệu gốc 139.919 bản tin (stock_news_2016 to 2026.csv)
│
├── models/                 # Trọng số và từ điển mô hình
│   ├── bilstm_vocab.json   # Từ điển từ vựng 20.000 từ của mô hình Bi-LSTM
│   ├── best_bilstm.pt      # Trọng số PyTorch Bi-LSTM đã huấn luyện (12.35 MB)
│   ├── finbert_benchmark_results.json # Kết quả thực nghiệm và ma trận nhầm lẫn
│   ├── bert_base_impact/   # Mô hình BERT-base đã fine-tune (model.safetensors qua Git LFS)
│   └── finbert_impact/     # Mô hình FinBERT Champion đã fine-tune (model.safetensors qua Git LFS)
│
├── app.py                  # Điểm khởi chạy chính Streamlit; CSS inject; 4 Tab điều hướng
├── model_engine.py         # Nạp mô hình, tiền xử lý văn bản, suy luận song song, XAI & 3D PCA
├── visualizer.py           # Dựng biểu đồ Plotly (Stacked Bar, Confidence Gauge, Heatmap, 3D PCA)
├── news_scraper.py         # Cào tin tức tài chính qua RSS (Yahoo Finance, Reuters...) & Fallback
│
├── requirements.txt        # Danh mục thư viện Python (PyTorch, Transformers, Streamlit, Plotly...)
├── run_web.bat             # Kịch bản Batch 1-Click khởi chạy nhanh trên Windows
├── SKILL.md                # Tài liệu hướng dẫn phong cách thiết kế giao diện
├── README.md               # Tài liệu hướng dẫn cài đặt và tổng quan dự án GitHub
└── SRS.md                  # Bản Đặc tả Yêu cầu Phần mềm Toàn diện (Tài liệu này)
```

---

## 6. MA TRẬN ÁNH XẠ CHỈ SỐ THỰC NGHIỆM ĐÃ TÍCH HỢP (BENCHMARK DATA MATRIX)

Dưới đây là các chỉ số thực nghiệm chính xác được nhúng trong mã nguồn hệ thống để đối chiếu với báo cáo học thuật:

### 6.1. Bảng Đối sánh Tập Validation ($N = 13.992$ mẫu)
| Tiêu chí Thực nghiệm | Bi-LSTM (Baseline) | BERT-base | FinBERT (Champion) | Đánh giá Chuyên môn |
|---|:---:|:---:|:---:|---|
| **Accuracy (Độ chính xác)** | 93.66% | **98.95%** | 98.92% | BERT-base và FinBERT áp đảo hoàn toàn |
| **Macro Precision** | 77.68% | 96.89% | **97.48%** | FinBERT dẫn đầu độ chuẩn xác trung bình |
| **Macro Recall** | 89.92% | **97.33%** | 95.95% | BERT-base bắt diện rộng tốt nhất |
| **Macro F1-Score** | 81.65% | **97.11%** | 96.69% | Transformer vượt trội Bi-LSTM +15.5% F1 |
| **HIGH Recall (Bắt rủi ro)** | 80.33% | **93.77%** | 89.51% | BERT-base và FinBERT đều bao phủ rủi ro cao |
| **HIGH Precision (Chuẩn xác rủi ro)** | 41.67% | 92.56% | **94.46%** | FinBERT áp đảo (94.46% Val / 96.96% Test) |
| **Báo động giả HIGH (False Alarms)** | 343 ca / 588 | 23 ca / 309 | **16 ca / 289** | FinBERT giảm 95.3% ca báo động giả so với Bi-LSTM |
| **Báo động giả từ LOW sang HIGH** | 85 ca | 0 ca | **0 ca (Tuyệt đối)** | FinBERT lọc nhiễu 100% không báo động nhầm |

### 6.2. Bảng Kiểm định Độc lập Tập Test ($N = 13.992$ mẫu - FinBERT Champion)
| Phân lớp Tác động | Test Precision | Test Recall | Test F1-Score | Số lượng mẫu kiểm thử ($N$) |
|:---|:---:|:---:|:---:|:---:|
| **LOW (Tác động thấp / Nhiễu)** | 98.51% | 99.37% | 98.94% | 5.737 |
| **MEDIUM (Tác động trung bình)** | 99.33% | 98.81% | 99.07% | 7.951 |
| **HIGH (Tác động mạnh / Rủi ro)** | **96.96%** | **94.41%** | **95.67%** | 304 |
| **Trung bình Macro (Macro Avg)** | **98.27%** | **97.53%** | **97.89%** | **13.992** |
| **Độ chính xác Tổng thể (Accuracy)** | \multicolumn{4}{c|}{**98.94%**} |

---

## 7. HƯỚNG DẪN CÀI ĐẶT & VẬN HÀNH (DEPLOYMENT GUIDE)

### 7.1. Yêu cầu Hệ thống
- **Hệ điều hành:** Windows 10/11 x64, macOS hoặc Linux Ubuntu 22.04 LTS.
- **Python:** Phiên bản `3.10` trở lên (Khuyến nghị `Python 3.11` – `3.13`).
- **Git & Git LFS:** Bắt buộc cài đặt [Git LFS](https://git-lfs.com/) để tải các tệp trọng số mô hình lớn (`.safetensors`).
- **RAM:** Tối thiểu 8 GB (Khuyến nghị 16 GB).
- **GPU (Tùy chọn):** NVIDIA GPU hỗ trợ CUDA (GTX 1650 trở lên). Nếu không có GPU, hệ thống tự động chạy trên CPU.

### 7.2. Các bước Cài đặt Môi trường
```bash
# 1. Cài đặt Git LFS
git lfs install

# 2. Clone repository từ GitHub
git clone https://github.com/DonnieNguyen43/D-B-O-T-C-NG-C-A-TIN-T-C-T-I-CH-NH-N-TH-TR-NG.git
cd D-B-O-T-C-NG-C-A-TIN-T-C-T-I-CH-NH-N-TH-TR-NG

# 3. Kéo toàn bộ trọng số mô hình lớn qua Git LFS
git lfs pull

# 4. Tạo và kích hoạt môi trường ảo (Khuyến nghị)
python -m venv venv
venv\Scripts\activate  # Trên Windows
# source venv/bin/activate  # Trên Linux/macOS

# 5. Cài đặt các gói thư viện phụ thuộc
pip install --upgrade pip
pip install -r requirements.txt
```

### 7.3. Khởi chạy Ứng dụng Web
- **Cách 1 (Nhanh nhất trên Windows):** Nhấp đúp chuột vào tệp [`run_web.bat`](file:///D:/Năm%203/ML/Btap%20lớn/Web%20dự%20báo%20mức%20độ%20tác%20động%20tin%20tức/run_web.bat).
- **Cách 2 (Qua dòng lệnh):**
  ```bash
  streamlit run app.py --server.port 8501
  ```
- **Truy cập Giao diện Web:** Mở trình duyệt bất kỳ tại địa chỉ: [http://localhost:8501](http://localhost:8501)

---

## 8. KẾT LUẬN & CAM KẾT BẢO TRÌ

Bản tài liệu SRS phiên bản `3.0.0-ACADEMIC FINTECH RELEASE` này phản ánh chính xác và đầy đủ 100% cấu trúc, chức năng, hệ thống thẩm mỹ và toàn bộ các tinh chỉnh kỹ thuật mới nhất của Hệ thống Web AI/FinTech Dự báo Mức độ Tác động của Tin tức Tài chính đến Thị trường.

Mọi đóng góp, nâng cấp hoặc bảo trì trong tương lai **bắt buộc phải đối chiếu và cập nhật song song với bản SRS này** để bảo đảm tính kỷ luật công nghệ và tính nghiêm cẩn học thuật của đề tài.
