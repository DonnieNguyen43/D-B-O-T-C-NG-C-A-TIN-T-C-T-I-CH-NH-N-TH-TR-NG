# BẢN ĐẶC TẢ YÊU CẦU PHẦN MỀM (SOFTWARE REQUIREMENTS SPECIFICATION - SRS)

## HỆ THỐNG WEB AI DỰ BÁO MỨC ĐỘ TÁC ĐỘNG CỦA TIN TỨC TÀI CHÍNH ĐẾN THỊ TRƯỜNG
### Financial News Market Impact Forecasting Web Application (NLP & Deep Learning)

> **Đề tài nghiên cứu:** *"Dự báo mức độ tác động của tin tức tài chính đến thị trường: Thực nghiệm so sánh giữa mô hình RNN (Bi-LSTM), BERT-base và FinBERT"*  
> **Tài liệu nguồn:** Báo cáo khoa học `BCaogiuakyML_2.docx` (Khoa Tài chính, Trường ĐH Kinh tế, ĐH Đà Nẵng)  
> **Phiên bản đặc tả:** `3.2.0-ACADEMIC STANDARDIZATION RELEASE`  
> **Ngày cập nhật:** `21/10/2026`  
> **Các mô hình đối sánh:**  
> - **Mô hình Bi-LSTM:** Mạng nơ-ron hồi quy hai chiều (`models/best_bilstm.pt` + `models/bilstm_vocab.json`)  
> - **Mô hình BERT-base:** Mô hình Transformer tổng quát (`models/bert_base_impact`) — Đạt hiệu năng tổng thể cao nhất trong thực nghiệm  
> - **Mô hình FinBERT:** Mô hình Transformer tài chính (`models/finbert_impact`)  
> **Ngôn ngữ & Công nghệ:** `Python 3.10+ / 3.13` | `Streamlit` | `PyTorch` | `HuggingFace Transformers` | `Plotly` | `Git LFS`  
> **Hệ thống Thiết kế UI/UX:** `Academic FinTech Light Theme` (Nền Kem ấm `#FAF6F0`, Thẻ Trắng `#FFFFFF`, Viền be `#E2D9CE`, Terracotta `#C96A4B`, Than nâu `#2E2620`, Vàng nghệ `#D9A441`, Xanh rêu `#7A9A8B`)  
> **Cấu trúc Điều hướng:** **3 Tabs cố định**  
> **Kho mã nguồn GitHub:** [https://github.com/DonnieNguyen43/D-B-O-T-C-NG-C-A-TIN-T-C-T-I-CH-NH-N-TH-TR-NG](https://github.com/DonnieNguyen43/D-B-O-T-C-NG-C-A-TIN-T-C-T-I-CH-NH-N-TH-TR-NG)

---

## 📌 QUY TẮC BẢO TRÌ & ĐỒNG BỘ SRS (SRS SYNCHRONIZATION MANDATE)

> [!IMPORTANT]
> **NGUYÊN TẮC BẮT BUỘC DÀNH CHO TOÀN BỘ ĐỘI NGŨ PHÁT TRIỂN & AI AGENT:**  
> 1. Bản tài liệu SRS này là **Nguồn sự thật duy nhất (Single Source of Truth - SSOT)** cho toàn bộ mã nguồn của dự án (`app.py`, `model_engine.py`, `visualizer.py`, `news_scraper.py`).
> 2. Giữ vững tính khách quan khoa học: Không sử dụng các thuật ngữ thiên vị hoặc các khuyến nghị giao dịch tài chính.
> 3. Toàn bộ nội dung tập trung thuần túy vào bài toán học máy: Phân tích xử lý ngôn ngữ tự nhiên và phân tầng mức độ tác động của thông tin tài chính (`LOW`, `MEDIUM`, `HIGH`).

---

## 1. TỔNG QUAN HỆ THỐNG & ĐỘNG LỰC NGHIÊN CỨU (SYSTEM OVERVIEW)

### 1.1. Bối cảnh & Luận cứ Khoa học Thực nghiệm (Section 5.3 Báo cáo)
Trong thị trường tài chính hiện đại, việc phân tầng chính xác mức độ tác động của tin tức (Market Impact Tiers: `LOW`, `MEDIUM`, `HIGH`) đóng vai trò quan trọng trong việc phân tích thông tin định lượng:
- **Đặc tính của mô hình Bi-LSTM:** Kiến trúc mạng hồi quy chuỗi tuần tự gặp giới hạn khi nén ngữ cảnh dài vào vector ẩn cuối cùng, dẫn đến tỷ lệ báo động giả trên phân lớp `HIGH` lên tới **62.23%** (346 ca báo động giả trên 556 lần cảnh báo trên tập Validation).
- **So sánh học thuật giữa FinBERT và BERT-base:**
  - **Mô hình FinBERT:** Được tiền huấn luyện chuyên biệt trên ngữ liệu tài chính doanh nghiệp (Financial PhraseBank, SEC filings) để tối ưu cho bài toán **Cảm xúc tài chính (Positive / Negative / Neutral)**. Tuy nhiên, *tác động thị trường không đồng nhất với cảm xúc ngôn từ*. Điển hình: Một thông cáo báo chí của ngân hàng trung ương về việc tăng lãi suất điều hành sử dụng văn phong trung tính (Neutral) nhưng lại có biên độ tác động thị trường rất lớn (HIGH Impact).
  - **Mô hình BERT-base:** Được huấn luyện trước trên kho ngữ liệu tổng quát rộng lớn (Wikipedia, BookCorpus) bao quát toàn diện các chủ đề kinh tế vĩ mô, địa chính trị, quan hệ quốc tế, ngân hàng trung ương — vốn chiếm **hơn 80% tác nhân kích hoạt nhóm HIGH**. Do đó, **BERT-base đạt hiệu năng tổng thể cao nhất trong thực nghiệm (Accuracy: 98.54%, Macro F1: 0.9589 trên tập Validation; Accuracy: 98.53%, Macro F1: 0.9625 trên tập Test độc lập ngoài mẫu)**.

### 1.2. Mục tiêu Hệ thống (System Objectives)
1. **Thiết lập Môi trường Thực nghiệm Trực quan:** Cung cấp giao diện so sánh song song 3 mô hình trên tập kiểm thử độc lập ngoài mẫu ($N = 13.992$ mẫu).
2. **Minh bạch hóa Trí tuệ Nhân tạo (Explainable AI - XAI):** Giải mã cơ chế Self-Attention tầng 11 của BERT-base (lấy trung bình qua 12 attention heads, lọc sạch dấu câu) và trực quan hóa không gian tiềm ẩn 3D PCA.
3. **Giám sát Dòng tin Real-Time & Phân tích Tác động:** Tự động thu thập luồng tin tức thời sự tài chính và đưa ra phân tích khách quan về ý nghĩa mức độ tác động của thông tin đến thị trường.

---

## 2. KIẾN TRÚC HỆ THỐNG & ĐƯỜNG ỐNG DỮ LIỆU (SYSTEM ARCHITECTURE)

### 2.1. Sơ đồ Luồng Dữ liệu Tổng thể
```
[Nguồn Dữ liệu]
  ├── Tập dữ liệu làm sạch: data/processed/ (test.csv, val.csv, train.csv)
  └── Luồng tin tức trực tiếp: RSS Feeds (Google News Financial, Yahoo Finance, CNBC Markets) + Bộ đệm 10 tin Fallback
        │
        ▼
[Đường ống Tiền xử lý NLP & Khử Trùng lặp (merge_title_and_description)]
  ├── Kiểm tra trùng lặp: Nếu Title == Desc hoặc Desc bắt đầu bằng Title -> Giữ Title duy nhất
  ├── Chuẩn hóa thực thể HTML: html.unescape()
  ├── Khử URLs & Thẻ HTML: r'https?://\S+|www\.\S+', r'<.*?>'
  ├── Chuẩn hóa khoảng trắng & Cắt chuỗi max_length = 64 tokens
        │
        ▼
[Bộ Ba Mô hình Học sâu (Parallel Inference Engine)]
  ├── Mô hình Bi-LSTM: best_bilstm.pt, vocab 20.000 từ, feat 128D
  ├── Mô hình BERT-base: bert_base_impact, 12 layers, 768D, 12 heads
  └── Mô hình FinBERT: finbert_impact, specialized financial pretraining
        │
        ▼
[Động cơ XAI & Hậu xử lý (Post-Processing & Explainability)]
  ├── Softmax Probabilities: [P_LOW, P_MEDIUM, P_HIGH]
  ├── Mức độ Đồng thuận Đa mô hình (Consensus: 3/3, 2/3, 1/3)
  ├── Bóc tách Attention Layer 11 từ BERT-base (Mean qua 12 Heads, Two-tier Defense)
  └── Chiếu Vector ẩn [CLS] 768D xuống Không gian 3D (PCA Latent Space Coordinates)
        │
        ▼
[Giao diện Web Streamlit Dashboard - 3 TABS CHÍNH DIỆN]
  ├── TAB 1: THỰC NGHIỆM DATASET ĐỘC LẬP (Testbed & Model Comparison)
  ├── TAB 2: SUY LUẬN REAL-TIME & CẢNH BÁO SỚM (Live Stream & Impact Analysis)
  └── TAB 3: TIỀN XỬ LÝ NLP & KHÔNG GIAN VECTOR 3D (XAI Pipeline & 3D Embeddings)
```

### 2.2. Thông số Kỹ thuật 3 Thế hệ Mô hình
| Thông số Kỹ thuật | Mô hình Bi-LSTM | Mô hình FinBERT | Mô hình BERT-base |
|---|---|---|---|
| **Họ kiến trúc** | Recurrent Neural Network (RNN) | Transformer Encoder (Specialized) | Transformer Encoder (General) |
| **Trọng số nền** | Khởi tạo ngẫu nhiên + Embedding | `ProsusAI/finbert` | `bert-base-uncased` (Google) |
| **Từ điển Từ vựng** | 20.000 từ (Tần số xuất hiện $\ge 2$) | 30.522 subwords | 30.522 subwords (WordPiece) |
| **Số chiều Vector ẩn** | 128 chiều ($2 \times 64$ chiều) | 768 chiều | 768 chiều |
| **Số tầng ẩn / Attention Heads** | 2 lớp Bi-LSTM xếp chồng | 12 Hidden Layers / 12 Heads | 12 Hidden Layers / 12 Heads |
| **Cơ chế XAI Trích xuất** | Nén vector ẩn cuối cùng | Phân tích phụ trợ | **Layer 11 Self-Attention Mean 12 Heads** |
| **Dung lượng tệp trọng số** | **12.35 MB** (`best_bilstm.pt`) | **417.67 MB** (`model.safetensors`) | **417.67 MB** (`model.safetensors`) |
| **Thời gian suy luận** | ~15 ms (CPU) | ~48 ms (CPU) | ~45 ms (CPU) |

---

## 3. THUẬT TOÁN TIỀN XỬ LÝ & KHỬ TRÙNG LẶP NỘI TẠI (MỤC 3.1 & 3.2 BÁO CÁO)

### 3.1. Thuật toán `merge_title_and_description(title, description)`
Toàn bộ chuỗi văn bản đầu vào cho cả môi trường thực nghiệm và suy luận thời gian thực bắt buộc phải đi qua quy trình chuẩn hóa:
```python
def merge_title_and_description(title: str, description: str = "") -> str:
    """
    Quy trình tiền xử lý chống rò rỉ dữ liệu & khử trùng lặp nội tại:
    1. Chuẩn hóa thực thể HTML bằng html.unescape()
    2. Loại bỏ liên kết URL: r'https?://\S+|www\.\S+'
    3. Loại bỏ thẻ HTML: r'<.*?>'
    4. Khử trùng lặp: Nếu title == description hoặc description bắt đầu bằng title, chỉ giữ title.
    5. Cắt chuỗi tối đa 64 tokens.
    """
```

---

## 4. ĐẶC TẢ YÊU CẦU CHỨC NĂNG (3 TABS GIAO DIỆN CHÍNH DIỆN)

### 4.1. Điều hướng Tabs Tối giản & Canh giữa (Centered Minimalist Bold Navigation)
- **Mã chức năng:** `FR-NAV-01`
- **Yêu cầu chi tiết:**
  - Thanh tab gồm đúng **3 Tabs cố định** đặt chính giữa:
    1. `🧪 Thực nghiệm Dataset độc lập`
    2. `📡 Suy luận Real-time & Cảnh báo sớm`
    3. `🪐 Tiền xử lý NLP & Không gian Vector 3D`
  - Bố cục `flex-wrap: nowrap !important; justify-content: center !important;` triệt tiêu rớt dòng.
  - Tab đang chọn: In đậm (`font-weight: 800`), màu Terracotta `#C96A4B`, thanh chỉ báo cao 3px bo tròn.

---

### 4.2. TAB 1: THỰC NGHIỆM DATASET ĐỘC LẬP (TESTBED & MODEL COMPARISON)
- **Mã chức năng:** `FR-TB-01`
- **Yêu cầu chi tiết:**
  1. **Lựa chọn Tập Dữ liệu (`FR-TB-01.1`):**
     - Mặc định: **"Tập Kiểm thử Độc lập (Test Set - 13,992 mẫu)"**.
     - Tùy chọn: **"Tập Xác thực (Validation Set - 13,992 mẫu)"** phục vụ đối soát huấn luyện.
  2. **Bộ lọc & Chọn Mẫu (`FR-TB-01.2`):**
     - Lọc theo Ground Truth (`ALL`, `LOW`, `MEDIUM`, `HIGH`).
     - Tìm kiếm từ khóa vĩ mô (*Fed, Inflation, Rate, Crisis, Plunge...*).
     - Nút **🎲 Lấy mẫu ngẫu nhiên** (Nền trắng/kem sáng thanh lịch, viền be `#D9CDC1`, chữ than nâu `#2E2620`).
     - Dropdown chọn bản tin (rút gọn tiêu đề 70 ký tự tránh tràn lề).
  3. **Thẻ Suy luận Trực diện 3 Mô hình (`FR-TB-01.3`):**
     - Cột 1: **Mô hình Bi-LSTM** (Mạng nơ-ron hồi quy hai chiều).
     - Cột 2: **Mô hình BERT-base** (Mô hình Transformer tổng quát - Hiệu năng cao nhất).
     - Cột 3: **Mô hình FinBERT** (Mô hình Transformer tài chính).
     - Hiển thị nhãn dự đoán, Độ tự tin (Confidence %), trạng thái đối chiếu Ground Truth (✅ CHÍNH XÁC / ❌ NHẦM LẪN).
     - **Thẻ hiển thị độ chuẩn xác thực nghiệm (Precision) động theo lớp dự đoán của BERT-base:**
       - Dự đoán **HIGH**: `Độ chuẩn xác thực nghiệm (HIGH Precision): 88.82% trên tập Test (HIGH Recall: 94.08% Test / 92.13% Val; Bắt đúng 286/304 sự kiện tác động mạnh ngoài mẫu)`.
       - Dự đoán **MEDIUM**: `Độ chuẩn xác thực nghiệm (MEDIUM Precision): 98.89% trên tập Test (98.52% Recall)`.
       - Dự đoán **LOW**: `Độ chuẩn xác thực nghiệm (LOW Precision): 98.57% trên tập Test (98.78% Recall)`.
     - **Thẻ Mô hình Bi-LSTM:** Hiển thị khách quan: *"Tỷ lệ báo động giả vượt 62% (dự đoán nhầm 338 mẫu MEDIUM và 8 mẫu LOW sang HIGH; Precision chỉ đạt 37.79% trên tập Val)"*.
  4. **Thanh Phân phối Xác suất Nằm ngang (`FR-TB-01.4`):**
     - Hiển thị $[P_{LOW}, P_{MEDIUM}, P_{HIGH}]$ của cả 3 mô hình với nhãn chú thích tách rời phía trên, tránh đè chữ.
  5. **Ma trận Nhầm lẫn Đồng bộ ngay trong Tab 1 (`FR-TB-01.5`):**
     - Khi chọn Tập Test: Hiển thị Ma trận nhầm lẫn kiểm định ngoài mẫu của **Mô hình BERT-base** ($N = 13.992$).
     - Khi chọn Tập Validation: Hiển thị Bộ ba Ma trận nhầm lẫn đối sánh của cả 3 mô hình (Bi-LSTM, BERT-base, FinBERT).
     - Thang màu xanh Slate/Terracotta tương phản cao đạt chuẩn WCAG AA, gán `key` duy nhất triệt tiêu `StreamlitDuplicateElementId`.

---

### 4.3. TAB 2: SUY LUẬN REAL-TIME & CẢNH BÁO SỚM (LIVE STREAM & IMPACT ANALYSIS)
- **Mã chức năng:** `FR-RT-02`
- **Yêu cầu chi tiết:**
  1. **Nguồn cấp tin & Thu thập tự động (`FR-RT-02.1`):**
     - Kết nối RSS trực tiếp từ: Google News Financial, Yahoo Finance, CNBC Markets.
     - Tự động kích hoạt bộ đệm 10 tin vĩ mô quốc tế mẫu nếu mất kết nối mạng.
     - Cung cấp nút **🔄 Thu thập tin mới nhất** và ô **✍️ Tự nhập/dán bản tin tài chính bất kỳ**.
  2. **Động cơ Phán quyết & Phân tích Ý nghĩa Tác động (`FR-RT-02.2`):**
     - Đưa văn bản qua hàm `merge_title_and_description()`.
     - Dự đoán mức độ tác động chính thức được đưa ra bởi **Mô hình BERT-base** kèm mức độ đồng thuận của 3 mô hình.
     - **Phân tích Ý nghĩa Mức độ Tác động của Thông tin:**
       - 🔴 **Phân lớp HIGH (Tác động cao):**
         - *Đặc tính thông tin:* Phản ánh các cú sốc kinh tế vĩ mô đột ngột (lãi suất ngân hàng trung ương, lạm phát tăng vọt, biến động địa chính trị, khủng hoảng thanh khoản).
         - *Tác động kỳ vọng:* Có khả năng tạo ra biên độ biến động rất lớn và lan tỏa rộng trên toàn bộ thị trường chứng khoán.
       - 🟡 **Phân lớp MEDIUM (Tác động trung bình):**
         - *Đặc tính thông tin:* Báo cáo kết quả hoạt động kinh doanh quý, biến động doanh thu, kế hoạch nhân sự hoặc hợp đồng mới của từng doanh nghiệp cụ thể.
         - *Tác động kỳ vọng:* Gây biến động cục bộ đối với từng cổ phiếu riêng lẻ hoặc một nhóm ngành nghề, không làm thay đổi xu hướng vĩ mô chung.
       - 🔵 **Phân lớp LOW (Tác động thấp):**
         - *Đặc tính thông tin:* Thông báo thủ tục quản trị doanh nghiệp thường nhật, lịch họp đại hội đồng cổ đông, công bố thông tin hành chính định kỳ.
         - *Tác động kỳ vọng:* Hầu như không tác động đến kỳ vọng lợi suất và giá cổ phiếu do thông tin đã nằm trong kế hoạch từ trước.

---

### 4.4. TAB 3: TIỀN XỬ LÝ NLP & KHÔNG GIAN VECTOR 3D (XAI PIPELINE & 3D EMBEDDINGS)
- **Mã chức năng:** `FR-PL-03`
- **Yêu cầu chi tiết:**
  1. **Trạm 1: Khử trùng lặp & Rò rỉ (Deduplication & Anti-Leakage) (`FR-PL-03.1`):**
     - Trực quan hóa đối chiếu: Chuỗi văn bản thô ban đầu vs. Chuỗi văn bản sau khi qua hàm `merge_title_and_description()`.
     - Thiết lập CSS chống tràn chữ (`word-break: break-word; white-space: pre-wrap; padding: 14px;`).
  2. **Trạm 2: Phân tách Token (Tokenization) (`FR-PL-03.2`):**
     - So sánh cách tách từ cấp từ vựng của Bi-LSTM (20.000 từ, `<pad>`, `<unk>`) và WordPiece subwords của BERT (`[CLS]`, `[SEP]`, `##`).
  3. **Trạm 3: Giải phẫu Cơ chế Chú ý Tầng 11 của Mô hình BERT-base (`FR-PL-03.3`):**
     - Trích xuất ma trận Attention từ token `[CLS]` ở tầng cuối (Layer 11): `outputs.attentions[11][0, :, 0, :]`.
     - **BẮT BUỘC lấy trung bình (MEAN) qua 12 Attention Heads**: `cls_attn.mean(dim=0).cpu().numpy()`. Tuyệt đối không dùng `torch.max()`.
     - **Lọc sạch Dấu câu & Special Tokens khỏi Bảng Top Từ Khóa:** Loại bỏ toàn bộ `[CLS]`, `[SEP]`, `[PAD]`, `,`, `.`, `;`, `:`, `!`, `?`, `-`, `/`, URL rác.
     - Gom nhóm các subwords (`crash` + `##es` -> `crashes`).
     - Bảng xếp hạng chỉ hiển thị các từ vựng tài chính thực tế mang tính bước ngoặt (*crashes, tensions, crude, fed, inflation, rate, plunge...*).
     - Chuẩn hóa Min-Max scaling điểm chú ý trên các từ vựng thực để màu sắc Heatmap nổi bật đúng các từ khóa rủi ro vĩ mô.
  4. **Trạm 4: Hoạt ảnh Không gian Vector Tiềm ẩn 3D Đa Mô hình (Multi-Model 3D Latent PCA Space) (`FR-PL-03.4`):**
     - Hỗ trợ bộ chọn chuyển đổi linh hoạt quan sát không gian tiềm ẩn giữa cả 3 mô hình học máy:
       - **Mô hình Bi-LSTM (Vector ẩn 128 chiều):** Vector nối 2 chiều forward & backward `[hn[-2], hn[-1]]`. Độ phân tán nội cụm lớn hơn ($\sigma \approx 0.42$), ranh giới LOW và MEDIUM gần nhau, phản ánh tính tuần tự của mạng RNN.
       - **Mô hình FinBERT (Vector ẩn 768 chiều):** Trích xuất từ token `[CLS]` của Transformer tài chính. Các cụm gom tụ chặt chẽ theo thuật ngữ tài chính ($\sigma \approx 0.28$).
       - **Mô hình BERT-base (Vector ẩn 768 chiều):** Trích xuất từ token `[CLS]` sau khi fine-tuning toàn diện trên 139,919 mẫu. Đạt khoảng cách liên cụm cực đại ($\sigma \approx 0.22$), phân tách tối ưu và sắc nét nhất giữa 3 phân lớp LOW - MEDIUM - HIGH.
     - Dựng biểu đồ `plotly.graph_objects.Scatter3d` thể hiện 3 đám mây điểm ngữ nghĩa: `LOW` (Xanh thép `#7A9A8B`), `MEDIUM` (Vàng nghệ `#D9A441`), `HIGH` (Đỏ Terracotta `#C96A4B`).
     - Chiếu tọa độ vector thực tế của bản tin đang phân tích thành **viên kim cương nổi bật** (`symbol='diamond'`) với đường nét đứt kết nối trực tiếp đến tâm cụm dự đoán.
     - Hiển thị bảng luận cứ đối sánh phân tích độ hội tụ của từng mô hình theo trục $PC_1, PC_2, PC_3$.

---

## 5. BẢNG SỐ LIỆU ĐÁNH GIÁ THỰC NGHIỆM CHUẨN HÓA (BENCHMARK DATA MATRIX)

Dưới đây là các số liệu thực nghiệm chính thức từ báo cáo khoa học `BCaogiuakyML_2.docx`:

### 5.1. Bảng Đối sánh Tập Validation ($N = 13.992$ mẫu)
| Chỉ số Thực nghiệm | Mô hình Bi-LSTM | Mô hình FinBERT | Mô hình BERT-base | Đánh giá Chuyên môn |
|---|:---:|:---:|:---:|---|
| **Accuracy (Độ chính xác)** | 93.17% | 98.41% | **98.54%** | BERT-base đạt độ chính xác tổng thể cao nhất |
| **Macro Precision** | 76.44% | 94.99% | **95.29%** | BERT-base dẫn đầu độ chuẩn xác trung bình |
| **Macro Recall** | 90.89% | 96.24% | **96.52%** | Khả năng bao quát toàn diện các phân lớp |
| **Macro F1-Score** | 80.59% | 95.60% | **95.89%** | Cân bằng tối ưu giữa Precision và Recall |
| **HIGH Recall (Bao phủ tác động mạnh)** | 84.26% | 91.48% | **92.13%** | Bắt đúng 281/305 sự kiện tác động mạnh |
| **Báo động giả nhóm HIGH** | 346 ca / 556 | 39 ca / 318 | **37 ca / 318** | BERT-base giảm mạnh nhất báo động giả |
| **Báo động giả từ LOW sang HIGH** | 8 ca | 1 ca | **0 ca (Tuyệt đối)** | BERT-base lọc sạch 100% không báo động nhầm tin LOW |

### 5.2. Ma trận Nhầm lẫn Chi tiết Tập Validation ($N = 13.992$ mẫu)
- **Mô hình Bi-LSTM:**
  - Thực tế LOW (5,737): Dự đoán LOW = 5,706 | MEDIUM = 23 | HIGH = 8
  - Thực tế MEDIUM (7,950): Dự đoán LOW = 458 | MEDIUM = 7,154 | HIGH = 338
  - Thực tế HIGH (305): Dự đoán LOW = 4 | MEDIUM = 44 | HIGH = 257
- **Mô hình FinBERT:**
  - Thực tế LOW (5,737): Dự đoán LOW = 5,700 | MEDIUM = 36 | HIGH = 1
  - Thực tế MEDIUM (7,950): Dự đoán LOW = 64 | MEDIUM = 7,848 | HIGH = 38
  - Thực tế HIGH (305): Dự đoán LOW = 0 | MEDIUM = 26 | HIGH = 279
- **Mô hình BERT-base:**
  - Thực tế LOW (5,737): Dự đoán LOW = 5,698 | MEDIUM = 39 | HIGH = **0**
  - Thực tế MEDIUM (7,950): Dự đoán LOW = 58 | MEDIUM = 7,855 | HIGH = **37**
  - Thực tế HIGH (305): Dự đoán LOW = 0 | MEDIUM = 24 | HIGH = **281**

### 5.3. Bảng Kiểm định Độc lập Ngoài Mẫu của Mô hình BERT-base trên Tập Test ($N = 13.992$ mẫu)
- **Độ chính xác Tổng thể (Accuracy):** **98.53%** (13,786 / 13,992 mẫu chính xác)
- **Chỉ số Trung bình Macro:** Macro Precision = **95.43%**, Macro Recall = **97.12%**, Macro F1-Score = **96.25%**

| Phân lớp Tác động | Test Precision | Test Recall | Test F1-Score | Số lượng Mẫu ($N$) | Đánh giá Chuyên môn |
|---|:---:|:---:|:---:|:---:|---|
| **LOW (Tác động thấp / Nhiễu)** | 98.57% | 98.78% | 98.68% | 5,737 | Nhận diện chính xác thông tin hành chính |
| **MEDIUM (Tác động trung bình)** | 98.89% | 98.52% | 98.70% | 7,951 | Dự báo chuẩn xác thông tin doanh nghiệp |
| **HIGH (Tác động cao / Rủi ro)** | **88.82%** | **94.08%** | **91.37%** | 304 | Bắt trúng 286/304 biến cố lớn ngoài mẫu |

---

## 6. ĐẶC TẢ YÊU CẦU PHI CHỨC NĂNG & THẨM MỸ (NFR)

### 6.1. Thẩm mỹ Học thuật Nền Kem Ấm (Academic FinTech Light Theme)
- **Bảng màu:**
  - Nền chính: Kem ấm `#FAF6F0`. Thẻ Card: Trắng `#FFFFFF` viền be `#E2D9CE`.
  - Phông chữ: `Playfair Display` (Tiêu đề), `Be Vietnam Pro` (Nội dung), `JetBrains Mono` (Số liệu).
  - Không có chữ trắng trên nền trắng. Mọi widget nhập liệu đều có chữ than nâu `#2E2620`.
  - Nút bấm `.stButton > button`: Nền trắng ngà `#FFFFFF`, chữ nâu `#2E2620`, viền `#D9CDC1`, hiệu ứng hover nền kem `#F3EAE0`, viền cam gạch `#C96A4B`.
- **Độc nhất Khóa Element (Key Enforcement):** Mọi biểu đồ `st.plotly_chart` đều có tham số `key="..."` riêng biệt.

---

## 7. HƯỚNG DẪN CÀI ĐẶT & KHỞI CHẠY CỤC BỘ (LOCAL SETUP)

```bash
# 1. Di chuyển vào thư mục cục bộ của dự án
cd "D:\Năm 3\ML\Btap lớn\Web dự báo mức độ tác động tin tức"

# 2. Khởi chạy ứng dụng Streamlit trên cổng mặc định 8501
streamlit run app.py
```
