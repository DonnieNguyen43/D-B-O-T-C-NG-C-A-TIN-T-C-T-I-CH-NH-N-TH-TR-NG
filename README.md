# HỆ THỐNG DỰ BÁO MỨC ĐỘ TÁC ĐỘNG CỦA TIN TỨC TÀI CHÍNH ĐẾN THỊ TRƯỜNG
### Financial News Market Impact Forecasting Web Application (NLP & Deep Learning)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-FinBERT%20%26%20BERT-yellow.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-informational.svg)](https://git-lfs.github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Hệ thống Dashboard phân tích định lượng và dự báo mức độ biến động/tác động của tin tức tài chính đến thị trường chứng khoán (S&P 500) dựa trên 3 kiến trúc mô hình học sâu & xử lý ngôn ngữ tự nhiên: **Bi-LSTM**, **BERT-base** và **FinBERT (Champion)**.

---

## 🎯 TỔNG QUAN HỆ THỐNG & TÍNH NĂNG CHÍNH

Giao diện ứng dụng được phát triển trên nền tảng **Streamlit** với ngôn ngữ thiết kế Academic FinTech hiện đại, bảng màu trung tính ấm áp (Terracotta #C96A4B, Giấy kem #FAF6F0, Chữ than nâu #2E2620), cấu trúc thành 3 phân hệ trực quan:

### 1. Phán đoán & So sánh 3 Mô hình
- **Đánh giá đa mô hình:** Cho phép nhập tin tức tài chính tùy biến hoặc chọn từ danh sách mẫu kịch bản thị trường để phân loại mức độ tác động thành 3 cấp độ: LOW, MEDIUM, HIGH.
- **Trực quan hóa Phân phối Xác suất:** Biểu đồ Stacked Bar Chart hiển thị tỷ lệ tự tin phân bổ cho 3 nhãn.
- **Biểu đồ Đo lường Độ Tự tin (Confidence Gauge):** Đồng hồ đo áp lực tác động tương đối chuẩn hóa.
- **Bản đồ Trọng số Chú ý (Self-Attention Heatmap):** Bóc tách lớp Multi-Head Attention cuối cùng của FinBERT để giải thích từ khóa tài chính mang tính quyết định.

### 2. Bắt tin thị trường thời gian thực
- **Tự động cào dữ liệu RSS:** Kết nối trực tiếp đến các kênh tài chính uy tín quốc tế (Yahoo Finance, Reuters, MarketWatch, Investing.com...).
- **Phán đoán tức thời:** Dự báo nhanh mức độ tác động của từng bản tin vừa phát sinh nhằm hỗ trợ quyết định giao dịch theo dòng thời sự.

### 3. Tiền xử lý NLP & Không gian 3D
- **Bộ tiền xử lý ngôn ngữ tự nhiên:** Minh họa tường minh 6 công đoạn làm sạch văn bản (Lowercasing, Regex URL/HTML removal, Symbol normalization, Lemmatization/Stemming, Tokenization).
- **Không gian Tiềm ẩn 3D (3D Latent Embedding Space):** Trực quan hóa vector ẩn 768 chiều được giảm chiều qua PCA xuống không gian 3 chiều với các cụm phân lớp tác động LOW, MEDIUM, HIGH.

---

## 🏗 CẤU TRÚC THƯ MỤC DỰ ÁN

`	ext
Web dự báo mức độ tác động tin tức/
├── .streamlit/
│   └── config.toml               # Cấu hình giao diện Streamlit (Theme Light FinTech)
├── data/
│   └── processed/                # Dữ liệu phân tách train/val/test và clean dataset
│       ├── test.csv
│       ├── val.csv
│       ├── train.csv
│       ├── stock_news_processed_clean.csv
│       └── stock_news_processed_full.csv
├── models/
│   ├── bilstm_vocab.json         # Từ điển từ vựng của mô hình Bi-LSTM
│   ├── best_bilstm.pt            # Trọng số mô hình Bi-LSTM đã huấn luyện
│   ├── finbert_benchmark_results.json # Kết quả thực nghiệm và ma trận nhầm lẫn
│   ├── bert_base_impact/         # Checkpoint mô hình BERT-base (Git LFS)
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── model.safetensors
│   └── finbert_impact/           # Checkpoint mô hình FinBERT Champion (Git LFS)
│       ├── config.json
│       ├── tokenizer.json
│       └── model.safetensors
├── app.py                        # Giao diện chính Streamlit Dashboard
├── model_engine.py               # Module nạp mô hình, trích xuất Attention & suy luận
├── visualizer.py                 # Module dựng đồ thị Plotly & thẻ chỉ số KPI
├── news_scraper.py               # Module cào tin tức thị trường tài chính trực tiếp
├── requirements.txt              # Danh sách thư viện Python phụ thuộc
├── run_web.bat                   # Kịch bản khởi chạy nhanh hệ thống trên Windows
├── SRS.md                        # Tài liệu đặc tả yêu cầu phần mềm chi tiết
└── README.md                     # Tài liệu tổng quan dự án
`

---

## ⚡ HƯỚNG DẪN CÀI ĐẶT & KHỞI CHẠY

### 1. Yêu cầu hệ thống
- **Hệ điều hành:** Windows 10/11, macOS, hoặc Linux
- **Python:** Phiên bản 3.10 trở lên
- **Git & Git LFS:** Đã cài đặt [Git Large File Storage](https://git-lfs.com/) để tải đầy đủ trọng số .safetensors.

### 2. Sao chép kho mã nguồn (Clone Repository)
`ash
# Cài đặt Git LFS trước khi clone
git lfs install

# Clone repository
git clone https://github.com/DonnieNguyen43/D-B-O-T-C-NG-C-A-TIN-T-C-T-I-CH-NH-N-TH-TR-NG.git
cd D-B-O-T-C-NG-C-A-TIN-T-C-T-I-CH-NH-N-TH-TR-NG

# Kéo đầy đủ các file trọng số lớn
git lfs pull
`

### 3. Cài đặt các thư viện phụ thuộc
Khuyến nghị tạo môi trường ảo Python (env hoặc conda):
`ash
python -m venv venv
venv\Scripts\activate       # Trên Windows
# source venv/bin/activate  # Trên macOS/Linux

pip install --upgrade pip
pip install -r requirements.txt
`

### 4. Khởi chạy ứng dụng
- **Cách 1 (Windows - Khởi chạy nhanh):** Nháy đúp chuột vào file un_web.bat.
- **Cách 2 (Dòng lệnh):**
`ash
streamlit run app.py
`
Sau khi khởi chạy, truy cập trình duyệt tại địa chỉ: http://localhost:8501.

---

## 📊 KẾT QUẢ THỰC NGHIỆM ĐỐI SÁNH (BENCHMARK)

| Mô hình | Macro F1-Score | Độ chính xác (Accuracy) | Thời gian suy luận (Inference Latency) |
| :--- | :---: | :---: | :---: |
| **Bi-LSTM** (Baseline) | 0.442 | 48.7% | **~15 ms** |
| **BERT-base** | 0.589 | 61.2% | ~45 ms |
| **FinBERT** (Champion) | **0.674** | **68.9%** | ~48 ms |

*FinBERT đạt hiệu năng vượt trội nhờ tiền huấn luyện trên kho văn bản tài chính chuyên sâu (Financial Corpora), nắm bắt ngữ nghĩa và độ biến động thị trường chính xác nhất.*

---

## 👥 THÀNH VIÊN THỰC HIỆN
- **Tác giả:** DonnieNguyen43 (Nhóm nghiên cứu Machine Learning Tài chính)
- **Khoa:** Tài chính - Ngân hàng, Trường Đại học Kinh tế – Đại học Đà Nẵng
