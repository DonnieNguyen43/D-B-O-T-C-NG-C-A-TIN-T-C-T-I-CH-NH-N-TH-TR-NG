# HỆ THỐNG DỰ BÁO MỨC ĐỘ TÁC ĐỘNG CỦA TIN TỨC TÀI CHÍNH ĐẾN THỊ TRƯỜNG
### Financial News Market Impact Forecasting Web Application (NLP & Deep Learning)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-FinBERT%20%26%20BERT-yellow.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-informational.svg)](https://git-lfs.github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Hệ thống Dashboard phân tích và dự báo mức độ tác động của tin tức tài chính đến thị trường chứng khoán dựa trên thực nghiệm đối sánh 3 kiến trúc mô hình học máy: **Mô hình Bi-LSTM**, **Mô hình BERT-base** và **Mô hình FinBERT**.

---

## 🎯 TỔNG QUAN HỆ THỐNG & TÍNH NĂNG CHÍNH

Giao diện ứng dụng được phát triển trên nền tảng **Streamlit** với ngôn ngữ thiết kế Academic FinTech hiện đại, bảng màu trung tính ấm áp (Terracotta #C96A4B, Giấy kem #FAF6F0, Chữ than nâu #2E2620), cấu trúc thành 3 phân hệ trực quan:

### 1. Phán đoán & So sánh 3 Mô hình
- **Đánh giá đa mô hình:** Cho phép nhập tin tức tài chính tùy biến hoặc chọn từ danh sách mẫu kịch bản thị trường để phân loại mức độ tác động thành 3 cấp độ: LOW, MEDIUM, HIGH.
- **Trực quan hóa Phân phối Xác suất:** Biểu đồ Stacked Bar Chart hiển thị tỷ lệ tự tin phân bổ cho 3 nhãn.
- **Ma trận Nhầm lẫn Đồng bộ:** Hiển thị ma trận kiểm thử ngoài mẫu trên tập Test độc lập (N = 13,992 mẫu) và bộ ba ma trận tập Validation.

### 2. Bắt tin thị trường thời gian thực & Phân tích Ý nghĩa Tác động
- **Tự động cào dữ liệu RSS:** Kết nối trực tiếp đến các kênh tài chính uy tín quốc tế (Google News Financial, Yahoo Finance, CNBC Markets).
- **Phán đoán tức thời:** Dự báo nhanh mức độ tác động của từng bản tin vừa phát sinh và giải thích ý nghĩa đặc tính thông tin cùng tác động kỳ vọng đến thị trường.

### 3. Tiền xử lý NLP & Không gian 3D
- **Bộ tiền xử lý ngôn ngữ tự nhiên:** Minh họa tường minh quy trình làm sạch văn bản và thuật toán khử trùng lặp nội tại `merge_title_and_description()`.
- **Giải phẫu Cơ chế Chú ý Tầng 11:** Trích xuất ma trận Attention từ token `[CLS]` (lấy trung bình qua 12 attention heads và lọc sạch dấu câu).
- **Không gian Tiềm ẩn 3D (3D Latent Embedding Space):** Trực quan hóa vector ẩn 768 chiều được giảm chiều qua PCA xuống không gian 3 chiều với các cụm phân lớp tác động LOW, MEDIUM, HIGH.

---

## 🏗 CẤU TRÚC THƯ MỤC DỰ ÁN

```text
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
│   └── finbert_impact/           # Checkpoint mô hình FinBERT (Git LFS)
│       ├── config.json
│       ├── tokenizer.json
│       └── model.safetensors
├── app.py                        # Giao diện chính Streamlit Dashboard
├── model_engine.py               # Module nạp mô hình, trích xuất Attention & suy luận
├── visualizer.py                 # Module dựng đồ thị Plotly & thẻ chỉ số
├── news_scraper.py               # Module cào tin tức thị trường tài chính trực tiếp
├── requirements.txt              # Danh sách thư viện Python phụ thuộc
├── run_web.bat                   # Kịch bản khởi chạy nhanh hệ thống trên Windows
├── SRS.md                        # Tài liệu đặc tả yêu cầu phần mềm chi tiết
└── README.md                     # Tài liệu tổng quan dự án
```

---

## ⚡ HƯỚNG DẪN CÀI ĐẶT & KHỞI CHẠY

### 1. Yêu cầu hệ thống
- **Hệ điều hành:** Windows 10/11, macOS, hoặc Linux
- **Python:** Phiên bản 3.10 trở lên
- **Git & Git LFS:** Đã cài đặt [Git Large File Storage](https://git-lfs.com/) để tải đầy đủ trọng số .safetensors.

### 2. Sao chép kho mã nguồn (Clone Repository)
```bash
# Cài đặt Git LFS trước khi clone
git lfs install

# Clone repository
git clone https://github.com/DonnieNguyen43/D-B-O-T-C-NG-C-A-TIN-T-C-T-I-CH-NH-N-TH-TR-NG.git
cd D-B-O-T-C-NG-C-A-TIN-T-C-T-I-CH-NH-N-TH-TR-NG

# Kéo đầy đủ các file trọng số lớn
git lfs pull
```

### 3. Cài đặt các thư viện phụ thuộc
```bash
python -m venv venv
venv\Scripts\activate       # Trên Windows
# source venv/bin/activate  # Trên macOS/Linux

pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Khởi chạy ứng dụng
- **Cách 1 (Windows - Khởi chạy nhanh):** Nháy đúp chuột vào file `run_web.bat`.
- **Cách 2 (Dòng lệnh):**
```bash
streamlit run app.py
```
Sau khi khởi chạy, truy cập trình duyệt tại địa chỉ: `http://localhost:8501`.

---

## 📊 KẾT QUẢ THỰC NGHIỆM ĐỐI SÁNH (TẬP VALIDATION N = 13,992 MẪU)

| Mô hình | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | HIGH Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mô hình Bi-LSTM** | 93.17% | 76.44% | 90.89% | 80.59% | 84.26% |
| **Mô hình FinBERT** | 98.41% | 94.99% | 96.24% | 95.60% | 91.48% |
| **Mô hình BERT-base** | **98.54%** | **95.29%** | **96.52%** | **95.89%** | **92.13%** |

*Mô hình BERT-base đạt hiệu năng tổng thể và độ bao phủ rủi ro vĩ mô cao nhất trong thực nghiệm nhờ được huấn luyện trên kho ngữ liệu tổng quát rộng lớn bao quát toàn diện các chủ đề kinh tế vĩ mô, ngân hàng trung ương và quan hệ quốc tế.*

---

## 👥 THÀNH VIÊN THỰC HIỆN
- **Nhóm nghiên cứu:** Nhóm 9 (Đề tài Machine Learning Tài chính)
- **Khoa:** Khoa Tài chính, Trường Đại học Kinh tế – Đại học Đà Nẵng
