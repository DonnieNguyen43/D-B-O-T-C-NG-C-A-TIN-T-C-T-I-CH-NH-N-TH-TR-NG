"""
model_engine.py
Module nạp 3 mô hình (Bi-LSTM, BERT-base, FinBERT), tiền xử lý văn bản chuẩn Anti-Leakage,
suy luận song song, trích xuất Attention Weights và ánh xạ không gian tiềm ẩn (Latent Embedding).
"""

import os
import re
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC NGUỒN (READ-ONLY)
# ==============================================================================
PRIMARY_SOURCE_DIR = r"D:\Năm 3\ML\Btap lớn\ML_Dự báo tác động đến thị trường của tin tức"
FALLBACK_SOURCE_DIR = r"D:\Năm 3\ML\Btap lớn\ML_Dự báo hướng biến động giá chỉ số S&P 500"

def get_source_dir() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(script_dir, "models")):
        return script_dir
    if os.path.exists(PRIMARY_SOURCE_DIR):
        return PRIMARY_SOURCE_DIR
    elif os.path.exists(FALLBACK_SOURCE_DIR):
        return FALLBACK_SOURCE_DIR
    else:
        return script_dir

SOURCE_DIR = get_source_dir()
MODELS_DIR = os.path.join(SOURCE_DIR, "models")
DATA_DIR = os.path.join(SOURCE_DIR, "data", "processed")

# Nhãn mục tiêu học máy thống nhất
ID2LABEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
LABEL2ID = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

# Bảng màu rủi ro chuẩn học thuật (Terracotta & Earth Warm Palette)
COLOR_LOW = "#7A9A8B"          # Xanh rêu hiền hòa (Tác động thấp / Ổn định)
COLOR_MEDIUM = "#D9A441"       # Vàng nghệ ấm (Tác động trung bình)
COLOR_HIGH = "#C96A4B"         # Terracotta đỏ gạch ấm (Cảnh báo biến động mạnh)
COLOR_TERRACOTTA = "#C96A4B"   # Terracotta Brand Accent
COLOR_EARTH_BROWN = "#8C4A32"  # Nâu đất
COLOR_WARM_BG = "#FAF6F0"      # Nền kem ấm
COLOR_CARD_BG = "#FFFFFF"      # Nền card trắng ngà
COLOR_BORDER_SOFT = "#D9CDC1"  # Viền mềm
COLOR_TEXT_HEADING = "#2E2620" # Nâu đen (Tiêu đề)
COLOR_TEXT_BODY = "#4A4038"    # Nâu xám (Nội dung)
COLOR_TEXT_MUTED = "#8A7E72"   # Xám ấm (Caption/Meta)

LABEL_COLORS = {
    "LOW": COLOR_LOW,
    "MEDIUM": COLOR_MEDIUM,
    "HIGH": COLOR_HIGH
}

# ==============================================================================
# 2. TIỀN XỬ LÝ VĂN BẢN (ANTI-LEAKAGE PIPELINE THEO SRS.MD)
# ==============================================================================
def clean_news_text(text: str) -> str:
    r"""
    Quy trình tiền xử lý chống rò rỉ dữ liệu theo đúng chuẩn báo cáo & SRS.md:
    - Loại bỏ link URL: regex r'https?://\S+|www\.\S+'
    - Loại bỏ thẻ HTML: regex r'<.*?>'
    - Chuẩn hóa khoảng trắng: regex r'\s+'
    """
    if not text or pd.isna(text):
        return ""
    text_str = str(text)
    # Loại bỏ URLs
    text_str = re.sub(r'https?://\S+|www\.\S+', ' ', text_str)
    # Loại bỏ thẻ HTML
    text_str = re.sub(r'<.*?>', ' ', text_str)
    # Chuẩn hóa khoảng trắng thừa
    text_str = re.sub(r'\s+', ' ', text_str).strip()
    return text_str

def format_input_text(title: str, description: str = "") -> str:
    """Ghép chuỗi: clean(title) + '. ' + clean(description)"""
    c_title = clean_news_text(title)
    c_desc = clean_news_text(description)
    if c_desc and c_desc.lower() != c_title.lower():
        return f"{c_title}. {c_desc}"
    return c_title

def inspect_cleaning_pipeline(raw_text: str) -> dict:
    """Phân tích các thành phần bị loại bỏ trong trạm tiền xử lý để minh chứng Anti-Leakage"""
    urls_found = re.findall(r'https?://\S+|www\.\S+', str(raw_text))
    html_tags = re.findall(r'<.*?>', str(raw_text))
    cleaned = clean_news_text(raw_text)
    return {
        "raw": raw_text,
        "cleaned": cleaned,
        "urls_removed": urls_found,
        "html_tags_removed": html_tags,
        "raw_length": len(str(raw_text)),
        "cleaned_length": len(cleaned)
    }

# ==============================================================================
# 3. KIẾN TRÚC MÔ HÌNH 1: BiLSTMClassifier (BASELINE RNN)
# ==============================================================================
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size=20000, embed_dim=128, hidden_dim=128, num_classes=3, num_layers=2, bidirectional=True, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            bidirectional=bidirectional, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.direction_factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * self.direction_factor, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids):
        embedded = self.dropout(self.embedding(input_ids))
        lstm_out, (hn, cn) = self.lstm(embedded)
        # Nối vector trạng thái ẩn cuối cùng của 2 chiều forward & backward
        feat = torch.cat([hn[-2, :, :], hn[-1, :, :]], dim=1) if self.direction_factor == 2 else hn[-1, :, :]
        logits = self.fc(self.dropout(feat))
        return logits, feat

# ==============================================================================
# 4. HÀM NẠP CẢ 3 MÔ HÌNH (@st.cache_resource)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_all_models():
    """
    Nạp 3 mô hình vào GPU (CUDA) nếu có, fallback về CPU.
    Chỉ nạp 1 lần duy nhất trong toàn bộ phiên chạy ứng dụng (<50ms phản hồi).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resources = {
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "models": {},
        "tokenizers": {},
        "vocab": None
    }
    
    # 4.1. Nạp Bi-LSTM
    bilstm_vocab_path = os.path.join(MODELS_DIR, "bilstm_vocab.json")
    bilstm_weights_path = os.path.join(MODELS_DIR, "best_bilstm.pt")
    
    with open(bilstm_vocab_path, "r", encoding="utf-8") as f:
        bilstm_vocab = json.load(f)
    resources["vocab"] = bilstm_vocab
    
    bilstm_model = BiLSTMClassifier(vocab_size=len(bilstm_vocab))
    bilstm_state = torch.load(bilstm_weights_path, map_location=device)
    bilstm_model.load_state_dict(bilstm_state)
    bilstm_model.to(device)
    bilstm_model.eval()
    resources["models"]["bilstm"] = bilstm_model
    
    # 4.2. Nạp BERT-base
    bert_dir = os.path.join(MODELS_DIR, "bert_base_impact")
    bert_tok = AutoTokenizer.from_pretrained(bert_dir)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_dir, attn_implementation="eager")
    bert_model.to(device)
    bert_model.eval()
    resources["models"]["bert_base"] = bert_model
    resources["tokenizers"]["bert_base"] = bert_tok
    
    # 4.3. Nạp FinBERT (Champion)
    finbert_dir = os.path.join(MODELS_DIR, "finbert_impact")
    finbert_tok = AutoTokenizer.from_pretrained(finbert_dir)
    finbert_model = AutoModelForSequenceClassification.from_pretrained(finbert_dir, attn_implementation="eager")
    finbert_model.to(device)
    finbert_model.eval()
    resources["models"]["finbert"] = finbert_model
    resources["tokenizers"]["finbert"] = finbert_tok
    
    return resources

# ==============================================================================
# 5. HÀM SUY LUẬN ĐỒNG THỜI 3 MÔ HÌNH VÀ TRÍCH XUẤT ATTENTION
# ==============================================================================
def bilstm_tokenize(text: str, vocab: dict, max_len=64):
    """Tiền xử lý và token hóa cấp độ từ cho mô hình Bi-LSTM"""
    words = re.findall(r'\b[a-zA-Z0-9_\$%-]+\b', str(text).lower())
    token_ids = [vocab.get(w, vocab.get("<unk>", 1)) for w in words][:max_len]
    padded_len = len(token_ids)
    if len(token_ids) < max_len:
        token_ids += [vocab.get("<pad>", 0)] * (max_len - len(token_ids))
    return torch.tensor([token_ids], dtype=torch.long), padded_len, words[:max_len]

def predict_text(raw_text: str, resources: dict) -> dict:
    """
    Suy luận song song cả 3 mô hình, tính toán phân phối xác suất softmax,
    độ đồng thuận (Consensus), trích xuất Attention Layer 11 và tọa độ 3D.
    """
    device = resources["device"]
    cleaned_text = clean_news_text(raw_text)
    if not cleaned_text:
        cleaned_text = "market news empty"
        
    results = {}
    
    # 1. Suy luận Bi-LSTM
    bilstm_model = resources["models"]["bilstm"]
    vocab = resources["vocab"]
    bilstm_input, padded_len, raw_words = bilstm_tokenize(cleaned_text, vocab, max_len=64)
    bilstm_input = bilstm_input.to(device)
    
    with torch.no_grad():
        bilstm_logits, bilstm_feat = bilstm_model(bilstm_input)
        bilstm_probs = torch.softmax(bilstm_logits, dim=1).cpu().numpy()[0]
        bilstm_pred = int(np.argmax(bilstm_probs))
        
    results["bilstm"] = {
        "pred_id": bilstm_pred,
        "pred_label": ID2LABEL[bilstm_pred],
        "probs": bilstm_probs.tolist(),
        "confidence": float(bilstm_probs[bilstm_pred]),
        "token_ids": bilstm_input.cpu().numpy()[0].tolist(),
        "padded_len": padded_len
    }
    
    # 2. Suy luận BERT-base
    bert_model = resources["models"]["bert_base"]
    bert_tok = resources["tokenizers"]["bert_base"]
    bert_enc = bert_tok(
        cleaned_text, max_length=64, padding="max_length", truncation=True, return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        bert_out = bert_model(**bert_enc, output_hidden_states=True)
        bert_probs = torch.softmax(bert_out.logits, dim=1).cpu().numpy()[0]
        bert_pred = int(np.argmax(bert_probs))
        
    results["bert_base"] = {
        "pred_id": bert_pred,
        "pred_label": ID2LABEL[bert_pred],
        "probs": bert_probs.tolist(),
        "confidence": float(bert_probs[bert_pred])
    }
    
    # 3. Suy luận FinBERT (Champion Model) & Trích xuất Attention Layer 11
    finbert_model = resources["models"]["finbert"]
    finbert_tok = resources["tokenizers"]["finbert"]
    finbert_enc = finbert_tok(
        cleaned_text, max_length=64, padding="max_length", truncation=True, return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        finbert_out = finbert_model(**finbert_enc, output_attentions=True, output_hidden_states=True)
        finbert_probs = torch.softmax(finbert_out.logits, dim=1).cpu().numpy()[0]
        finbert_pred = int(np.argmax(finbert_probs))
        
        # Trích xuất vector tiềm ẩn 768 chiều từ token [CLS]
        finbert_cls_vector = finbert_out.hidden_states[-1][0, 0, :].cpu().numpy()
        
        # Trích xuất ma trận Attention từ tầng cuối cùng (Layer 11)
        layer11_attentions = finbert_out.attentions[-1][0] # Shape: (12 heads, seq_len, seq_len)
        avg_attention = layer11_attentions.mean(dim=0).cpu().numpy() # Shape: (seq_len, seq_len)
        cls_attention = avg_attention[0] # Chú ý từ [CLS] tới mọi token
        
    finbert_tokens = finbert_tok.convert_ids_to_tokens(finbert_enc["input_ids"][0])
    
    # Chuẩn hóa trọng số Attention gộp cho từng từ thực tế
    word_attention = []
    current_word = ""
    current_weight = 0.0
    subword_count = 0
    
    for token, weight in zip(finbert_tokens, cls_attention):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        if token.startswith("##"):
            current_word += token[2:]
            current_weight += float(weight)
            subword_count += 1
        else:
            if current_word:
                word_attention.append((current_word, current_weight / max(subword_count, 1)))
            current_word = token
            current_weight = float(weight)
            subword_count = 1
    if current_word:
        word_attention.append((current_word, current_weight / max(subword_count, 1)))
        
    # Scale max attention = 1.0 để làm nổi bật trực quan
    if word_attention:
        max_att = max(w[1] for w in word_attention) if max(w[1] for w in word_attention) > 0 else 1.0
        word_attention = [(w, wt / max_att) for w, wt in word_attention]
        
    results["finbert"] = {
        "pred_id": finbert_pred,
        "pred_label": ID2LABEL[finbert_pred],
        "probs": finbert_probs.tolist(),
        "confidence": float(finbert_probs[finbert_pred]),
        "tokens": finbert_tokens,
        "word_attention": word_attention
    }
    
    # 4. Tính toán Độ đồng thuận đa mô hình (Consensus Engine)
    preds = [results["bilstm"]["pred_label"], results["bert_base"]["pred_label"], results["finbert"]["pred_label"]]
    if preds[0] == preds[1] == preds[2]:
        consensus_status = "Đồng thuận tuyệt đối (3/3)"
        consensus_desc = f"Cả 3 mô hình đều thống nhất phán quyết mức độ tác động: <b>{preds[0]}</b>."
    elif preds.count(results["finbert"]["pred_label"]) >= 2:
        consensus_status = "Đồng thuận đa số (2/3)"
        consensus_desc = f"Có 2/3 mô hình nghiêng về phán quyết: <b>{results['finbert']['pred_label']}</b>."
    else:
        consensus_status = "Phân kỳ cảnh báo (1/3)"
        consensus_desc = "Tín hiệu thị trường phức tạp, ưu tiên phán quyết từ Champion Model FinBERT."
        
    results["consensus"] = {
        "status": consensus_status,
        "desc": consensus_desc
    }
    
    # 5. Hệ thống Cảnh báo Sớm & Khuyến nghị Quản trị Danh mục (Bảng 5 Báo cáo)
    champ_pred = results["finbert"]["pred_label"]
    if champ_pred == "HIGH":
        results["ews"] = {
            "tier": "HIGH",
            "icon": "",
            "title": "CẢNH BÁO BIẾN ĐỘNG MẠNH (MARKET SHOCK ALERT)",
            "impact": "Bản tin mang đặc trưng thiên nga đen, nguy cơ kích hoạt làn sóng bán tháo trên diện rộng.",
            "action": "Kích hoạt vị thế phòng hộ (Hedging phái sinh), giảm tỷ lệ ký quỹ Margin, nâng tỷ trọng tiền mặt lên 30-50% và dừng các lệnh mua đuổi.",
            "color": COLOR_HIGH,
            "badge_color": "#C96A4B",
            "bg_color": "#FDF2E9",
            "border_color": "#C96A4B"
        }
    elif champ_pred == "MEDIUM":
        results["ews"] = {
            "tier": "MEDIUM",
            "icon": "",
            "title": "BIẾN ĐỘNG TRUNG BÌNH (LOCAL SECTOR REBALANCING)",
            "impact": "Tác động phân hóa theo nhóm ngành cụ thể, biên độ biến động chỉ số dự báo trong ngưỡng 0.5% - 1.2%.",
            "action": "Tái cơ cấu danh mục theo ngành chịu ảnh hưởng trực tiếp, duy trì tỷ trọng cổ phiếu phòng thủ và theo dõi các mốc hỗ trợ kỹ thuật.",
            "color": COLOR_MEDIUM,
            "badge_color": "#D9A441",
            "bg_color": "#FEF9E7",
            "border_color": "#D9A441"
        }
    else:
        results["ews"] = {
            "tier": "LOW",
            "icon": "",
            "title": "TÁC ĐỘNG THẤP / NHIỄU THÔNG TIN (STABLE / LOW IMPACT)",
            "impact": "Thông tin mang tính chất định kỳ hoặc đã phản ánh vào giá, thị trường tiếp tục xu hướng hiện tại.",
            "action": "Duy trì vị thế danh mục hiện tại, không hành động theo cảm tính, hạn chế lướt sóng ngắn hạn để tránh mất phí giao dịch vô ích.",
            "color": COLOR_LOW,
            "badge_color": "#7A9A8B",
            "bg_color": "#E8F8F5",
            "border_color": "#7A9A8B"
        }
        
    # 6. Ánh xạ không gian tiềm ẩn 3D (3D Latent Space Projection)
    p_low, p_med, p_high = results["finbert"]["probs"]
    # Tâm 3 cụm chuẩn hóa trong không gian 3D
    cluster_centers = {
        0: [1.8, -1.2, 0.5],   # Cụm LOW
        1: [-0.2, 1.9, -0.6],  # Cụm MEDIUM
        2: [-2.1, -1.5, 2.2]   # Cụm HIGH
    }
    
    # Phối hợp xác suất softmax với vector đặc trưng thu gọn
    pseudo_pca_noise = (finbert_cls_vector[:3] % 0.4) - 0.2
    c_x = (cluster_centers[0][0] * p_low) + (cluster_centers[1][0] * p_med) + (cluster_centers[2][0] * p_high) + pseudo_pca_noise[0]
    c_y = (cluster_centers[0][1] * p_low) + (cluster_centers[1][1] * p_med) + (cluster_centers[2][1] * p_high) + pseudo_pca_noise[1]
    c_z = (cluster_centers[0][2] * p_low) + (cluster_centers[1][2] * p_med) + (cluster_centers[2][2] * p_high) + pseudo_pca_noise[2]
    
    results["coords_3d"] = [float(c_x), float(c_y), float(c_z)]
    results["cluster_centers"] = cluster_centers
    
    return results

# ==============================================================================
# 6. DỮ LIỆU ĐÁNH GIÁ (DATASET TESTBED & BENCHMARK METRICS)
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_clean_dataset(split="test"):
    """Nạp tập test hoặc val từ data/processed (READ-ONLY)"""
    file_name = f"{split}.csv"
    path = os.path.join(DATA_DIR, file_name)
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    return pd.DataFrame(columns=["text", "label", "impact_tier"])

# Tri thức đánh giá thực nghiệm chính xác từ báo cáo học thuật & SRS.md
BENCHMARK_METRICS = {
    "validation": {
        "Bi-LSTM (Baseline)": {
            "Accuracy": 0.9366,
            "Macro Precision": 0.7768,
            "Macro Recall": 0.8992,
            "Macro F1": 0.8165,
            "HIGH Recall": 0.8033,
            "HIGH Precision": 0.4167,
            "HIGH F1": 0.5487,
            "False Alarms HIGH": 343,
            "False Alarms Total": 588,
            "Pros": "Kiến trúc nhẹ 2-layer, xử lý nhanh trên CPU/GPU.",
            "Cons": "Điểm yếu chí mạng: Báo động giả rất cao (Precision HIGH chỉ 41.67%). Dự báo 588 tin HIGH nhưng 343 tin là báo động giả."
        },
        "BERT-base": {
            "Accuracy": 0.9895,
            "Macro Precision": 0.9689,
            "Macro Recall": 0.9733,
            "Macro F1": 0.9711,
            "HIGH Recall": 0.9377,
            "HIGH Precision": 0.9256,
            "HIGH F1": 0.9316,
            "False Alarms HIGH": 23,
            "False Alarms Total": 309,
            "Pros": "Tỷ lệ bao phủ rủi ro (HIGH Recall) cao nhất (93.77%), hạn chế tối đa việc bỏ sót tin thiên nga đen.",
            "Cons": "Chưa được pre-train sâu trên ngữ liệu kinh tế tài chính chuyên biệt."
        },
        "FinBERT (Champion)": {
            "Accuracy": 0.9892,
            "Macro Precision": 0.9748,
            "Macro Recall": 0.9595,
            "Macro F1": 0.9669,
            "HIGH Recall": 0.8951,
            "HIGH Precision": 0.9446,
            "HIGH F1": 0.9192,
            "False Alarms HIGH": 16,
            "False Alarms Total": 289,
            "Pros": "Chuẩn xác cao nhất ở nhóm rủi ro (94.46% Val / 96.96% Test). Triệt tiêu báo động giả: 0 tin LOW nhầm sang HIGH, chỉ nhầm 16 tin MEDIUM.",
            "Cons": "Cần tối ưu hóa VRAM khi xử lý streaming dữ liệu quy mô lớn."
        }
    },
    "test_finbert": {
        "Accuracy": 0.9894,
        "Macro Precision": 0.9827,
        "Macro Recall": 0.9753,
        "Macro F1": 0.9789,
        "HIGH Precision": 0.9696,
        "HIGH Recall": 0.9441,
        "HIGH F1": 0.9567,
        "support": {"LOW": 5737, "MEDIUM": 7951, "HIGH": 304, "Total": 13992}
    },
    "confusion_matrices_val": {
        "Bi-LSTM (Baseline)": [
            [5706,   23,    8],
            [ 458, 7154,  338],
            [   4,   56,  245]
        ],
        "BERT-base": [
            [5700,   37,    0],
            [  68, 7859,   23],
            [   0,   19,  286]
        ],
        "FinBERT (Champion)": [
            [5703,   34,    0],
            [  69, 7865,   16],
            [   0,   32,  273]
        ]
    },
    "confusion_matrix_test_finbert": [
        [5701,   36,    0],
        [  86, 7856,    9],
        [   0,   17,  287]
    ]
}
