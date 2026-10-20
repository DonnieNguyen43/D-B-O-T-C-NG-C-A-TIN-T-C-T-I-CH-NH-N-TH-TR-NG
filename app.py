"""
app.py - Hệ Thống Phân Loại Mức Độ Tác Động Của Tin Tức Tài Chính Đến Thị Thị Trường Chứng Khoán
Đồ án Nghiên cứu Machine Learning: Nhóm 9 | Khoa Tài chính, Trường ĐH Kinh tế, ĐH Đà Nẵng | GVHD: PGS.TS. Nguyễn Văn Chức
Thiết kế theo chuẩn Premium Academic & FinTech Dashboard (Không sử dụng emoji, Typography in hoa trang trọng, SVG Logo độc quyền).
"""

import os
import random
import numpy as np
import pandas as pd
import streamlit as st

# Cấu hình giao diện Streamlit chuẩn Academic Dashboard (Không sử dụng emoji)
st.set_page_config(
    page_title="Phân Loại Tác Động Tin Tức Tài Chính | Machine Learning Research",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Nạp các module nghiệp vụ
from model_engine import (
    load_all_models,
    predict_text,
    load_clean_dataset,
    clean_news_text,
    format_input_text,
    inspect_cleaning_pipeline,
    BENCHMARK_METRICS,
    ID2LABEL,
    LABEL2ID,
    LABEL_COLORS,
    COLOR_LOW,
    COLOR_MEDIUM,
    COLOR_HIGH,
    COLOR_TERRACOTTA,
    COLOR_EARTH_BROWN,
    COLOR_WARM_BG,
    COLOR_CARD_BG,
    COLOR_BORDER_SOFT,
    COLOR_TEXT_HEADING,
    COLOR_TEXT_BODY,
    COLOR_TEXT_MUTED
)

from news_scraper import fetch_all_financial_news, FALLBACK_NEWS_DATABASE
from visualizer import (
    render_probability_bars,
    render_benchmark_comparison_chart,
    render_confusion_matrices,
    render_3d_latent_space,
    render_attention_heatmap_html,
    COLOR_BG_CARD,
    COLOR_BORDER,
    COLOR_GRID
)

def render_html(html_str: str):
    """
    Render HTML an toàn trong Streamlit, loại bỏ hoàn toàn hiện tượng Markdown
    hiểu nhầm dòng thụt lề thành indented code block (<pre><code>).
    """
    clean_lines = []
    for line in html_str.splitlines():
        s = line.strip()
        if not s or (s.startswith('<!--') and s.endswith('-->')):
            continue
        clean_lines.append(s)
    st.markdown(" ".join(clean_lines), unsafe_allow_html=True)

def render_probability_distribution_view(results_dict, key_id):
    """
    Render tiêu đề và chú giải màu sắc rõ nét phía trên biểu đồ thanh ngang,
    loại bỏ hoàn toàn nguy cơ đè chữ/dính chữ bên trong Plotly.
    """
    render_html("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:16px; margin-bottom:4px; flex-wrap:wrap; gap:10px;">
        <div style="font-size:13px; font-weight:700; color:#2E2620; text-transform:uppercase; letter-spacing:0.04em;">
            PHÂN PHỐI XÁC SUẤT SUY LUẬN TRỰC DIỆN [P_LOW, P_MEDIUM, P_HIGH]
        </div>
        <div style="display:flex; gap:16px; font-size:12px; font-weight:600; color:#4A4038; align-items:center; flex-wrap:wrap;">
            <span><span style="display:inline-block; width:10px; height:10px; border-radius:2px; background:#7A9A8B; vertical-align:middle; margin-right:4px;"></span> LOW (Thấp)</span>
            <span><span style="display:inline-block; width:10px; height:10px; border-radius:2px; background:#D9A441; vertical-align:middle; margin-right:4px;"></span> MEDIUM (Trung bình)</span>
            <span><span style="display:inline-block; width:10px; height:10px; border-radius:2px; background:#C96A4B; vertical-align:middle; margin-right:4px;"></span> HIGH (Mạnh / Rủi ro)</span>
        </div>
    </div>
    """)
    st.plotly_chart(render_probability_bars(results_dict), use_container_width=True, key=key_id)

# ==============================================================================
# CSS OVERRIDE HỌC THUẬT: TYPOGRAPHY, CHỐNG TRÀN CHỮ, CONTAINER 1200PX & SCROLL ANIMATION
# ==============================================================================
st.markdown("""
<style>
    /* 1. CHUẨN HÓA 2 PHÔNG CHỮ SANS-SERIF & MONOSPACE */
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400;1,500;1,600;1,700&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');
    
    /* Ẩn overlay header của Streamlit */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        z-index: 1 !important;
    }
    
    /* Giới hạn khung hình chuẩn 1200px cho trải nghiệm đọc tối ưu trên màn hình lớn */
    .block-container {
        max-width: 1200px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-top: 1.2rem !important;
        padding-bottom: 6rem !important;
    }
    
    /* Nền toàn cục kem ấm */
    html, body, [class*="css"] {
        font-family: 'Be Vietnam Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    .stApp {
        background-color: #FAF6F0 !important;
        color: #4A4038 !important;
    }
    
    /* 2. HEADINGS SCALE - ĐỘ ĐẬM 800, MÀU NÂU ĐEN #2E2620 */
    h1, h2, h3, h4, h5, .academic-title {
        font-family: 'Be Vietnam Pro', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #2E2620 !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
    }
    
    h1 {
        font-size: 26px !important;
        font-weight: 800 !important;
        margin-top: 6px !important;
        margin-bottom: 12px !important;
        line-height: 1.3 !important;
    }
    
    h2 {
        font-size: 21px !important;
        font-weight: 800 !important;
        margin-top: 18px !important;
        margin-bottom: 10px !important;
        line-height: 1.35 !important;
    }
    
    h3 {
        font-size: 18px !important;
        font-weight: 800 !important;
        margin-top: 16px !important;
        margin-bottom: 8px !important;
        line-height: 1.4 !important;
    }
    
    h4 {
        font-size: 15.5px !important;
        font-weight: 800 !important;
        margin-top: 12px !important;
        margin-bottom: 6px !important;
    }
    
    h5 {
        font-size: 14px !important;
        font-weight: 800 !important;
        color: #4A4038 !important;
    }
    
    p, li, span, label, div {
        line-height: 1.65;
    }

    /* 3. FIX TRIỆT ĐỂ LỖI TRÀN CHỮ TRONG SELECTBOX & MENUS */
    .stSelectbox,
    div[data-baseweb="select"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] div[role="button"],
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] * {
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }
    
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #D9CDC1 !important;
        border-radius: 10px !important;
        color: #2E2620 !important;
        min-height: 42px !important;
    }
    
    div[data-baseweb="select"] * {
        color: #2E2620 !important;
        -webkit-text-fill-color: #2E2620 !important;
        font-family: 'Be Vietnam Pro', sans-serif !important;
        font-size: 14px !important;
    }
    
    ul[role="listbox"] li,
    div[data-baseweb="popover"] li {
        background-color: #FFFFFF !important;
        color: #2E2620 !important;
        font-family: 'Be Vietnam Pro', sans-serif !important;
        padding: 9px 15px !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        font-size: 13.5px !important;
    }
    
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background-color: #FDF2E9 !important;
        color: #8C4A32 !important;
    }
    li[role="option"]:hover *,
    li[role="option"][aria-selected="true"] * {
        color: #8C4A32 !important;
        -webkit-text-fill-color: #8C4A32 !important;
    }
    
    /* Ô nhập liệu Form Widget */
    label[data-testid="stWidgetLabel"], 
    .stWidgetLabel p, 
    label[data-testid="stWidgetLabel"] span {
        color: #2E2620 !important;
        font-family: 'Be Vietnam Pro', sans-serif !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    
    input[type="text"], 
    input, 
    textarea, 
    div[data-baseweb="input"] input, 
    div[data-baseweb="textarea"] textarea {
        color: #2E2620 !important;
        background-color: #FFFFFF !important;
        -webkit-text-fill-color: #2E2620 !important;
        border: 1px solid #D9CDC1 !important;
        border-radius: 10px !important;
        font-family: 'Be Vietnam Pro', sans-serif !important;
        font-size: 14px !important;
        box-shadow: none !important;
    }
    input:focus, textarea:focus {
        border-color: #C96A4B !important;
        box-shadow: 0 0 0 2px rgba(201, 106, 75, 0.2) !important;
    }
    
    /* Radio Buttons */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label span {
        color: #2E2620 !important;
        font-family: 'Be Vietnam Pro', sans-serif !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }
    
    /* 4. ĐIỀU HƯỚNG TABS IN HOA, LETTER-SPACING RỘNG, BORDER-BOTTOM SẮC NÉT */
    div[data-baseweb="tab-list"] {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 8px !important;
        background-color: #F0E6DC !important;
        padding: 6px 10px !important;
        border-radius: 12px !important;
        border: 1px solid #D9CDC1 !important;
        margin-bottom: 24px !important;
        align-items: center !important;
    }
    
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        border: 1px solid transparent !important;
        border-bottom: 3px solid transparent !important;
        border-radius: 8px !important;
        padding: 9px 18px !important;
        color: #7A7067 !important;
        font-family: 'Be Vietnam Pro', sans-serif !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important;
    }
    
    button[data-baseweb="tab"]:hover {
        color: #C96A4B !important;
        background-color: rgba(255, 255, 255, 0.6) !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #8C4A32 !important;
        font-weight: 800 !important;
        border: 1px solid #D9CDC1 !important;
        border-bottom: 3px solid #C96A4B !important;
        box-shadow: 0 3px 10px rgba(46, 38, 32, 0.08) !important;
    }
    
    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] span {
        color: inherit !important;
        font-size: inherit !important;
        font-weight: inherit !important;
        letter-spacing: inherit !important;
    }

    /* 5. HERO SECTION CARD - PREMIUM ACADEMIC & FINTECH */
    .hero-container {
        position: relative !important;
        overflow: hidden !important;
        background: #FFFFFF !important;
        border: 1px solid #D9CDC1 !important;
        border-top: 4px solid #C96A4B !important;
        border-radius: 20px !important;
        padding: 44px 36px 32px 36px !important;
        margin-bottom: 26px !important;
        box-shadow: 0 6px 24px rgba(46, 38, 32, 0.05) !important;
        text-align: center !important;
    }
    
    .hero-logo-box {
        display: inline-flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 18px;
    }
    
    .hero-title-h1 {
        font-family: 'Be Vietnam Pro', sans-serif !important;
        font-size: 29px !important;
        font-weight: 800 !important;
        color: #2E2620 !important;
        letter-spacing: -0.025em !important;
        line-height: 1.3 !important;
        margin-bottom: 10px !important;
        text-transform: uppercase !important;
    }
    
    .hero-subtitle {
        font-size: 15.5px !important;
        color: #4A4038 !important;
        line-height: 1.65 !important;
        max-width: 860px !important;
        margin: 0 auto 16px auto !important;
        font-weight: 400 !important;
    }
    
    .hero-course-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 22px;
        border-radius: 9999px;
        background: #FAF6F0;
        border: 1px solid #D9CDC1;
        font-size: 13px;
        font-weight: 600;
        color: #8C4A32;
        margin-top: 8px;
        box-shadow: 0 1px 4px rgba(46, 38, 32, 0.04);
        max-width: 100%;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    
    .hero-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #C96A4B;
        display: inline-block;
        flex-shrink: 0;
    }
    
    /* Scroll Down Indicator Animation */
    @keyframes scrollBounce {
        0%, 100% {
            transform: translateY(0);
            opacity: 0.85;
        }
        50% {
            transform: translateY(6px);
            opacity: 0.35;
        }
    }
    
    .hero-scroll-indicator {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 24px;
        animation: scrollBounce 2.2s infinite ease-in-out;
    }

    /* Đồng bộ chiều cao các cột trong Streamlit layout */
    [data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }
    
    [data-testid="column"] > div {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 auto !important;
        height: 100% !important;
    }
    
    [data-testid="column"] [data-testid="stVerticalBlock"] {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 auto !important;
        height: 100% !important;
    }

    /* 6. TECHNICAL STAT CARDS & KPI CARDS */
    .academic-stat-card {
        background: #FFFFFF !important;
        border: 1px solid #D9CDC1 !important;
        border-radius: 12px !important;
        padding: 20px 22px !important;
        box-shadow: 0 2px 10px rgba(46, 38, 32, 0.05) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
        min-height: 185px !important;
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        box-sizing: border-box !important;
    }
    .academic-stat-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(46, 38, 32, 0.09) !important;
        border-color: #C96A4B !important;
    }
    .stat-number {
        font-family: 'JetBrains Mono', 'Be Vietnam Pro', monospace, sans-serif !important;
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #8C4A32 !important;
        margin-bottom: 6px !important;
        letter-spacing: -0.01em !important;
        line-height: 1.2 !important;
    }
    .stat-label {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #2E2620 !important;
        margin-bottom: 8px !important;
        line-height: 1.3 !important;
    }
    .stat-desc {
        font-size: 12.5px !important;
        color: #7A7067 !important;
        line-height: 1.55 !important;
        margin-top: auto !important;
    }
    
    .academic-card {
        background: #FFFFFF !important;
        border: 1px solid #D9CDC1 !important;
        border-radius: 12px !important;
        padding: 22px 24px !important;
        margin-bottom: 18px !important;
        box-shadow: 0 2px 10px rgba(46, 38, 32, 0.05) !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    
    .kpi-card {
        background: #FFFFFF !important;
        border: 1px solid #D9CDC1 !important;
        border-radius: 12px !important;
        padding: 20px 22px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 2px 10px rgba(46, 38, 32, 0.05) !important;
        transition: transform 0.2s ease, border-color 0.2s ease !important;
        min-height: 235px !important;
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        box-sizing: border-box !important;
    }
    .kpi-card:hover {
        border-color: #C96A4B !important;
        transform: translateY(-2px) !important;
    }
    .kpi-card-champion {
        background: #FAF6F0 !important;
        border: 1.5px solid #C96A4B !important;
        box-shadow: 0 4px 16px rgba(201, 106, 75, 0.12) !important;
    }
    .kpi-title {
        font-size: 13px !important;
        font-weight: 700 !important;
        color: #7A7067 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        margin-bottom: 4px !important;
    }
    .kpi-value {
        font-size: 30px !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
        margin: 6px 0 !important;
        line-height: 1.2 !important;
    }
    .kpi-pill {
        display: inline-block !important;
        padding: 4px 12px !important;
        border-radius: 6px !important;
        font-size: 11.5px !important;
        font-weight: 700 !important;
        letter-spacing: 0.03em !important;
    }
    .kpi-footer-note {
        font-size: 12px !important;
        color: #7A7067 !important;
        line-height: 1.5 !important;
        border-top: 1px solid #D9CDC1 !important;
        padding-top: 10px !important;
        margin-top: auto !important;
        min-height: 52px !important;
        display: flex !important;
        align-items: center !important;
        box-sizing: border-box !important;
    }
    
    .ews-alert-desk {
        border-radius: 12px !important;
        padding: 22px 24px !important;
        margin: 16px 0 22px 0 !important;
        border-left: 5px solid !important;
        box-shadow: 0 3px 14px rgba(46, 38, 32, 0.06) !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    
    .sample-container {
        background: #FFFFFF !important;
        border: 1px solid #D9CDC1 !important;
        border-radius: 12px !important;
        padding: 18px 22px !important;
        margin: 14px 0 20px 0 !important;
        box-shadow: 0 2px 10px rgba(46, 38, 32, 0.05) !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    
    .academic-text-box {
        background: #FAF6F0 !important;
        border: 1px solid #D9CDC1 !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 13px !important;
        color: #2E2620 !important;
        line-height: 1.65 !important;
        word-break: break-word !important;
        overflow-wrap: break-word !important;
        white-space: pre-wrap !important;
        box-shadow: inset 0 1px 3px rgba(46, 38, 32, 0.04) !important;
    }
    
    .academic-note {
        border-left: 3px solid #C96A4B !important;
        padding: 12px 18px !important;
        color: #4A4038 !important;
        font-size: 14.5px !important;
        line-height: 1.65 !important;
        margin: 14px 0 !important;
        background: #FFFFFF !important;
        border-radius: 0 8px 8px 0 !important;
        border-top: 1px solid #D9CDC1 !important;
        border-right: 1px solid #D9CDC1 !important;
        border-bottom: 1px solid #D9CDC1 !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    
    /* Nút bấm giao diện */
    .stButton > button {
        background-color: #C96A4B !important;
        color: #FFFFFF !important;
        border: 1px solid #B05335 !important;
        border-radius: 10px !important;
        font-family: 'Be Vietnam Pro', sans-serif !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 8px 20px !important;
        box-shadow: 0 2px 8px rgba(201, 106, 75, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #8C4A32 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(140, 74, 50, 0.3) !important;
    }
    
    [data-testid="stDataFrame"] {
        border-radius: 10px !important;
        border: 1px solid #D9CDC1 !important;
        box-shadow: 0 2px 8px rgba(46, 38, 32, 0.04) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# KHỞI TẠO HỆ THỐNG MÔ HÌNH VỚI CACHING
# ==============================================================================
@st.cache_resource(show_spinner=True)
def init_system():
    with st.spinner("Đang nạp trọng số 3 kiến trúc mô hình (Bi-LSTM, BERT-base, FinBERT) vào bộ nhớ..."):
        return load_all_models()

try:
    resources = init_system()
except Exception as e:
    st.error(f"Lỗi khi khởi tạo tài nguyên mô hình: {e}")
    st.stop()

# ==============================================================================
# HERO SECTION - THIẾT KẾ ĐỘC QUYỀN (SVG LOGO & ĐỒ HỌA 3D NỀN HAI BÊN)
# ==============================================================================
render_html(f"""
<div class="hero-container">
    <div style="position: absolute; top: -30px; left: -30px; width: 230px; height: 230px; opacity: 0.06; transform: rotate(-15deg); pointer-events: none;">
        <svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
            <rect x="6" y="6" width="88" height="88" rx="22" stroke="#8C4A32" stroke-width="3"/>
            <line x1="30" y1="20" x2="30" y2="80" stroke="#8C4A32" stroke-width="2.5"/>
            <rect x="23" y="35" width="14" height="30" rx="3" fill="#8C4A32"/>
            <line x1="50" y1="15" x2="50" y2="85" stroke="#8C4A32" stroke-width="2.5"/>
            <rect x="42" y="25" width="16" height="45" rx="3" fill="#8C4A32"/>
            <line x1="70" y1="22" x2="70" y2="78" stroke="#8C4A32" stroke-width="2.5"/>
            <rect x="63" y="32" width="14" height="35" rx="3" fill="#8C4A32"/>
            <path d="M30 50 L50 30 L70 42" stroke="#8C4A32" stroke-width="3" stroke-dasharray="3 2"/>
            <circle cx="30" cy="50" r="6" fill="#8C4A32"/>
            <circle cx="50" cy="30" r="8" fill="#8C4A32"/>
            <circle cx="70" cy="42" r="6" fill="#8C4A32"/>
        </svg>
    </div>

    <div style="position: absolute; bottom: -40px; right: -35px; width: 250px; height: 250px; opacity: 0.06; transform: rotate(20deg); pointer-events: none;">
        <svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
            <rect x="6" y="6" width="88" height="88" rx="22" stroke="#C96A4B" stroke-width="3"/>
            <line x1="30" y1="20" x2="30" y2="80" stroke="#C96A4B" stroke-width="2.5"/>
            <rect x="23" y="35" width="14" height="30" rx="3" fill="#C96A4B"/>
            <line x1="50" y1="15" x2="50" y2="85" stroke="#C96A4B" stroke-width="2.5"/>
            <rect x="42" y="25" width="16" height="45" rx="3" fill="#C96A4B"/>
            <line x1="70" y1="22" x2="70" y2="78" stroke="#C96A4B" stroke-width="2.5"/>
            <rect x="63" y="32" width="14" height="35" rx="3" fill="#C96A4B"/>
            <path d="M30 50 L50 30 L70 42" stroke="#C96A4B" stroke-width="3" stroke-dasharray="3 2"/>
            <circle cx="30" cy="50" r="6" fill="#C96A4B"/>
            <circle cx="50" cy="30" r="8" fill="#C96A4B"/>
            <circle cx="70" cy="42" r="6" fill="#C96A4B"/>
        </svg>
    </div>

    <div class="hero-logo-box">
        <svg width="80" height="80" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="logoPrimaryGrad" x1="10" y1="10" x2="90" y2="90" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stop-color="#C96A4B"/>
                    <stop offset="100%" stop-color="#8C4A32"/>
                </linearGradient>
                <linearGradient id="candleGrad1" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#7A9A8B"/>
                    <stop offset="100%" stop-color="#5C7C6D"/>
                </linearGradient>
                <linearGradient id="candleGrad2" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#E8A388"/>
                    <stop offset="100%" stop-color="#C96A4B"/>
                </linearGradient>
                <linearGradient id="candleGrad3" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#E5B869"/>
                    <stop offset="100%" stop-color="#D9A441"/>
                </linearGradient>
            </defs>

            <rect x="6" y="6" width="88" height="88" rx="22" fill="#FFFFFF" stroke="url(#logoPrimaryGrad)" stroke-width="2.5"/>
            <rect x="10" y="10" width="80" height="80" rx="18" fill="#FAF6F0" opacity="0.6"/>

            <line x1="30" y1="36" x2="30" y2="68" stroke="#7A9A8B" stroke-width="1.8" stroke-linecap="round"/>
            <rect x="25" y="44" width="10" height="16" rx="2.5" fill="url(#candleGrad1)"/>

            <line x1="50" y1="22" x2="50" y2="78" stroke="#8C4A32" stroke-width="2" stroke-linecap="round"/>
            <rect x="44.5" y="30" width="11" height="30" rx="2.5" fill="url(#candleGrad2)"/>

            <line x1="70" y1="30" x2="70" y2="72" stroke="#D9A441" stroke-width="1.8" stroke-linecap="round"/>
            <rect x="65" y="38" width="10" height="22" rx="2.5" fill="url(#candleGrad3)"/>

            <path d="M30 52 L50 36 L70 44" stroke="#8C4A32" stroke-width="2" stroke-dasharray="3 2" opacity="0.8"/>
            <path d="M30 60 L50 60 L70 54" stroke="#C96A4B" stroke-width="1.5" opacity="0.6"/>

            <circle cx="30" cy="52" r="4.5" fill="#FFFFFF" stroke="#7A9A8B" stroke-width="2.5"/>
            <circle cx="50" cy="36" r="6" fill="#FFFFFF" stroke="#C96A4B" stroke-width="3"/>
            <circle cx="70" cy="44" r="4.5" fill="#FFFFFF" stroke="#D9A441" stroke-width="2.5"/>

            <circle cx="50" cy="36" r="2.2" fill="#8C4A32"/>
        </svg>
    </div>

    <div class="hero-title-h1">
        DỰ BÁO TÁC ĐỘNG CỦA TIN TỨC TÀI CHÍNH ĐẾN THỊ TRƯỜNG
    </div>

    <div class="hero-subtitle">
        Hệ thống Đánh giá và Phân loại Rủi ro Định lượng dựa trên Kiến trúc Học sâu (Bi-LSTM, BERT, FinBERT).
    </div>

    <div>
        <div class="hero-course-pill">
            <span class="hero-dot"></span>
            Đồ án Nghiên cứu Machine Learning: Nhóm 9 | Khoa Tài chính, Trường ĐH Kinh tế, ĐH Đà Nẵng | GVHD: PGS.TS. Nguyễn Văn Chức
        </div>
    </div>

    <div class="hero-scroll-indicator" title="Cuộn trang để xem nội dung nghiên cứu">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#8C4A32" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M7 10l5 5 5-5"/>
            <path d="M7 4l5 5 5-5" opacity="0.35"/>
        </svg>
    </div>
</div>
""")

# 3 Thẻ thống kê kỹ thuật (Technical Architecture Cards)
c_st1, c_st2, c_st3 = st.columns(3)

with c_st1:
    render_html("""
    <div class="academic-stat-card">
        <div class="stat-number">13,992</div>
        <div class="stat-label">Mẫu Kiểm Thử Độc Lập (Test Set)</div>
        <div class="stat-desc">Dữ liệu kiểm thử độc lập đã làm sạch chuẩn Anti-Leakage, loại bỏ hoàn toàn siêu liên kết và thẻ định dạng.</div>
    </div>
    """)

with c_st2:
    render_html("""
    <div class="academic-stat-card" style="border-left: 4px solid #C96A4B;">
        <div class="stat-number" style="color: #C96A4B;">94.20%</div>
        <div class="stat-label">Macro F1-Score (FinBERT)</div>
        <div class="stat-desc">Độ chuẩn xác nhận diện rủi ro đạt 96.96% trên tập Test, triệt tiêu hoàn toàn báo động giả từ tin nhiễu.</div>
    </div>
    """)

with c_st3:
    render_html("""
    <div class="academic-stat-card">
        <div class="stat-number">3 Kiến Trúc</div>
        <div class="stat-label">Tiến Trình Phát Triển Kiến Trúc</div>
        <div class="stat-desc">Đối sánh thực nghiệm từ mạng nơ-ron hồi quy tuần tự (Bi-LSTM) đến cơ chế Self-Attention toàn cục và tinh chỉnh miền tài chính.</div>
    </div>
    """)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# TOP-LEVEL TABS ĐIỀU HƯỚNG (CHỮ IN HOA, LETTER-SPACING RỘNG, THIẾT KẾ SANG TRỌNG)
# ==============================================================================
tab_testbed, tab_realtime, tab_pipeline, tab_benchmark = st.tabs([
    "THỰC NGHIỆM DATASET ĐỘC LẬP",
    "SUY LUẬN REAL-TIME & CẢNH BÁO SỚM",
    "TIỀN XỬ LÝ NLP & KHÔNG GIAN 3D",
    "ĐỐI SÁNH BENCHMARK & BÁO CÁO KỸ THUẬT"
])

# ==============================================================================
# PHẦN 1: THỰC NGHIỆM TRÊN TẬP DỮ LIỆU ĐỘC LẬP & MA TRẬN NHẦM LẪN
# ==============================================================================
with tab_testbed:
    st.markdown("### THỰC NGHIỆM TRÊN TẬP DỮ LIỆU ĐÃ LÀM SẠCH (DATASET TESTBED)")
    render_html("""
    <div class="academic-note">
        <b>LƯU Ý THỰC NGHIỆM:</b> Tập kiểm thử độc lập gồm 13,992 mẫu chưa từng xuất hiện trong quá trình huấn luyện và tinh chỉnh tham số, được dùng để đối chiếu độ tin cậy và kiểm chứng hiện tượng phân kỳ dự đoán giữa các kiến trúc trong điều kiện thực tế.
    </div>
    """)
    
    col_split, col_tier, col_kw = st.columns([1.3, 1.2, 2.5])
    with col_split:
        data_split = st.selectbox(
            "1. Tập dữ liệu kiểm thử:",
            ["test", "val"],
            format_func=lambda x: "Tập Test độc lập (13,992 mẫu)" if x == "test" else "Tập Validation (13,992 mẫu)",
            index=0,
            key="tb_select_dataset_split"
        )
        
    df_data = load_clean_dataset(data_split)
    
    with col_tier:
        tier_filter = st.selectbox(
            "2. Lọc theo Ground Truth thực tế:",
            ["Tất cả (All)", "LOW (Nhãn 0)", "MEDIUM (Nhãn 1)", "HIGH (Nhãn 2)"],
            index=0,
            key="tb_select_tier_filter"
        )
        
    with col_kw:
        search_kw = st.text_input(
            "3. Tìm kiếm theo từ khóa (vd: Fed, Inflation, Rate, Revenue, Profit):",
            value="",
            key="tb_input_search_kw"
        )
        
    # Lọc dữ liệu
    df_filtered = df_data.copy()
    if tier_filter != "Tất cả (All)":
        target_name = tier_filter.split(" ")[0]
        df_filtered = df_filtered[df_filtered["impact_tier"] == target_name]
    if search_kw.strip():
        df_filtered = df_filtered[df_filtered["text"].str.contains(search_kw.strip(), case=False, na=False)]
        
    # Chọn mẫu ngẫu nhiên hoặc từ dropdown
    c_btn, c_drop = st.columns([1, 3])
    with c_btn:
        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
        btn_random = st.button("Lấy mẫu ngẫu nhiên (Random Sample)", use_container_width=True, key="tb_btn_random_sample")
        
    if "sample_id" not in st.session_state or btn_random:
        if len(df_filtered) > 0:
            st.session_state.sample_id = int(random.choice(df_filtered.index.tolist()))
        else:
            st.session_state.sample_id = 0
            
    with c_drop:
        if len(df_filtered) > 0:
            indices = df_filtered.index.tolist()
            display_indices = indices[:80]
            if st.session_state.sample_id not in display_indices:
                display_indices = [st.session_state.sample_id] + display_indices[:79]
                
            def format_sample_title(idx):
                tier = df_data.loc[idx, 'impact_tier']
                raw_txt = str(df_data.loc[idx, 'text']).strip()
                short_txt = (raw_txt[:68] + "...") if len(raw_txt) > 70 else raw_txt
                return f"#{idx} [{tier}] {short_txt}"
                
            selected_sample_id = st.selectbox(
                f"Hoặc chọn mẫu cụ thể từ {len(df_filtered):,} kết quả:",
                options=display_indices,
                index=display_indices.index(st.session_state.sample_id) if st.session_state.sample_id in display_indices else 0,
                format_func=format_sample_title,
                key="tb_select_sample_dropdown"
            )
            st.session_state.sample_id = selected_sample_id
            
    # Hiển thị nội dung bản tin được chọn
    curr_id = st.session_state.sample_id
    if curr_id in df_data.index:
        sample_row = df_data.loc[curr_id]
        sample_text = str(sample_row["text"])
        ground_truth = sample_row["impact_tier"]
        gt_color = LABEL_COLORS[ground_truth]
        
        render_html(f"""
        <div class="sample-container" style="border-left: 5px solid {gt_color};">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
                <span style="font-size:13px; font-weight:700; color:#8C4A32; text-transform:uppercase; letter-spacing:0.04em;">
                    Mẫu tin ID #{curr_id} từ: <b>Tập {data_split.upper()}</b>
                </span>
                <span style="background:{gt_color}; color:#FFFFFF; padding:4px 12px; border-radius:6px; font-weight:700; font-size:12px; font-family:'JetBrains Mono';">
                    Ground Truth (Thực tế): {ground_truth} (Lớp {sample_row['label']})
                </span>
            </div>
            <div style="font-size: 15px; color: #2E2620; line-height: 1.65;">
                {sample_text}
            </div>
        </div>
        """)
        
        # CHẠY SUY LUẬN QUA CẢ 3 MÔ HÌNH
        results = predict_text(sample_text, resources)
        
        # 1. THẺ KPI METRIC CARDS
        st.markdown("#### KẾT QUẢ SUY LUẬN TRỰC DIỆN CỦA 3 MÔ HÌNH SO VỚI GROUND TRUTH")
        col_m1, col_m2, col_m3 = st.columns(3)
        
        # Bi-LSTM Card
        with col_m1:
            p_lbl = results["bilstm"]["pred_label"]
            conf = results["bilstm"]["confidence"] * 100
            is_match = (p_lbl == ground_truth)
            status_text = "CHÍNH XÁC" if is_match else "SAI LỆCH"
            status_color = "#117864" if is_match else "#8C4A32"
            render_html(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-title">
                        <span>1. Bi-LSTM (Baseline)</span>
                        <span style="color:{status_color}; font-weight:700; font-size:11.5px;">{status_text}</span>
                    </div>
                    <div class="kpi-value" style="color: {LABEL_COLORS[p_lbl]};">
                        {conf:.2f}%
                    </div>
                    <div style="margin-bottom: 10px;">
                        <span class="kpi-pill" style="background:{LABEL_COLORS[p_lbl]}; color:white;">
                            DỰ ĐOÁN: {p_lbl}
                        </span>
                    </div>
                </div>
                <div class="kpi-footer-note">
                    <span><i>Hạn chế:</i> Tỷ lệ báo động giả cao (Precision HIGH chỉ <b>41.67%</b>).</span>
                </div>
            </div>
            """)
            
        # BERT-base Card
        with col_m2:
            p_lbl = results["bert_base"]["pred_label"]
            conf = results["bert_base"]["confidence"] * 100
            is_match = (p_lbl == ground_truth)
            status_text = "CHÍNH XÁC" if is_match else "SAI LỆCH"
            status_color = "#117864" if is_match else "#8C4A32"
            render_html(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-title">
                        <span>2. BERT-base (Transformer)</span>
                        <span style="color:{status_color}; font-weight:700; font-size:11.5px;">{status_text}</span>
                    </div>
                    <div class="kpi-value" style="color: {LABEL_COLORS[p_lbl]};">
                        {conf:.2f}%
                    </div>
                    <div style="margin-bottom: 10px;">
                        <span class="kpi-pill" style="background:{LABEL_COLORS[p_lbl]}; color:white;">
                            DỰ ĐOÁN: {p_lbl}
                        </span>
                    </div>
                </div>
                <div class="kpi-footer-note">
                    <span><i>Ưu điểm:</i> Bao phủ rủi ro diện rộng (Recall HIGH <b>93.77%</b>).</span>
                </div>
            </div>
            """)

        # FinBERT Champion Card
        with col_m3:
            p_lbl = results["finbert"]["pred_label"]
            conf = results["finbert"]["confidence"] * 100
            is_match = (p_lbl == ground_truth)
            status_text = "CHÍNH XÁC" if is_match else "SAI LỆCH"
            status_color = "#117864" if is_match else "#8C4A32"
            render_html(f"""
            <div class="kpi-card kpi-card-champion">
                <div>
                    <div class="kpi-title">
                        <span style="color:#8C4A32; font-weight:700;">3. FinBERT (Champion)</span>
                        <span style="color:{status_color}; font-weight:700; font-size:11.5px;">{status_text}</span>
                    </div>
                    <div class="kpi-value" style="color: {LABEL_COLORS[p_lbl]};">
                        {conf:.2f}%
                    </div>
                    <div style="margin-bottom: 10px;">
                        <span class="kpi-pill" style="background:{LABEL_COLORS[p_lbl]}; color:white;">
                            DỰ ĐOÁN: {p_lbl}
                        </span>
                    </div>
                </div>
                <div class="kpi-footer-note" style="border-top-color: #C96A4B; color: #4A4038;">
                    <span><i>Hiệu năng tối ưu:</i> Chuẩn xác rủi ro <b>94.46% (Val) / 96.96% (Test)</b>, 0 ca báo động giả từ LOW.</span>
                </div>
            </div>
            """)

        # 2. THANH XÁC SUẤT NẰM NGANG
        render_probability_distribution_view(results, "tb_prob_bar_chart")

        # 3. ĐỒNG BỘ MA TRẬN NHẦM LẪN THEO TẬP DỮ LIỆU ĐANG CHỌN
        st.markdown("---")
        if data_split == "test":
            st.markdown("#### MA TRẬN NHẦM LẪN KIỂM THỬ ĐỘC LẬP TRÊN TẬP TEST (N = 13,992 MẪU)")
            st.markdown("""
            *Ghi chú kỹ thuật:* Tập Test gồm 13,992 mẫu độc lập hoàn toàn mới (chưa từng tham gia tinh chỉnh siêu tham số).
            Dưới đây là ma trận kiểm định của Champion Model **FinBERT** (Accuracy = **98.94%**, HIGH Precision = **96.96%**, HIGH Recall = **94.41%**).
            """)
            test_cm_dict = {"FinBERT (Independent Test Champion)": BENCHMARK_METRICS["confusion_matrix_test_finbert"]}
            st.plotly_chart(render_confusion_matrices(test_cm_dict, split_title="TẬP TEST ĐỘC LẬP (N = 13,992)"), use_container_width=True, key="tb_cm_test_chart")
        else:
            st.markdown("#### MA TRẬN NHẦM LẪN ĐỐI SÁNH TRÊN TẬP VALIDATION (N = 13,992 MẪU)")
            st.markdown("""
            *Ghi chú kỹ thuật:* Tập Validation đối sánh trực tiếp cả 3 mô hình nhằm đánh giá độ hội tụ và sự phân kỳ giữa Baseline Bi-LSTM và 2 kiến trúc Transformer.
            """)
            st.plotly_chart(render_confusion_matrices(BENCHMARK_METRICS["confusion_matrices_val"], split_title="TẬP VALIDATION (N = 13,992)"), use_container_width=True, key="tb_cm_val_chart")

# ==============================================================================
# PHẦN 2: DỰ BÁO LUỒNG TIN REAL-TIME
# ==============================================================================
with tab_realtime:
    st.markdown("### DỰ BÁO LUỒNG TIN REAL-TIME & CẢNH BÁO SỚM")
    
    # Hàng điều khiển thu thập tin tức
    col_f1, col_f2 = st.columns([1.2, 2.8])
    with col_f1:
        btn_fetch = st.button("Cập nhật tin mới (Fetch Live News)", use_container_width=True, key="rt_btn_fetch_news")
    with col_f2:
        src_filter = st.selectbox(
            "Nguồn cấp tin tức:",
            ["Tất cả nguồn uy tín (Google News, Yahoo, CNBC)", "Google News Financial", "Yahoo Finance", "CNBC Markets"],
            key="rt_select_source_filter"
        )
        
    if "live_news_cache" not in st.session_state or btn_fetch:
        with st.spinner("Đang kết nối luồng tin tức tài chính mới nhất..."):
            items, is_fallback = fetch_all_financial_news(limit_per_source=8)
            st.session_state.live_news_cache = items
            st.session_state.is_fallback = is_fallback
            
    live_items = st.session_state.live_news_cache
    if st.session_state.get("is_fallback", False):
        st.caption("Đang sử dụng bộ đệm tin tức dự phòng do mạng hoặc nguồn RSS giới hạn.")
        
    if src_filter != "Tất cả nguồn uy tín (Google News, Yahoo, CNBC)":
        filtered_items = [x for x in live_items if x["source"] == src_filter]
        if not filtered_items:
            filtered_items = live_items
    else:
        filtered_items = live_items
        
    def format_live_title(i):
        item = filtered_items[i]
        src = item['source']
        t = item['title'].strip()
        short_t = (t[:75] + "...") if len(t) > 78 else t
        return f"[{src}] {short_t}"
        
    selected_live_idx = st.selectbox(
        "Chọn bản tin tài chính để xem và phán đoán mức độ tác động:",
        options=range(len(filtered_items)),
        format_func=format_live_title,
        key="rt_select_live_article"
    )
    
    article = filtered_items[selected_live_idx]
    title_in = article["title"]
    desc_in = article.get("description", "")
    source_in = article.get("source", "Tài chính")
    pub_in = article.get("published", "Vừa cập nhật")
    
    # Khung hiển thị tóm tắt bản tin đang chọn
    render_html(f"""
    <div class="sample-container" style="border-left: 5px solid #C96A4B; margin-top: 14px; margin-bottom: 16px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
            <span style="font-size:12.5px; font-weight:700; color:#8C4A32; text-transform:uppercase; letter-spacing:0.04em;">
                Nguồn: {source_in} | {pub_in}
            </span>
        </div>
        <div style="font-size: 16px; font-weight: 700; color: #2E2620; margin-bottom: 6px; line-height: 1.4;">
            {title_in}
        </div>
        <div style="font-size: 14px; color: #4A4038; line-height: 1.6;">
            {desc_in if desc_in else 'Không có mô tả chi tiết kèm theo bản tin.'}
        </div>
    </div>
    """)
    
    # Chạy phán đoán qua mô hình
    full_live_text = format_input_text(title_in, desc_in)
    res_live = predict_text(full_live_text, resources)
    champ_pred = res_live["finbert"]["pred_label"]
    champ_conf = res_live["finbert"]["confidence"] * 100
    pred_color = LABEL_COLORS[champ_pred]
    
    # Hiển thị kết quả phán đoán đơn giản, trực quan
    st.markdown("#### KẾT QUẢ PHÁN ĐOÁN MỨC ĐỘ TÁC ĐỘNG")
    
    c_res1, c_res2, c_res3 = st.columns(3)
    with c_res1:
        impact_desc = "Tác động mạnh / Rủi ro cao" if champ_pred == "HIGH" else ("Tác động trung bình" if champ_pred == "MEDIUM" else "Tác động thấp / Ổn định")
        render_html(f"""
        <div class="kpi-card kpi-card-champion" style="min-height: 160px; padding: 18px 20px;">
            <div style="font-size: 11.5px; font-weight: 700; color: #8C4A32; text-transform: uppercase; letter-spacing: 0.04em;">
                PHÁN ĐOÁN CHÍNH (FINBERT)
            </div>
            <div style="font-size: 26px; font-weight: 800; color: {pred_color}; font-family: 'JetBrains Mono', monospace; margin: 6px 0;">
                {champ_pred} ({champ_conf:.2f}%)
            </div>
            <div style="font-size: 12.5px; color: #4A4038; margin-top: auto;">
                Mức độ: <b>{impact_desc}</b>
            </div>
        </div>
        """)
        
    with c_res2:
        bilstm_pred = res_live["bilstm"]["pred_label"]
        bilstm_conf = res_live["bilstm"]["confidence"] * 100
        render_html(f"""
        <div class="kpi-card" style="min-height: 160px; padding: 18px 20px;">
            <div style="font-size: 11.5px; font-weight: 700; color: #7A7067; text-transform: uppercase; letter-spacing: 0.04em;">
                ĐỐI CHIẾU BI-LSTM (BASELINE)
            </div>
            <div style="font-size: 24px; font-weight: 700; color: {LABEL_COLORS[bilstm_pred]}; font-family: 'JetBrains Mono', monospace; margin: 6px 0;">
                {bilstm_pred} ({bilstm_conf:.2f}%)
            </div>
            <div style="font-size: 12px; color: #7A7067; margin-top: auto;">
                Mô hình chuỗi tuần tự RNN
            </div>
        </div>
        """)
        
    with c_res3:
        bert_pred = res_live["bert_base"]["pred_label"]
        bert_conf = res_live["bert_base"]["confidence"] * 100
        render_html(f"""
        <div class="kpi-card" style="min-height: 160px; padding: 18px 20px;">
            <div style="font-size: 11.5px; font-weight: 700; color: #7A7067; text-transform: uppercase; letter-spacing: 0.04em;">
                ĐỐI CHIẾU BERT-BASE
            </div>
            <div style="font-size: 24px; font-weight: 700; color: {LABEL_COLORS[bert_pred]}; font-family: 'JetBrains Mono', monospace; margin: 6px 0;">
                {bert_pred} ({bert_conf:.2f}%)
            </div>
            <div style="font-size: 12px; color: #7A7067; margin-top: auto;">
                Kiến trúc Transformer tổng quát
            </div>
        </div>
        """)
        
    # Thanh phân phối xác suất
    render_probability_distribution_view(res_live, "rt_prob_bar_chart")
    
    # Cho phép nhập tin tùy chỉnh nếu muốn thử nghiệm thêm
    with st.expander("Hoặc kiểm tra bản tin tài chính tùy chỉnh (Nhập văn bản thủ công)", expanded=False):
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            if st.button("Mẫu tin: Fed tăng lãi suất (HIGH)", key="rt_btn_preset_1"):
                st.session_state.custom_t = "Federal Reserve unexpectedly hikes interest rates by 75 basis points amid inflation surge"
                st.session_state.custom_d = "Markets tumble as the central bank signals aggressive monetary tightening to cool down the overheated economy."
        with c_p2:
            if st.button("Mẫu tin: BCTC đi ngang (MEDIUM)", key="rt_btn_preset_2"):
                st.session_state.custom_t = "Quarterly corporate earnings remain largely flat for small-cap manufacturing firms"
                st.session_state.custom_d = "Operating margins saw minimal change compared to the previous financial quarter with modest order backlog."
        with c_p3:
            if st.button("Mẫu tin: Lịch họp ĐHCĐ (LOW)", key="rt_btn_preset_3"):
                st.session_state.custom_t = "Company schedules annual general meeting for next month to approve routine dividend distribution"
                st.session_state.custom_d = "The board will discuss regular administrative procedures, proxy voting documentation, and routine statutory filings."
                
        c_title = st.text_input("Tiêu đề bản tin tùy chỉnh:", value=st.session_state.get("custom_t", ""), placeholder="Nhập tiêu đề tiếng Anh...", key="rt_c_title")
        c_desc = st.text_area("Tóm tắt nội dung:", value=st.session_state.get("custom_d", ""), placeholder="Nhập tóm tắt bản tin...", height=70, key="rt_c_desc")
        
        if c_title.strip() or c_desc.strip():
            c_text = format_input_text(c_title, c_desc)
            c_res = predict_text(c_text, resources)
            c_champ = c_res["finbert"]["pred_label"]
            c_conf = c_res["finbert"]["confidence"] * 100
            st.success(f"Phán đoán cho tin tùy chỉnh: **{c_champ}** (Độ tin cậy: {c_conf:.2f}%)")
            render_probability_distribution_view(c_res, "rt_custom_prob_bars")

# ==============================================================================
# PHẦN 3: ĐƯỜNG ỐNG TIỀN XỬ LÝ NLP & KHÔNG GIAN BIỂU DIỄN 3D
# ==============================================================================
with tab_pipeline:
    st.markdown("### ĐƯỜNG ỐNG TIỀN XỬ LÝ NLP & KHÔNG GIAN 3D")
    st.markdown("""
    Trực quan hóa chi tiết 4 trạm xử lý trong kiến trúc học máy: Từ văn bản thô chưa chuẩn hóa, quy trình làm sạch chống rò rỉ dữ liệu (Anti-Leakage), phân tách token, trích xuất ma trận Self-Attention đến không gian biểu diễn đa chiều.
    """)
    
    demo_samples = [
        "Sensex crashes 1,150 pts on rising crude prices; Nifty holds 22,000 amid Middle East tensions https://reuters.com/market-update",
        "Quarterly corporate earnings remain largely flat for small-cap manufacturing firms with modest order backlog. <i>Source: PTI</i>",
        "Company schedules annual general meeting for next month to approve dividend distribution and routine proxy voting.",
        "Breaking: Federal Reserve unexpectedly hikes policy rate by 50 bps amid persistent inflation shock wave."
    ]
    demo_selected = st.selectbox(
        "Chọn câu văn mẫu để giải phẫu qua từng trạm xử lý:",
        options=demo_samples,
        index=0,
        key="pl_select_demo_sample"
    )
    
    res_demo = predict_text(demo_selected, resources)
    clean_info = inspect_cleaning_pipeline(demo_selected)
    
    st1, st2, st3, st4 = st.tabs([
        "TRẠM 1: KHỬ RÒ RỈ DỮ LIỆU (ANTI-LEAKAGE)",
        "TRẠM 2: PHÂN TÁCH TOKEN & MA TRẬN HÓA",
        "TRẠM 3: CƠ CHẾ TỰ CHÚ Ý (SELF-ATTENTION)",
        "TRẠM 4: KHÔNG GIAN BIỂU DIỄN 3D"
    ])
    
    # TRẠM 1
    with st1:
        st.markdown("#### 1. TIỀN XỬ LÝ CHUẨN HÓA & KHỬ RÒ RỈ DỮ LIỆU (ANTI-LEAKAGE PIPELINE)")
        c_raw, c_clean = st.columns(2)
        with c_raw:
            st.markdown("##### Văn bản gốc (Raw Input):")
            render_html(f'<div class="academic-text-box">{clean_info["raw"]}</div>')
            st.caption(f"Độ dài: {clean_info['raw_length']} ký tự.")
        with c_clean:
            st.markdown("##### Văn bản sau chuẩn hóa (Cleaned Text):")
            render_html(f'<div class="academic-text-box" style="background:#FFFFFF; border-color:#C96A4B;">{clean_info["cleaned"]}</div>')
            st.caption(f"Độ dài sau làm sạch: {clean_info['cleaned_length']} ký tự.")
            
        st.markdown("##### Các thành phần rò rỉ bị loại bỏ triệt để:")
        col_u, col_h = st.columns(2)
        with col_u:
            if clean_info["urls_removed"]:
                st.warning(f"Đã phát hiện và loại bỏ {len(clean_info['urls_removed'])} liên kết URL: {clean_info['urls_removed']}")
            else:
                st.success("Không phát hiện liên kết URL rò rỉ.")
        with col_h:
            if clean_info["html_tags_removed"]:
                st.warning(f"Đã loại bỏ {len(clean_info['html_tags_removed'])} thẻ HTML: {clean_info['html_tags_removed']}")
            else:
                st.success("Không phát hiện thẻ HTML rò rỉ.")
                
        st.markdown("""
        **Quy tắc phòng ngừa Data Leakage:**
        - **Khử URL & Domain:** Loại bỏ các chuỗi dạng `http://`, `https://`, `www.` để mô hình không học lén độ uy tín của tên miền xuất bản.
        - **Khử Thẻ HTML:** Loại bỏ toàn bộ `<i>`, `<b>`, `<p>`, `<a>` phát sinh trong quá trình thu thập Web Scraper.
        - **Loại bỏ Metadata:** Loại bỏ hoàn toàn các trường dữ liệu phụ (`categories`, `relevance_score`, `matched_keywords`) để đảm bảo tính khách quan và thuần túy ngữ nghĩa văn bản.
        """)

    # TRẠM 2
    with st2:
        st.markdown("#### 2. PHÂN TÁCH TOKEN & MA TRẬN HÓA (MAX_LENGTH = 64 TOKENS)")
        st.markdown("""
        Độ dài chuỗi chuẩn hóa cố định ở `max_length = 64` tokens (bao phủ >99.8% độ dài bản tin thực nghiệm).
        """)
        c_lstm_tok, c_bert_tok = st.columns(2)
        with c_lstm_tok:
            st.markdown("##### A. Bi-LSTM Tokenizer (Word-level Regex + 20,000 Vocab):")
            st.markdown(f"""
            - **Cơ chế:** Tách từ nguyên gốc bằng biểu thức chính quy `r'\\b[a-zA-Z0-9_\\$%-]+\\b'`.
            - **Từ điển:** 20,000 từ. Token đặc biệt: `<pad>: 0`, `<unk>: 1`.
            - **Số token thực tế:** {res_demo['bilstm']['padded_len']} từ (phần còn lại điền `<pad>: 0`).
            """)
            token_preview = f"{res_demo['bilstm']['token_ids'][:25]} ... [0, 0, 0]"
            render_html(f'<div class="academic-text-box">{token_preview}</div>')
        with c_bert_tok:
            st.markdown("##### B. FinBERT Tokenizer (WordPiece Subword):")
            fin_toks = [t for t in res_demo["finbert"]["tokens"] if t != "[PAD]"]
            st.markdown(f"""
            - **Cơ chế:** Phân rã từ con Subword (`##`) giúp triệt tiêu hoàn toàn hiện tượng từ ngoài từ điển (Out-Of-Vocabulary - OOV).
            - **Token đặc biệt:** `[CLS]` (Classification token ở đầu câu), `[SEP]` (Phân tách cuối câu), `[PAD]`.
            - **Số token sau phân rã:** {len(fin_toks)} tokens.
            """)
            subword_preview = f"{fin_toks[:18]} ... [SEP]"
            render_html(f'<div class="academic-text-box">{subword_preview}</div>')

    # TRẠM 3
    with st3:
        st.markdown("#### 3. CƠ CHẾ TỰ CHÚ Ý (SELF-ATTENTION) & HEATMAP TRỌNG SỐ TẦNG 11")
        st.markdown("""
        Trích xuất trực tiếp **Attention Weights** từ tầng cuối cùng (Layer 11) của FinBERT từ token `[CLS]` tới mọi từ trong câu.
        Cơ chế Self-Attention thể hiện chính xác từ ngữ tài chính nào đang dẫn dắt quyết định phân loại rủi ro:
        """)
        
        # Render Heatmap HTML
        render_html(render_attention_heatmap_html(res_demo["finbert"]["word_attention"]))
        
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        # Bảng Top Attended Keywords
        top_words = sorted(res_demo["finbert"]["word_attention"], key=lambda x: x[1], reverse=True)[:8]
        col_t1, col_t2 = st.columns([1.5, 2.5])
        with col_t1:
            st.markdown("##### Top Từ Khóa Tập Trung Chú Ý (Highest Attention):")
            top_df = pd.DataFrame(top_words, columns=["Từ Khóa Tài Chính", "Trọng số Attention (0.0 - 1.0)"])
            top_df["Trọng số Attention (0.0 - 1.0)"] = top_df["Trọng số Attention (0.0 - 1.0)"].map(lambda x: f"{x:.4f}")
            st.dataframe(top_df, use_container_width=True)
        with col_t2:
            st.markdown("##### Đối chiếu Cơ chế Tính toán:")
            st.markdown("""
            - **Bi-LSTM (Tuần tự):** Duyệt từng bước qua $\\overrightarrow{h_t}$ và $\\overleftarrow{h_t}$. Toàn bộ ngữ nghĩa bị nén cưỡng bức vào 1 vector ẩn cuối cùng $[h_{-2}, h_{-1}]$, gây nút thắt cổ chai và tạo ra **343 ca báo động giả**.
            - **Transformer Multi-Head Attention (Toàn cục):** Tính ma trận tương quan giữa mọi cặp từ theo công thức:
              $$Attention(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$
              Giúp nhận diện tức thì các từ khóa bước ngoặt vĩ mô (*inflation, interest, rate, hike, plunge*) mà không bị suy giảm theo chiều dài câu.
            """)

    # TRẠM 4
    with st4:
        st.markdown("#### 4. HOẠT ẢNH KHÔNG GIAN VECTOR 3D (3D LATENT EMBEDDING SPACE)")
        st.markdown("""
        Vector biểu diễn đa chiều (768 chiều từ `[CLS]`) được chiếu vào không gian 3D tương đối theo tâm 3 cụm rủi ro:
        - **Vùng LOW (Xanh rêu):** Tác động thấp, ổn định.
        - **Vùng MEDIUM (Vàng nghệ):** Tác động trung bình.
        - **Vùng HIGH (Terracotta):** Tác động mạnh / Rủi ro cao.
        """)
        st.plotly_chart(render_3d_latent_space(
            res_demo["coords_3d"],
            res_demo["finbert"]["pred_label"],
            res_demo["cluster_centers"],
            sample_title=demo_selected[:35]
        ), use_container_width=True, key="pl_3d_latent_chart")

# ==============================================================================
# PHẦN 4: BENCHMARK TỔNG THỂ & BÁO CÁO KỸ THUẬT
# ==============================================================================
with tab_benchmark:
    st.markdown("### TỔNG HỢP HIỆU NĂNG THỰC NGHIỆM & KẾT LUẬN NGHIÊN CỨU")
    st.markdown("""
    Đối sánh khoa học toàn diện giữa 3 mô hình trên tập **Validation (N = 13,992 mẫu)** và kiểm định độc lập của **FinBERT Champion** trên tập **Test (N = 13,992 mẫu)**.
    """)
    
    t_v, t_t, t_c = st.tabs([
        "1. ĐÁNH GIÁ LỰA CHỌN MÔ HÌNH (VALIDATION N=13,992)",
        "2. KIỂM ĐỊNH ĐỘC LẬP TEST SET (N=13,992)",
        "3. ĐÚC KẾT HỌC MÁY & Ý NGHĨA THỰC TIỄN"
    ])
    
    with t_v:
        st.markdown("#### Bảng So sánh Hiệu năng Thực nghiệm Đa chiều (Model Selection Benchmark)")
        val_d = BENCHMARK_METRICS["validation"]
        rows = []
        for m, d in val_d.items():
            rows.append({
                "Mô hình": m,
                "Accuracy": f"{d['Accuracy']*100:.2f}%",
                "Macro Precision": f"{d['Macro Precision']*100:.2f}%",
                "Macro Recall": f"{d['Macro Recall']*100:.2f}%",
                "Macro F1-Score": f"{d['Macro F1']*100:.2f}%",
                "HIGH Recall (Bắt rủi ro)": f"{d['HIGH Recall']*100:.2f}%",
                "HIGH Precision (Chuẩn xác rủi ro)": f"{d['HIGH Precision']*100:.2f}%",
                "Báo động giả HIGH": f"{d['False Alarms HIGH']} ca / {d['False Alarms Total']} dự đoán"
            })
        st.dataframe(pd.DataFrame(rows).set_index("Mô hình"), use_container_width=True)
        
        st.plotly_chart(render_benchmark_comparison_chart(BENCHMARK_METRICS), use_container_width=True, key="bm_bar_comparison_chart")
        st.plotly_chart(render_confusion_matrices(BENCHMARK_METRICS["confusion_matrices_val"], split_title="TẬP VALIDATION (N = 13,992)"), use_container_width=True, key="bm_cm_val_chart")
        
    with t_t:
        st.markdown("#### Kiểm Định Độc Lập Mô Hình Tối Ưu (FinBERT Champion) trên Tập Test (N = 13,992)")
        t_data = BENCHMARK_METRICS["test_finbert"]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", f"{t_data['Accuracy']*100:.2f}%")
        c2.metric("Test Macro F1", f"{t_data['Macro F1']*100:.2f}%")
        c3.metric("Test HIGH Precision", f"{t_data['HIGH Precision']*100:.2f}%")
        c4.metric("Test HIGH Recall", f"{t_data['HIGH Recall']*100:.2f}%")
        
        st.markdown("""
        | Phân lớp Tác động | Test Precision | Test Recall | Test F1-Score | Số lượng mẫu kiểm thử (Support) |
        | :--- | :---: | :---: | :---: | :---: |
        | **LOW (Tác động thấp)** | 98.51% | 99.37% | 98.94% | 5,737 |
        | **MEDIUM (Tác động trung bình)** | 99.33% | 98.81% | 99.07% | 7,951 |
        | **HIGH (Tác động cao / Rủi ro)** | **96.96%** | **94.41%** | **95.67%** | 304 |
        | **Trung bình Macro (Macro Avg)** | **98.27%** | **97.53%** | **97.89%** | **13,992** |
        """)
        
        test_cm_single = {"FinBERT (Independent Test Champion)": BENCHMARK_METRICS["confusion_matrix_test_finbert"]}
        st.plotly_chart(render_confusion_matrices(test_cm_single, split_title="TẬP TEST ĐỘC LẬP (N = 13,992)"), use_container_width=True, key="bm_cm_test_chart")
        
    with t_c:
        st.markdown("#### ĐÚC KẾT KỸ THUẬT HỌC MÁY & Ý NGHĨA QUẢN TRỊ ĐẦU TƯ")
        st.markdown("""
        1. **Vì sao FinBERT là Champion Model trong môi trường FinTech thực tế?**
           - Trong giao dịch tài chính định lượng, **chi phí của một cảnh báo giả (False Positive) là cực kỳ tốn kém** (phí phòng hộ Hedging, trượt giá slippage, bán tháo sai danh mục).
           - FinBERT đạt độ chuẩn xác lớp HIGH lên tới **94.46% (Val) và 96.96% (Test)**. Đặc biệt, **0 mẫu tin LOW nào bị FinBERT dự báo nhầm sang HIGH**, triệt tiêu gần như hoàn toàn tín hiệu nhiễu.
        2. **Hạn chế cố hữu của kiến trúc mạng hồi quy tuần tự (Bi-LSTM):**
           - Mặc dù Accuracy tổng thể đạt 93.66%, nhưng ở nhóm thiểu số cực đoan (HIGH - tỷ lệ mất cân bằng 1:26), Bi-LSTM tạo ra tới **343 ca báo động giả** (Precision chỉ đạt 41.67%). Việc phụ thuộc vào Bi-LSTM trong hệ thống tự động sẽ làm tê liệt quỹ đầu tư bởi các lệnh phòng hộ không cần thiết.
        3. **Vai trò của mô hình BERT-base:**
           - Đạt Recall lớp HIGH cao nhất (93.77%), phù hợp cho các cơ quan điều hành chính sách vĩ mô có xu hướng "thà cảnh báo nhầm còn hơn bỏ sót".
        """)

# ==============================================================================
# FOOTER HỌC THUẬT
# ==============================================================================
render_html("""
<div style="text-align:center; padding: 42px 0 32px 0; border-top: 1px solid #D9CDC1; margin-top: 60px; font-family:'Be Vietnam Pro', sans-serif;">
    <div style="color: #4A4038; font-weight: 600; font-size: 13.5px; line-height: 1.6;">
        Đề tài Nghiên cứu Machine Learning: Nhóm 9 | Khoa Tài chính, Trường ĐH Kinh tế, ĐH Đà Nẵng | GVHD: PGS.TS. Nguyễn Văn Chức
    </div>
    <div style="color: #8C8075; font-size: 12px; margin-top: 8px; line-height: 1.6;">
        Đối sánh thực nghiệm giữa Bi-LSTM (Baseline), BERT-base và FinBERT (Champion) | Triển khai bằng Streamlit & PyTorch
    </div>
</div>
""")
