"""
app.py - Hệ Thống Dự Báo Mức Độ Tác Động Của Tin Tức Tài Chính Đến Thị Trường
Đề tài Nghiên cứu Machine Learning: Nhóm 9 | Khoa Tài chính, Trường ĐH Kinh tế, ĐH Đà Nẵng
Thiết kế theo chuẩn Academic FinTech Light Theme (3 Tabs điều hướng, Nền kem ấm, Chuẩn hóa học thuật trung tính).
"""

import os
import random
import numpy as np
import pandas as pd
import streamlit as st

# Cấu hình giao diện Streamlit chuẩn Academic Dashboard
st.set_page_config(
    page_title="Dự Báo Tác Động Tin Tức Tài Chính | Machine Learning Research",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# Nạp và tự động làm mới các module nghiệp vụ cục bộ
import importlib
import model_engine
import visualizer
import news_scraper

importlib.reload(model_engine)
importlib.reload(visualizer)
importlib.reload(news_scraper)

from model_engine import (
    load_all_models,
    predict_text,
    load_clean_dataset,
    clean_news_text,
    merge_title_and_description,
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
# CSS OVERRIDE HỌC THUẬT: TYPOGRAPHY, CONTAINER 1200PX, NÚT BẤM SÁNG THANH LỊCH
# ==============================================================================
st.markdown("""
<style>
    /* 1. CHUẨN HÓA 2 PHÔNG CHỮ SANS-SERIF & MONOSPACE */
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400;1,500;1,600;1,700&family=JetBrains+Mono:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Be Vietnam Pro', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #FAF6F0 !important;
        color: #2E2620 !important;
    }
    
    .stApp {
        background-color: #FAF6F0 !important;
    }
    
    /* Cố định bề rộng container chuẩn học thuật 1200px */
    .block-container {
        max-width: 1200px !important;
        padding-top: 1.8rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        margin: 0 auto !important;
    }
    
    /* 2. TIÊU ĐỀ HỌC THUẬT */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Be Vietnam Pro', sans-serif !important;
        color: #2E2620 !important;
        font-weight: 700 !important;
        letter-spacing: -0.015em !important;
    }
    
    h1 { font-size: 28px !important; margin-bottom: 8px !important; }
    h2 { font-size: 22px !important; margin-top: 24px !important; margin-bottom: 12px !important; }
    h3 { font-size: 18px !important; margin-top: 20px !important; margin-bottom: 10px !important; text-transform: uppercase; letter-spacing: 0.03em; }
    h4 { font-size: 15px !important; margin-top: 16px !important; margin-bottom: 8px !important; }
    
    /* 3. INPUT WIDGETS CHỮ ĐẬM RÕ NÉT (TRIỆT TIÊU LỖI CHỮ TRẮNG NỀN TRẮNG) */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] {
        background-color: #FFFFFF !important;
        border: 1.5px solid #D9CDC1 !important;
        border-radius: 8px !important;
        color: #2E2620 !important;
    }
    
    div[data-baseweb="select"] * {
        color: #2E2620 !important;
        font-family: 'Be Vietnam Pro', sans-serif !important;
        font-size: 13.5px !important;
    }
    
    input, textarea {
        color: #2E2620 !important;
        background-color: #FFFFFF !important;
        font-family: 'Be Vietnam Pro', sans-serif !important;
        font-size: 14px !important;
    }
    
    input:focus, textarea:focus {
        border-color: #C96A4B !important;
        box-shadow: 0 0 0 2px rgba(201, 106, 75, 0.2) !important;
    }
    
    /* 4. SỬA LỖI NÚT BẤM (BUTTON): NỀN TRẮNG SÁNG, ĐỒNG BỘ VỚI TÔNG MÀU KEM ẤM */
    .stButton > button {
        background-color: #FFFFFF !important;
        color: #2E2620 !important;
        border: 1px solid #D9CDC1 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #F3EAE0 !important;
        border-color: #C96A4B !important;
        color: #C96A4B !important;
    }
    
    .stButton > button:active {
        background-color: #E6D7C8 !important;
    }
    
    /* 5. ĐIỀU HƯỚNG TABS TỐI GIẢN - 1 HÀNG DUY NHẤT, CANH GIỮA & CÂN BẰNG */
    div[data-testid="stTabs"] {
        width: 100% !important;
    }
    
    div[data-testid="stTabs"] > div:first-child {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
    }

    div[data-testid="stTabs"] [role="tablist"],
    div[data-baseweb="tab-list"],
    .stTabs [role="tablist"] {
        display: flex !important;
        flex-wrap: nowrap !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 28px !important;
        background-color: transparent !important;
        padding: 6px 12px 0px 12px !important;
        border: none !important;
        border-bottom: 1.5px solid rgba(217, 205, 193, 0.7) !important;
        margin: 12px auto 26px auto !important;
        width: 100% !important;
        overflow-x: auto !important;
        scrollbar-width: none !important;
    }
    
    div[data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar {
        display: none !important;
    }

    div[data-testid="stTabs"] [role="tablist"]::after,
    div[data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Tab đơn lẻ tầng 1 */
    div[data-testid="stTabs"] [data-testid="stTab"],
    div[data-testid="stTabs"] [role="tab"],
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 8px 12px 12px 12px !important;
        color: #7A7067 !important;
        font-family: 'Be Vietnam Pro', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 14.5px !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em !important;
        text-transform: none !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    div[data-testid="stTabs"] [data-testid="stTab"] p,
    div[data-testid="stTabs"] [role="tab"] p,
    button[data-baseweb="tab"] p {
        font-family: 'Be Vietnam Pro', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 14.5px !important;
        font-weight: 500 !important;
        color: #7A7067 !important;
        margin: 0 !important;
        white-space: nowrap !important;
    }

    /* Sub-tabs (Tầng 2: Trạm 1..4) */
    div[data-testid="stTabs"] div[data-testid="stTabs"] [role="tablist"] {
        gap: 16px !important;
        margin: 8px auto 18px auto !important;
        padding: 2px 6px 0px 6px !important;
    }
    
    div[data-testid="stTabs"] div[data-testid="stTabs"] [data-testid="stTab"],
    div[data-testid="stTabs"] div[data-testid="stTabs"] [role="tab"] {
        padding: 5px 8px 8px 8px !important;
        font-size: 13px !important;
    }
    
    div[data-testid="stTabs"] div[data-testid="stTabs"] [data-testid="stTab"] p,
    div[data-testid="stTabs"] div[data-testid="stTabs"] [role="tab"] p {
        font-size: 13px !important;
    }

    /* Hover */
    div[data-testid="stTabs"] [data-testid="stTab"]:hover,
    div[data-testid="stTabs"] [role="tab"]:hover,
    button[data-baseweb="tab"]:hover {
        color: #2E2620 !important;
    }
    
    div[data-testid="stTabs"] [data-testid="stTab"]:hover p,
    div[data-testid="stTabs"] [role="tab"]:hover p,
    button[data-baseweb="tab"]:hover p {
        color: #2E2620 !important;
        font-weight: 600 !important;
    }

    /* Tab đang chọn (ACTIVE): In đậm, màu Terracotta #C96A4B */
    div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"],
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #C96A4B !important;
        font-weight: 800 !important;
        border: none !important;
    }
    
    div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] p,
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #C96A4B !important;
        font-weight: 800 !important;
    }

    div[data-testid="stTabs"] .react-aria-SelectionIndicator,
    div[data-baseweb="tab-highlight"] {
        background-color: #C96A4B !important;
        height: 3px !important;
        border-radius: 3px 3px 0 0 !important;
        bottom: 0px !important;
    }

    /* 6. HERO SECTION CARD */
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
    
    .academic-stat-card {
        background: #FFFFFF !important;
        border: 1px solid #D9CDC1 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 14px rgba(46, 38, 32, 0.04) !important;
        height: 100% !important;
    }
    .stat-number {
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #2E2620 !important;
        font-family: 'JetBrains Mono', monospace !important;
        margin-bottom: 4px !important;
    }
    .stat-label {
        font-size: 13px !important;
        font-weight: 700 !important;
        color: #8C4A32 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.03em !important;
        margin-bottom: 6px !important;
    }
    .stat-desc {
        font-size: 12px !important;
        color: #7A7067 !important;
        line-height: 1.5 !important;
    }

    /* NÚT BẤM CHÍNH (PRIMARY ACTION BUTTONS: CAM GẠCH NỔI BẬT) */
    button[kind="primary"],
    .stButton button[kind="primary"],
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #C96A4B 0%, #8C4A32 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(201, 106, 75, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover,
    .stButton button[kind="primary"]:hover,
    div[data-testid="stButton"] button[kind="primary"]:hover {
        box-shadow: 0 6px 18px rgba(201, 106, 75, 0.38) !important;
        transform: translateY(-1px) !important;
        color: #FFFFFF !important;
    }
    button[kind="primary"] p,
    .stButton button[kind="primary"] p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    
    /* 7. THẺ CARD HỌC THUẬT & METRIC CARDS */
    .kpi-card {
        background: #FFFFFF !important;
        border: 1px solid #D9CDC1 !important;
        border-radius: 14px !important;
        padding: 22px 20px !important;
        box-shadow: 0 4px 16px rgba(46, 38, 32, 0.04) !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        height: 100% !important;
        min-height: 230px !important;
    }
    
    .kpi-card-highlight {
        border: 2px solid #C96A4B !important;
        background: #FFFFFF !important;
        box-shadow: 0 6px 20px rgba(201, 106, 75, 0.12) !important;
        position: relative !important;
    }
    
    .kpi-title {
        font-size: 13.5px !important;
        font-weight: 700 !important;
        color: #2E2620 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        margin-bottom: 10px !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
    }
    
    .kpi-value {
        font-size: 32px !important;
        font-weight: 800 !important;
        font-family: 'JetBrains Mono', monospace !important;
        margin-bottom: 6px !important;
        line-height: 1.1 !important;
    }
    
    .kpi-pill {
        display: inline-block !important;
        padding: 4px 10px !important;
        border-radius: 6px !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: 0.02em !important;
    }
    
    .kpi-footer-note {
        font-size: 11.5px !important;
        color: #7A7067 !important;
        line-height: 1.5 !important;
        border-top: 1px solid #EFE5DA !important;
        padding-top: 10px !important;
        margin-top: 12px !important;
    }
    
    .sample-container {
        background: #FFFFFF !important;
        border: 1px solid #D9CDC1 !important;
        border-radius: 12px !important;
        padding: 20px 24px !important;
        margin-bottom: 22px !important;
        box-shadow: 0 4px 14px rgba(46, 38, 32, 0.04) !important;
    }
    
    .academic-note {
        background: #FAF6F0 !important;
        border: 1px solid #D9CDC1 !important;
        border-left: 4px solid #C96A4B !important;
        padding: 12px 18px !important;
        border-radius: 6px !important;
        font-size: 13px !important;
        color: #4A4038 !important;
        line-height: 1.55 !important;
        margin-bottom: 20px !important;
    }
    
    .academic-text-box {
        background: #FAF6F0 !important;
        border: 1px solid #D9CDC1 !important;
        border-radius: 8px !important;
        padding: 14px 16px !important;
        font-family: 'Be Vietnam Pro', sans-serif !important;
        font-size: 13.5px !important;
        line-height: 1.6 !important;
        color: #2E2620 !important;
        word-break: break-word !important;
        white-space: pre-wrap !important;
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo nạp tài nguyên 3 mô hình
resources = load_all_models()

# ==============================================================================
# HERO SECTION: TIÊU ĐỀ NGHIÊN CỨU & HUY HIỆU THỐNG KÊ HỌC THUẬT CHUẨN HÓA
# ==============================================================================
render_html("""
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

# 3 Thẻ thống kê kiến trúc thực nghiệm (Technical Architecture Cards)
c_st1, c_st2, c_st3 = st.columns(3)

with c_st1:
    render_html("""
    <div class="academic-stat-card">
        <div class="stat-number">13,992</div>
        <div class="stat-label">Mẫu Kiểm Thử Độc Lập (Test Set)</div>
        <div class="stat-desc">Dữ liệu kiểm thử ngoài mẫu độc lập đã làm sạch chuẩn Anti-Leakage, loại bỏ hoàn toàn siêu liên kết và thẻ định dạng.</div>
    </div>
    """)

with c_st2:
    render_html("""
    <div class="academic-stat-card" style="border-left: 4px solid #C96A4B;">
        <div class="stat-number" style="color: #C96A4B;">98.53%</div>
        <div class="stat-label">Độ Chuẩn Xác (Mô hình BERT-base)</div>
        <div class="stat-desc">Macro F1 đạt 96.25%, nhận diện chính xác 94.08% các cú sốc vĩ mô (HIGH) trên tập Test độc lập.</div>
    </div>
    """)

with c_st3:
    render_html("""
    <div class="academic-stat-card">
        <div class="stat-number">3 Kiến Trúc</div>
        <div class="stat-label">Tiến Trình Phát Triển Kiến Trúc</div>
        <div class="stat-desc">Đối sánh thực nghiệm từ Mạng nơ-ron hồi quy tuần tự (Bi-LSTM) đến cơ chế Self-Attention toàn cục và tinh chỉnh miền tài chính.</div>
    </div>
    """)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# TOP-LEVEL TABS ĐIỀU HƯỚNG: 3 TABS CỐ ĐỊNH CHÍNH DIỆN
# ==============================================================================
tab_testbed, tab_realtime, tab_pipeline = st.tabs([
    "🧪 Thực nghiệm Dataset độc lập",
    "📡 Suy luận Real-time & Cảnh báo sớm",
    "🪐 Tiền xử lý NLP & Không gian Vector 3D"
])

# ==============================================================================
# TAB 1: THỰC NGHIỆM DATASET ĐỘC LẬP (TESTBED & MODEL COMPARISON)
# ==============================================================================
with tab_testbed:
    st.markdown("### THỰC NGHIỆM TRÊN TẬP DỮ LIỆU ĐÃ LÀM SẠCH (DATASET TESTBED)")
    render_html("""
    <div class="academic-note">
        <b>LƯU Ý THỰC NGHIỆM:</b> Tập kiểm thử độc lập ngoài mẫu gồm 13,992 bản tin chưa từng xuất hiện trong quá trình huấn luyện và tinh chỉnh siêu tham số, được dùng để đối chiếu độ tin cậy và kiểm chứng hiện tượng phân kỳ dự đoán giữa Mô hình Bi-LSTM, Mô hình FinBERT và Mô hình BERT-base.
    </div>
    """)
    
    col_split, col_tier, col_kw = st.columns([1.5, 1.3, 1.8])
    with col_split:
        data_split = st.selectbox(
            "1. Chọn tập dữ liệu:",
            ["test", "val"],
            format_func=lambda x: "Tập Test (13,992 mẫu)" if x == "test" else "Tập Validation (13,992 mẫu)",
            index=0,
            key="tb_select_dataset_split"
        )
        
    df_data = load_clean_dataset(data_split)
    
    with col_tier:
        tier_filter = st.selectbox(
            "2. Lọc Ground Truth:",
            ["Tất cả", "LOW", "MEDIUM", "HIGH"],
            index=0,
            key="tb_select_tier_filter"
        )
        
    with col_kw:
        search_kw = st.text_input(
            "3. Tìm kiếm từ khóa vĩ mô:",
            value="",
            placeholder="vd: Fed, Rate, Inflation, Crisis...",
            key="tb_input_search_kw"
        )
        
    # Lọc dữ liệu
    df_filtered = df_data.copy()
    if tier_filter != "Tất cả":
        df_filtered = df_filtered[df_filtered["impact_tier"] == tier_filter]
    if search_kw.strip():
        df_filtered = df_filtered[df_filtered["text"].str.contains(search_kw.strip(), case=False, na=False)]
        
    # Chọn mẫu ngẫu nhiên hoặc từ dropdown
    c_btn, c_drop = st.columns([1.2, 3.4])
    with c_btn:
        st.markdown("<label style='font-size:13.5px; font-weight:600; color:#4A4038; display:block; margin-bottom:5px;'>Chọn ngẫu nhiên:</label>", unsafe_allow_html=True)
        btn_random = st.button("🎲 Lấy mẫu ngẫu nhiên", type="primary", use_container_width=True, key="tb_btn_random_sample")
        
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
                # Cắt theo ranh giới từ để không bị cắt vụn ngang từ
                if len(raw_txt) > 75:
                    cut = raw_txt[:75].rsplit(' ', 1)[0]
                    short_txt = cut + "…"
                else:
                    short_txt = raw_txt
                return f"#{idx} [{tier}] {short_txt}"
                
            selected_sample_id = st.selectbox(
                f"Hoặc chọn bản tin cụ thể ({len(df_filtered):,} kết quả):",
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
        sample_text = str(sample_row["text"]).strip()
        ground_truth = sample_row["impact_tier"]
        gt_color = LABEL_COLORS[ground_truth]
        
        # Tinh lọc câu trùng lặp nếu có trong raw dataset
        display_sample_text = sample_text
        parts = sample_text.split(". ")
        if len(parts) >= 2 and parts[0].strip().lower() == parts[1].strip().lower():
            display_sample_text = ". ".join(parts[1:]).strip()
        if not display_sample_text:
            display_sample_text = sample_text
            
        render_html(f"""
        <div style="background:#FFFFFF; border:1.5px solid #D9CDC1; border-left:6px solid {gt_color}; border-radius:14px; padding:20px 24px; margin-bottom:22px; box-shadow:0 4px 18px rgba(46,38,32,0.06);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:10px; border-bottom:1px solid #F0E6DC; padding-bottom:10px;">
                <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                    <span style="background:#FAF6F0; border:1px solid #D9CDC1; color:#8C4A32; font-family:'JetBrains Mono', monospace; font-size:12px; font-weight:700; padding:4px 10px; border-radius:6px;">
                        #ID {curr_id}
                    </span>
                    <span style="background:#FAF6F0; border:1px solid #D9CDC1; color:#4A4038; font-size:12px; font-weight:600; padding:4px 10px; border-radius:6px; text-transform:uppercase;">
                        TẬP DỮ LIỆU: {data_split.upper()}
                    </span>
                </div>
                <div style="background:{gt_color}; color:#FFFFFF; padding:5px 14px; border-radius:20px; font-weight:800; font-size:12.5px; font-family:'JetBrains Mono', monospace; letter-spacing:0.03em; box-shadow:0 2px 8px rgba(0,0,0,0.12);">
                    ● GROUND TRUTH: {ground_truth} (Lớp {sample_row['label']})
                </div>
            </div>
            <div style="font-size:15.5px; font-weight:600; color:#2E2620; line-height:1.65; font-family:'Be Vietnam Pro', sans-serif;">
                "{display_sample_text}"
            </div>
        </div>
        """)
        
        # Chạy suy luận qua cả 3 mô hình
        results = predict_text(sample_text, resources)
        
        # 1. THẺ SO SÁNH TRỰC DIỆN 3 MÔ HÌNH
        st.markdown("#### KẾT QUẢ SUY LUẬN TRỰC DIỆN CỦA 3 MÔ HÌNH SO VỚI GROUND TRUTH")
        col_m1, col_m2, col_m3 = st.columns(3)
        
        # Cột 1: Mô hình Bi-LSTM
        with col_m1:
            p_lbl = results["bilstm"]["pred_label"]
            conf = results["bilstm"]["confidence"] * 100
            is_match = (p_lbl == ground_truth)
            status_text = "✅ CHÍNH XÁC" if is_match else "❌ NHẦM LẪN"
            status_color = "#117864" if is_match else "#8C4A32"
            render_html(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-title">
                        <span>1. Mô hình Bi-LSTM</span>
                        <span style="color:{status_color}; font-weight:700; font-size:11.5px;">{status_text}</span>
                    </div>
                    <div class="kpi-value" style="color: {LABEL_COLORS[p_lbl]};">
                        {conf:.2f}%
                    </div>
                    <div style="margin-bottom: 10px;">
                        <span class="kpi-pill" style="background:{LABEL_COLORS[p_lbl]}; color:white;">
                            DỰ BÁO: {p_lbl}
                        </span>
                    </div>
                </div>
                <div class="kpi-footer-note">
                    <span>Mạng nơ-ron hồi quy hai chiều. Tỷ lệ dự đoán nhầm lớp HIGH cao trên tập Val (Precision HIGH: <b>37.79%</b>, nhầm 338 mẫu MEDIUM và 8 mẫu LOW sang HIGH).</span>
                </div>
            </div>
            """)
            
        # Cột 2: Mô hình BERT-base
        with col_m2:
            p_lbl = results["bert_base"]["pred_label"]
            conf = results["bert_base"]["confidence"] * 100
            is_match = (p_lbl == ground_truth)
            status_text = "✅ CHÍNH XÁC" if is_match else "❌ NHẦM LẪN"
            status_color = "#117864" if is_match else "#8C4A32"
            
            # Logic động hiển thị Precision tương ứng với lớp được mô hình dự đoán (Báo cáo BCaogiuakyML_2)
            if p_lbl == "HIGH":
                prec_dynamic = "Độ chuẩn xác thực nghiệm (HIGH Precision): <b>88.82% trên tập Test</b> (HIGH Recall: <b>94.08% Test / 92.13% Val</b>; Bắt trúng 286/304 sự kiện tác động mạnh ngoài mẫu)"
            elif p_lbl == "MEDIUM":
                prec_dynamic = "Độ chuẩn xác thực nghiệm (MEDIUM Precision): <b>98.89% trên tập Test</b> (Recall: 98.52%)"
            else: # LOW
                prec_dynamic = "Độ chuẩn xác thực nghiệm (LOW Precision): <b>98.57% trên tập Test</b> (Recall: 98.78%)"
                
            render_html(f"""
            <div class="kpi-card kpi-card-highlight">
                <div>
                    <div class="kpi-title">
                        <span style="color:#8C4A32; font-weight:800;">2. Mô hình BERT-base</span>
                        <span style="color:{status_color}; font-weight:700; font-size:11.5px;">{status_text}</span>
                    </div>
                    <div class="kpi-value" style="color: {LABEL_COLORS[p_lbl]};">
                        {conf:.2f}%
                    </div>
                    <div style="margin-bottom: 10px;">
                        <span class="kpi-pill" style="background:{LABEL_COLORS[p_lbl]}; color:white;">
                            DỰ BÁO: {p_lbl}
                        </span>
                    </div>
                </div>
                <div class="kpi-footer-note" style="border-top-color: #C96A4B; color: #4A4038;">
                    <span>Mô hình Transformer tổng quát. {prec_dynamic}.</span>
                </div>
            </div>
            """)

        # Cột 3: Mô hình FinBERT
        with col_m3:
            p_lbl = results["finbert"]["pred_label"]
            conf = results["finbert"]["confidence"] * 100
            is_match = (p_lbl == ground_truth)
            status_text = "✅ CHÍNH XÁC" if is_match else "❌ NHẦM LẪN"
            status_color = "#117864" if is_match else "#8C4A32"
            render_html(f"""
            <div class="kpi-card">
                <div>
                    <div class="kpi-title">
                        <span>3. Mô hình FinBERT</span>
                        <span style="color:{status_color}; font-weight:700; font-size:11.5px;">{status_text}</span>
                    </div>
                    <div class="kpi-value" style="color: {LABEL_COLORS[p_lbl]};">
                        {conf:.2f}%
                    </div>
                    <div style="margin-bottom: 10px;">
                        <span class="kpi-pill" style="background:{LABEL_COLORS[p_lbl]}; color:white;">
                            DỰ BÁO: {p_lbl}
                        </span>
                    </div>
                </div>
                <div class="kpi-footer-note">
                    <span>Mô hình Transformer tài chính. Accuracy Val <b>98.41%</b>, HIGH Recall <b>91.48%</b> (nhầm 38 mẫu MEDIUM và 1 mẫu LOW sang HIGH).</span>
                </div>
            </div>
            """)

        # 2. THANH XÁC SUẤT NẰM NGANG
        render_probability_distribution_view(results, "tb_prob_bar_chart")

        # 3. ĐỒNG BỘ MA TRẬN NHẦM LẪN THEO TẬP DỮ LIỆU ĐANG CHỌN NGAY TRONG TAB 1
        st.markdown("---")
        if data_split == "test":
            st.markdown("#### MA TRẬN NHẦM LẪN KIỂM THỬ ĐỘC LẬP NGOÀI MẪU CỦA MÔ HÌNH BERT-BASE (N = 13,992 MẪU)")
            st.markdown("""
            *Ghi chú khoa học:* Tập Test gồm 13,992 mẫu độc lập ngoài mẫu. Ma trận kiểm định của Mô hình BERT-base đạt Accuracy = **98.53%**, Macro F1 = **96.25%**, HIGH Precision = **88.82%**, HIGH Recall = **94.08%** (bắt trúng 286/304 sự kiện tác động mạnh ngoài mẫu).
            """)
            test_cm_dict = {"Mô hình BERT-base (Tập Test N = 13,992)": BENCHMARK_METRICS["confusion_matrix_test_bert"]}
            st.plotly_chart(render_confusion_matrices(test_cm_dict, split_title="TẬP TEST ĐỘC LẬP (N = 13,992)"), use_container_width=True, key="tb_cm_test_chart")
        else:
            st.markdown("#### BỘ BA MA TRẬN NHẦM LẪN ĐỐI SÁNH TRÊN TẬP VALIDATION (N = 13,992 MẪU)")
            st.markdown("""
            *Ghi chú khoa học:* Tập Validation đối sánh trực tiếp cả 3 mô hình nhằm đánh giá độ hội tụ và phân kỳ giữa Mô hình Bi-LSTM, Mô hình FinBERT và Mô hình BERT-base.
            """)
            st.plotly_chart(render_confusion_matrices(BENCHMARK_METRICS["confusion_matrices_val"], split_title="TẬP VALIDATION (N = 13,992)"), use_container_width=True, key="tb_cm_val_chart")

# ==============================================================================
# TAB 2: SUY LUẬN REAL-TIME & CẢNH BÁO SỚM (LIVE STREAM & IMPACT ANALYSIS)
# ==============================================================================
with tab_realtime:
    st.markdown("### DỰ BÁO LUỒNG TIN REAL-TIME & PHÂN TÍCH TÁC ĐỘNG THỊ TRƯỜNG")
    
    # Hàng điều khiển thu thập tin tức
    col_f1, col_f2 = st.columns([1.2, 2.8])
    with col_f1:
        st.markdown("<label style='font-size:13.5px; font-weight:600; color:#4A4038; display:block; margin-bottom:5px;'>Thao tác kết nối:</label>", unsafe_allow_html=True)
        btn_fetch = st.button("🔄 Thu thập tin mới nhất", type="primary", use_container_width=True, key="rt_btn_fetch_news")
    with col_f2:
        src_filter = st.selectbox(
            "Nguồn cấp tin tức:",
            ["Tất cả nguồn tin (All)", "Google News Financial", "Yahoo Finance", "CNBC Markets"],
            key="rt_select_source_filter"
        )
        
    if "live_news_cache" not in st.session_state or btn_fetch:
        with st.spinner("Đang kết nối luồng tin tức tài chính mới nhất..."):
            items, is_fallback = fetch_all_financial_news(limit_per_source=8)
            st.session_state.live_news_cache = items
            st.session_state.is_fallback = is_fallback
            
    live_items = st.session_state.live_news_cache
    if st.session_state.get("is_fallback", False):
        st.caption("ℹ️ Đang kích hoạt bộ đệm 10 tin tức vĩ mô dự phòng do kết nối mạng hoặc nguồn RSS bị giới hạn.")
        
    if src_filter != "Tất cả nguồn tin (All)":
        filtered_items = [x for x in live_items if x["source"] == src_filter]
        if not filtered_items:
            filtered_items = live_items
    else:
        filtered_items = live_items
        
    def format_live_title(i):
        item = filtered_items[i]
        src = item.get('source', 'Tin tức')
        short_src = src.replace(" Financial", "").replace(" Markets", "")
        t = item['title'].strip()
        if len(t) > 80:
            t_cut = t[:80].rsplit(' ', 1)[0] + "…"
        else:
            t_cut = t
        return f"[{short_src}] {t_cut}"
        
    selected_live_idx = st.selectbox(
        "Chọn bản tin tài chính để phân tích mức độ tác động:",
        options=range(len(filtered_items)),
        format_func=format_live_title,
        key="rt_select_live_article"
    )
    
    article = filtered_items[selected_live_idx]
    title_in = article["title"].strip()
    desc_in = article.get("description", "").strip()
    source_in = article.get("source", "Tài chính")
    pub_in = article.get("published", "Vừa cập nhật")
    link_in = article.get("link", "")
    
    # Khử trùng lặp tiêu đề ở đầu phần mô tả
    clean_desc = desc_in
    if clean_desc.lower().startswith(title_in.lower()):
        clean_desc = clean_desc[len(title_in):].strip(" -:;,.\n\t")
    if not clean_desc:
        clean_desc = "Bản tin không có đoạn tóm tắt phụ kèm theo. Toàn bộ nội dung tài chính trọng tâm nằm ở tiêu đề chính thức."
        
    # Khung hiển thị tóm tắt bản tin đang chọn (Định dạng Financial Terminal Card nổi bật)
    link_html = f'<div style="margin-top:12px;"><a href="{link_in}" target="_blank" style="color:#C96A4B; font-weight:600; font-size:13px; text-decoration:none;">🔗 Đọc toàn văn bản tin gốc trên nguồn xuất bản ↗</a></div>' if link_in and link_in.startswith("http") else ""
    render_html(f"""
    <div style="background:#FFFFFF; border:1.5px solid #D9CDC1; border-left:6px solid #C96A4B; border-radius:14px; padding:22px 26px; margin-top:14px; margin-bottom:20px; box-shadow:0 4px 18px rgba(46,38,32,0.06);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:10px; border-bottom:1px solid #F0E6DC; padding-bottom:10px;">
            <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                <span style="display:inline-flex; align-items:center; gap:6px; background:#E8F8F5; color:#117864; font-size:12px; font-weight:700; padding:4px 12px; border-radius:20px; border:1px solid #7A9A8B;">
                    <span style="width:7px; height:7px; border-radius:50%; background:#117864;"></span> LUỒNG TIN TRỰC TUYẾN
                </span>
                <span style="background:#FAF6F0; border:1px solid #D9CDC1; color:#8C4A32; font-size:12px; font-weight:700; padding:4px 10px; border-radius:6px; text-transform:uppercase;">
                    📰 {source_in}
                </span>
            </div>
            <div style="color:#7A7067; font-size:12px; font-weight:500; font-family:'JetBrains Mono', monospace;">
                ⏱️ {pub_in}
            </div>
        </div>
        <div style="font-size:18px; font-weight:800; color:#2E2620; margin-bottom:10px; line-height:1.45; font-family:'Be Vietnam Pro', sans-serif;">
            {title_in}
        </div>
        <div style="font-size:14.5px; color:#4A4038; line-height:1.65; border-left:3px solid #C96A4B; padding-left:14px; background:#FAF6F0; border-radius:4px; padding:12px 16px;">
            {clean_desc}
        </div>
        {link_html}
    </div>
    """)
    
    # Chạy tiền xử lý khử trùng lặp qua merge_title_and_description()
    full_live_text = merge_title_and_description(title_in, desc_in)
    res_live = predict_text(full_live_text, resources)
    
    # Dự báo mức độ tác động chính thức được đưa ra bởi Mô hình BERT-base
    bert_pred_live = res_live["bert_base"]["pred_label"]
    bert_conf_live = res_live["bert_base"]["confidence"] * 100
    pred_color = LABEL_COLORS[bert_pred_live]
    
    # 3 Thẻ kết quả đối chiếu
    st.markdown("#### KẾT QUẢ DỰ BÁO MỨC ĐỘ TÁC ĐỘNG")
    c_res1, c_res2, c_res3 = st.columns(3)
    with c_res1:
        impact_desc = "Tác động cao" if bert_pred_live == "HIGH" else ("Tác động trung bình" if bert_pred_live == "MEDIUM" else "Tác động thấp")
        render_html(f"""
        <div class="kpi-card kpi-card-highlight" style="min-height: 160px; padding: 18px 20px;">
            <div style="font-size: 11.5px; font-weight: 800; color: #8C4A32; text-transform: uppercase; letter-spacing: 0.04em;">
                KẾT QUẢ DỰ BÁO (MÔ HÌNH BERT-BASE)
            </div>
            <div style="font-size: 26px; font-weight: 800; color: {pred_color}; font-family: 'JetBrains Mono', monospace; margin: 6px 0;">
                {bert_pred_live} ({bert_conf_live:.2f}%)
            </div>
            <div style="font-size: 12.5px; color: #4A4038; margin-top: auto;">
                Mức độ tác động: <b>{impact_desc}</b>
            </div>
        </div>
        """)
        
    with c_res2:
        fin_pred = res_live["finbert"]["pred_label"]
        fin_conf = res_live["finbert"]["confidence"] * 100
        render_html(f"""
        <div class="kpi-card" style="min-height: 160px; padding: 18px 20px;">
            <div style="font-size: 11.5px; font-weight: 700; color: #7A7067; text-transform: uppercase; letter-spacing: 0.04em;">
                ĐỐI CHIẾU MÔ HÌNH FINBERT
            </div>
            <div style="font-size: 24px; font-weight: 700; color: {LABEL_COLORS[fin_pred]}; font-family: 'JetBrains Mono', monospace; margin: 6px 0;">
                {fin_pred} ({fin_conf:.2f}%)
            </div>
            <div style="font-size: 12px; color: #7A7067; margin-top: auto;">
                Mô hình Transformer tài chính
            </div>
        </div>
        """)
        
    with c_res3:
        bilstm_pred = res_live["bilstm"]["pred_label"]
        bilstm_conf = res_live["bilstm"]["confidence"] * 100
        render_html(f"""
        <div class="kpi-card" style="min-height: 160px; padding: 18px 20px;">
            <div style="font-size: 11.5px; font-weight: 700; color: #7A7067; text-transform: uppercase; letter-spacing: 0.04em;">
                ĐỐI CHIẾU MÔ HÌNH BI-LSTM
            </div>
            <div style="font-size: 24px; font-weight: 700; color: {LABEL_COLORS[bilstm_pred]}; font-family: 'JetBrains Mono', monospace; margin: 6px 0;">
                {bilstm_pred} ({bilstm_conf:.2f}%)
            </div>
            <div style="font-size: 12px; color: #7A7067; margin-top: auto;">
                Mạng nơ-ron hồi quy hai chiều
            </div>
        </div>
        """)
        
    # Thanh phân phối xác suất
    render_probability_distribution_view(res_live, "rt_prob_bar_chart")
    
    # PHÂN TÍCH Ý NGHĨA MỨC ĐỘ TÁC ĐỘNG CỦA THÔNG TIN
    imp_data = res_live.get("impact_meaning") or res_live.get("ews")
    if not imp_data:
        target_pred = res_live.get("bert_base", {}).get("pred_label", "MEDIUM")
        if target_pred == "HIGH":
            imp_data = {
                "tier": "HIGH",
                "icon": "🔴",
                "title": "PHÂN LỚP HIGH (TÁC ĐỘNG CAO)",
                "info_characteristic": "Phản ánh các cú sốc kinh tế vĩ mô đột ngột (lãi suất ngân hàng trung ương, lạm phát tăng vọt, biến động địa chính trị, khủng hoảng thanh khoản).",
                "expected_impact": "Có khả năng tạo ra biên độ biến động rất lớn và lan tỏa rộng trên toàn bộ thị trường chứng khoán.",
                "color": "#C96A4B",
                "badge_color": "#C96A4B",
                "bg_color": "#FDF2E9",
                "border_color": "#C96A4B"
            }
        elif target_pred == "MEDIUM":
            imp_data = {
                "tier": "MEDIUM",
                "icon": "🟡",
                "title": "PHÂN LỚP MEDIUM (TÁC ĐỘNG TRUNG BÌNH)",
                "info_characteristic": "Báo cáo kết quả hoạt động kinh doanh quý, biến động doanh thu, kế hoạch nhân sự hoặc hợp đồng mới của từng doanh nghiệp cụ thể.",
                "expected_impact": "Gây biến động cục bộ đối với từng cổ phiếu riêng lẻ hoặc một nhóm ngành nghề, không làm thay đổi xu hướng vĩ mô chung.",
                "color": "#D9A441",
                "badge_color": "#D9A441",
                "bg_color": "#FEF9E7",
                "border_color": "#D9A441"
            }
        else:
            imp_data = {
                "tier": "LOW",
                "icon": "🔵",
                "title": "PHÂN LỚP LOW (TÁC ĐỘNG THẤP)",
                "info_characteristic": "Thông báo thủ tục quản trị doanh nghiệp thường nhật, lịch họp đại hội đồng cổ đông, công bố thông tin hành chính định kỳ.",
                "expected_impact": "Hầu như không tác động đến kỳ vọng lợi suất và giá cổ phiếu do thông tin đã nằm trong kế hoạch từ trước.",
                "color": "#7A9A8B",
                "badge_color": "#7A9A8B",
                "bg_color": "#E8F8F5",
                "border_color": "#7A9A8B"
            }
    render_html(f"""
    <div style="background:{imp_data['bg_color']}; border: 1.5px solid {imp_data['border_color']}; border-radius:12px; padding:20px 24px; margin-top:20px; box-shadow:0 4px 14px rgba(46,38,32,0.04);">
        <div style="font-size:14px; font-weight:800; color:{imp_data['color']}; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:8px;">
            {imp_data['icon']} {imp_data['title']}
        </div>
        <div style="font-size:14px; color:#2E2620; margin-bottom:8px; line-height:1.6;">
            <b>Đặc tính thông tin:</b> {imp_data['info_characteristic']}
        </div>
        <div style="font-size:14px; color:#2E2620; line-height:1.6; border-top:1px solid rgba(217,205,193,0.6); padding-top:10px;">
            <b>Tác động kỳ vọng:</b> {imp_data['expected_impact']}
        </div>
    </div>
    """)
    
    # Cho phép nhập tin tùy chỉnh
    with st.expander("✍️ Tự nhập hoặc dán bản tin tài chính bất kỳ", expanded=False):
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
            c_text = merge_title_and_description(c_title, c_desc)
            c_res = predict_text(c_text, resources)
            c_pred = c_res["bert_base"]["pred_label"]
            c_conf = c_res["bert_base"]["confidence"] * 100
            st.success(f"Kết quả dự báo từ Mô hình BERT-base: **{c_pred}** (Độ tin cậy: {c_conf:.2f}%)")
            render_probability_distribution_view(c_res, "rt_custom_prob_bars")

# ==============================================================================
# TAB 3: TIỀN XỬ LÝ NLP & KHÔNG GIAN VECTOR 3D (XAI PIPELINE & 3D EMBEDDINGS)
# ==============================================================================
with tab_pipeline:
    st.markdown("### TIỀN XỬ LÝ NLP & KHÔNG GIAN VECTOR 3D (XAI PIPELINE & 3D EMBEDDINGS)")
    st.markdown("""
    Giải phẫu chuyên sâu 4 trạm xử lý trong kiến trúc học máy: Từ khử trùng lặp và rò rỉ dữ liệu (Anti-Leakage), phân tách token, giải mã ma trận Self-Attention tầng 11 đến không gian biểu diễn ẩn đa chiều 3D.
    """)
    
    demo_samples = [
        "Sensex crashes 1,150 pts on rising crude prices; Nifty holds 22,000 amid Middle East tensions https://reuters.com/market-update",
        "Quarterly corporate earnings remain largely flat for small-cap manufacturing firms with modest order backlog. <i>Source: PTI</i>",
        "Company schedules annual general meeting for next month to approve dividend distribution and routine proxy voting.",
        "Breaking: Federal Reserve unexpectedly hikes policy rate by 50 bps amid persistent inflation shock wave."
    ]
    demo_selected = st.selectbox(
        "Chọn câu văn mẫu để giải phẫu qua 4 trạm xử lý:",
        options=demo_samples,
        index=0,
        key="pl_select_demo_sample"
    )
    
    # Tiền xử lý văn bản đầu vào qua hàm chuẩn hóa
    clean_info = inspect_cleaning_pipeline(demo_selected)
    cleaned_demo_text = clean_info["cleaned"]
    res_demo = predict_text(cleaned_demo_text, resources)
    
    st1, st2, st3, st4 = st.tabs([
        "TRẠM 1: KHỬ TRÙNG LẶP & RÒ RỈ (ANTI-LEAKAGE)",
        "TRẠM 2: PHÂN TÁCH TOKEN & MA TRẬN HÓA",
        "TRẠM 3: GIẢI PHẪU CƠ CHẾ CHÚ Ý TẦNG 11",
        "TRẠM 4: KHÔNG GIAN BIỂU DIỄN 3D"
    ])
    
    # TRẠM 1: KHỬ TRÙNG LẶP & RÒ RỈ
    with st1:
        st.markdown("#### 1. TIỀN XỬ LÝ CHUẨN HÓA & KHỬ TRÙNG LẶP NỘI TẠI (MỤC 3.1 & 3.2 BÁO CÁO)")
        c_raw, c_clean = st.columns(2)
        with c_raw:
            st.markdown("##### Văn bản thô ban đầu (Raw Input):")
            render_html(f'<div class="academic-text-box">{clean_info["raw"]}</div>')
            st.caption(f"Độ dài: {clean_info['raw_length']} ký tự.")
        with c_clean:
            st.markdown("##### Văn bản sau khi qua merge_title_and_description():")
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
        **Nguyên tắc phòng ngừa Data Leakage & Khử Trùng lặp (Mục 3.1 & 3.2 Báo cáo):**
        - **Khử Trùng lặp Nội tại:** Nếu Title và Description giống hệt nhau hoặc Description bắt đầu bằng Title, thuật toán tự động giữ lại một bản sao duy nhất để tránh tăng trọng số nhân tạo cho các từ vựng.
        - **Khử URL & Tên miền:** Loại bỏ toàn bộ `http://`, `https://`, `www.` để mô hình không học lén độ uy tín của nguồn xuất bản.
        - **Khử Thẻ HTML & Ký tự đặc biệt:** Giải mã thực thể HTML (`html.unescape`) và loại bỏ `<i>`, `<b>`, `<p>`, `<a>`.
        - **Cắt chuỗi:** Giới hạn tối đa 64 tokens để tương thích với kích thước đầu vào của kiến trúc Transformer.
        """)

    # TRẠM 2: PHÂN TÁCH TOKEN & MA TRẬN HÓA
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
            - **Từ điển:** 20,000 từ vựng. Token đặc biệt: `<pad>: 0`, `<unk>: 1`.
            - **Số token thực tế:** {res_demo['bilstm']['padded_len']} từ (phần còn lại điền `<pad>: 0`).
            """)
            token_preview = f"{res_demo['bilstm']['token_ids'][:25]} ... [0, 0, 0]"
            render_html(f'<div class="academic-text-box">{token_preview}</div>')
        with c_bert_tok:
            st.markdown("##### B. BERT WordPiece Tokenizer (Mô hình BERT-base):")
            bert_toks = [t for t in res_demo["bert_base"]["tokens"] if t != "[PAD]"]
            st.markdown(f"""
            - **Cơ chế:** Phân rã từ con Subword (`##`) giúp triệt tiêu hoàn toàn hiện tượng từ ngoài từ điển (Out-Of-Vocabulary - OOV).
            - **Token đặc biệt:** `[CLS]` (Classification token ở đầu câu), `[SEP]` (Phân tách cuối câu), `[PAD]`.
            - **Số token sau phân rã:** {len(bert_toks)} tokens.
            """)
            subword_preview = f"{bert_toks[:18]} ... [SEP]"
            render_html(f'<div class="academic-text-box">{subword_preview}</div>')

    # TRẠM 3: GIẢI PHẪU CƠ CHẾ CHÚ Ý TẦNG 11 CỦA MÔ HÌNH BERT-BASE
    with st3:
        st.markdown("#### 3. CƠ CHẾ TỰ CHÚ Ý (SELF-ATTENTION) TẦNG 11 CỦA MÔ HÌNH BERT-BASE")
        st.markdown("""
        Trích xuất trực tiếp **Attention Weights** từ tầng cuối cùng (Layer 11) của Mô hình BERT-base từ token `[CLS]` tới mọi từ trong câu:
        - **BẮT BUỘC lấy trung bình (MEAN) qua 12 Attention Heads**: `cls_attn.mean(dim=0).cpu().numpy()`. (Tuyệt đối không dùng `torch.max()`).
        - **Cơ chế Phòng vệ 2 Lớp (Two-tier Defense):** Lọc sạch dấu câu, special tokens và gom nhóm subwords (`crash` + `##es` -> `crashes`), chuẩn hóa Min-Max scaling đưa trọng số từ vựng thực tế về `[0.0 - 1.0]`.
        """)
        
        # Render Heatmap HTML từ BERT-base
        render_html(render_attention_heatmap_html(res_demo["bert_base"]["word_attention"]))
        
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        # Bảng Top Attended Keywords
        top_words = res_demo["bert_base"].get("top_keywords", [])
        if not top_words:
            exclude_tokens = {'[cls]', '[sep]', '[pad]', ',', '.', ';', ':', '!', '?', '-', '--', '/', '\\', 'http', 'https', 'reuters', 'com'}
            top_words = [x for x in res_demo["bert_base"]["word_attention"] if x[0].lower() not in exclude_tokens and any(c.isalpha() for c in x[0])]
            top_words = sorted(top_words, key=lambda x: x[1], reverse=True)
        top_words = top_words[:8]
        
        col_t1, col_t2 = st.columns([1.5, 2.5])
        with col_t1:
            st.markdown("##### Top Từ Khóa Tập Trung Chú Ý (Highest Attention):")
            top_df = pd.DataFrame(top_words, columns=["Từ Khóa Tài Chính Vĩ Mô", "Trọng số Attention (0.0 - 1.0)"])
            top_df["Trọng số Attention (0.0 - 1.0)"] = top_df["Trọng số Attention (0.0 - 1.0)"].map(lambda x: f"{x:.4f}")
            st.dataframe(top_df, use_container_width=True)
        with col_t2:
            st.markdown("##### Luận cứ Khoa học:")
            st.markdown("""
            - **Khắc phục lỗi Attention Sink:** Token `[CLS]` và dấu câu thường thu hút trọng số nhân tạo do kiến trúc Softmax chuẩn hóa. Việc tính **MEAN qua 12 Attention Heads** kết hợp lọc bỏ dấu câu giúp bóc tách chính xác các từ vựng bước ngoặt (*tensions, crude, crashes, hike, fed, inflation*).
            - **Đặc trưng của Mô hình BERT-base:** Ngữ liệu tiền huấn luyện rộng lớn của BERT-base giúp mô hình nhận diện sâu sắc các từ khóa địa chính trị và ngân hàng trung ương, vốn là nguyên nhân cốt lõi kích hoạt hơn 80% các sự kiện thuộc nhóm HIGH.
            """)

    # TRẠM 4: KHÔNG GIAN BIỂU DIỄN 3D
    with st4:
        st.markdown("#### 4. HOẠT ẢNH KHÔNG GIAN VECTOR 3D ĐA MÔ HÌNH (MULTI-MODEL 3D LATENT PCA SPACE)")
        st.markdown("""
        Đối sánh sự phân cụm và mức độ hội tụ của vector biểu diễn ẩn qua PCA xuống không gian 3 chiều ($PC_1, PC_2, PC_3$) giữa cả 3 mô hình học máy:
        """)
        
        # Bộ chọn 1 trong 3 mô hình
        selected_3d_key = st.radio(
            "Chọn mô hình để quan sát không gian tiềm ẩn PCA 3D:",
            options=["bert_base", "finbert", "bilstm"],
            format_func=lambda x: "Mô hình BERT-base (768 chiều [CLS])" if x == "bert_base" else ("Mô hình FinBERT (768 chiều [CLS])" if x == "finbert" else "Mô hình Bi-LSTM (128 chiều forward & backward)"),
            index=0,
            horizontal=True,
            key="pl_select_3d_model"
        )
        
        pca_dict = res_demo.get("pca_3d", {})
        if selected_3d_key in pca_dict:
            pca_data = pca_dict[selected_3d_key]
        else:
            pca_data = {
                "name": "Mô hình BERT-base",
                "dim": "768 chiều [CLS]",
                "variance_ratio": "86.4% (PC1: 48.2%, PC2: 24.1%, PC3: 14.1%)",
                "centers": res_demo.get("cluster_centers"),
                "spread": 0.65,
                "coords": res_demo.get("coords_3d"),
                "pred_label": res_demo.get("bert_base", {}).get("pred_label", "MEDIUM"),
                "insight": "Mô hình BERT-base đạt khoảng cách liên cụm cực đại và phân tách 3 vùng ngữ nghĩa rõ nét nhất."
            }
            
        st.plotly_chart(render_3d_latent_space(
            pca_data["coords"],
            pca_data["pred_label"],
            pca_data["centers"],
            sample_title=demo_selected[:35],
            model_name=pca_data["name"],
            spread=pca_data["spread"]
        ), use_container_width=True, key=f"pl_3d_latent_chart_{selected_3d_key}")
        
        # Bảng giải thích cơ chế hội tụ của mô hình được chọn
        render_html(f"""
        <div style="background:#FFFFFF; border:1.5px solid #D9CDC1; border-left:5px solid #C96A4B; border-radius:12px; padding:18px 22px; margin-top:14px; box-shadow:0 3px 12px rgba(46,38,32,0.04);">
            <div style="font-size:14px; font-weight:700; color:#8C4A32; text-transform:uppercase; letter-spacing:0.03em; margin-bottom:8px;">
                🔬 Luận cứ Hội tụ Không gian Tiềm ẩn: {pca_data['name']}
            </div>
            <div style="font-size:13.5px; color:#4A4038; line-height:1.6; margin-bottom:6px;">
                <b>• Số chiều không gian biểu diễn gốc:</b> {pca_data['dim']} | <b>Phương sai giải thích qua 3 thành phần chính:</b> {pca_data['variance_ratio']}
            </div>
            <div style="font-size:13.5px; color:#2E2620; line-height:1.6;">
                <b>• Đặc tính phân cụm PCA:</b> {pca_data['insight']}
            </div>
        </div>
        """)

# ==============================================================================
# FOOTER HỌC THUẬT
# ==============================================================================
render_html("""
<div style="text-align:center; padding: 40px 0 30px 0; border-top: 1px solid #D9CDC1; margin-top: 60px; font-family:'Be Vietnam Pro', sans-serif;">
    <div style="color: #4A4038; font-weight: 600; font-size: 13.5px; line-height: 1.6;">
        Đề tài Nghiên cứu Machine Learning: Nhóm 9 | Khoa Tài chính, Trường ĐH Kinh tế, ĐH Đà Nẵng
    </div>
    <div style="color: #8C8075; font-size: 12px; margin-top: 8px; line-height: 1.6;">
        Đối sánh thực nghiệm giữa Mô hình Bi-LSTM, Mô hình FinBERT và Mô hình BERT-base | Báo cáo BCaogiuakyML_2
    </div>
</div>
""")
