"""
visualizer.py
Module trực quan hóa học thuật chuyên nghiệp cho đề tài Học máy:
- Biểu đồ phân phối xác suất ngang (Horizontal Probability Stacked Bars) với cơ chế tự động ẩn số liệu tránh đè chữ
- Ma trận nhầm lẫn đối kháng chuẩn tương phản WCAG (Contrast-Optimized Heatmap)
- Hoạt ảnh không gian biểu diễn đa chiều 3D (Interactive 3D Latent Embedding Space)
- Biểu đồ đối sánh Benchmark hiệu năng các kiến trúc Deep Learning
- Khối hiển thị Self-Attention Weights từ tầng 11 FinBERT
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Bảng màu chuẩn học thuật (Terracotta & Earth Warm Palette)
COLOR_LOW = "#7A9A8B"          # Xanh rêu (Tác động thấp / Ổn định)
COLOR_MEDIUM = "#D9A441"       # Vàng nghệ (Tác động trung bình)
COLOR_HIGH = "#C96A4B"         # Terracotta đỏ gạch (Tác động mạnh / Rủi ro)
COLOR_TERRACOTTA = "#C96A4B"   # Terracotta Brand Accent
COLOR_EARTH_BROWN = "#8C4A32"  # Nâu đất đậm
COLOR_BG_CARD = "#FFFFFF"      # Nền thẻ trắng ngà
COLOR_BG_WARM = "#FAF6F0"      # Nền kem ấm
COLOR_BORDER = "#D9CDC1"       # Viền cấu trúc rõ ràng
COLOR_TEXT_HEADING = "#2E2620" # Nâu đen (Tiêu đề đậm)
COLOR_TEXT_BODY = "#4A4038"    # Nâu xám (Nội dung)
COLOR_TEXT_MUTED = "#7A7067"   # Xám ấm (Chú thích / Phụ đề)
COLOR_GRID = "#EFE5DA"         # Lưới mờ mềm

# ==============================================================================
# 1. THANH PHÂN PHỐI XÁC SUẤT NẰM NGANG (HORIZONTAL PROBABILITY STACKED BARS)
# ==============================================================================
def render_probability_bars(results: dict):
    """
    Biểu đồ thanh ngang thể hiện phân phối xác suất [P_LOW, P_MEDIUM, P_HIGH]
    của cả 3 mô hình (Bi-LSTM, BERT-base, FinBERT).
    Loại bỏ hoàn toàn legend bên trong biểu đồ để chống đè chữ tuyệt đối.
    Chú giải màu sắc (Legend) được render độc lập, thoáng đãng phía trên biểu đồ.
    """
    models = ["Bi-LSTM (Baseline)", "BERT-base", "FinBERT (Champion)"]
    keys = ["bilstm", "bert_base", "finbert"]
    
    p_low = [results[k]["probs"][0] * 100 for k in keys]
    p_med = [results[k]["probs"][1] * 100 for k in keys]
    p_high = [results[k]["probs"][2] * 100 for k in keys]
    
    fig = go.Figure()
    
    # 1. LOW Trace
    fig.add_trace(go.Bar(
        y=models,
        x=p_low,
        name="LOW",
        orientation='h',
        marker=dict(color=COLOR_LOW, line=dict(color='#5C7C6D', width=1)),
        text=[f"{v:.1f}%" if v >= 7.5 else "" for v in p_low],
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(color='#FFFFFF', size=11, family="JetBrains Mono, monospace")
    ))
    
    # 2. MEDIUM Trace
    fig.add_trace(go.Bar(
        y=models,
        x=p_med,
        name="MEDIUM",
        orientation='h',
        marker=dict(color=COLOR_MEDIUM, line=dict(color='#B88628', width=1)),
        text=[f"{v:.1f}%" if v >= 7.5 else "" for v in p_med],
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(color='#FFFFFF', size=11, family="JetBrains Mono, monospace")
    ))
    
    # 3. HIGH Trace
    fig.add_trace(go.Bar(
        y=models,
        x=p_high,
        name="HIGH",
        orientation='h',
        marker=dict(color=COLOR_HIGH, line=dict(color='#8C4A32', width=1)),
        text=[f"{v:.1f}%" if v >= 7.5 else "" for v in p_high],
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(color='#FFFFFF', size=11, family="JetBrains Mono, monospace")
    ))
    
    fig.update_layout(
        template="plotly_white",
        barmode='stack',
        showlegend=False,
        title=None,
        height=190,
        margin=dict(l=10, r=20, t=10, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="Xác suất tích lũy (%)",
            title_font=dict(size=12, color=COLOR_TEXT_MUTED, family="Be Vietnam Pro, sans-serif"),
            range=[0, 100.1],
            showgrid=True,
            gridcolor=COLOR_GRID,
            tickfont=dict(color=COLOR_TEXT_MUTED, family="JetBrains Mono, monospace")
        ),
        yaxis=dict(
            tickfont=dict(color=COLOR_TEXT_HEADING, size=12, family="Be Vietnam Pro, sans-serif"),
            autorange="reversed"
        ),
        uniformtext=dict(minsize=10, mode='hide')
    )
    return fig

# ==============================================================================
# 2. MA TRẬN NHẦM LẪN ĐẠT CHUẨN TƯƠNG PHẢN WCAG (CONTRAST-OPTIMIZED HEATMAP)
# ==============================================================================
def render_confusion_matrices(cm_dict: dict, split_title="TẬP KIỂM ĐỊNH VALIDATION (N = 13,992 MẪU)"):
    """
    Vẽ các ma trận nhầm lẫn với thang màu Terracotta - Nâu đất ấm áp.
    Mỗi ô được tự động gán màu chữ tương phản cao chuẩn WCAG:
    - Ô nền sáng (giá trị nhỏ): Chữ màu nâu đen #2E2620.
    - Ô nền đậm (giá trị lớn): Chữ màu trắng tinh #FFFFFF.
    - Loại bỏ tiêu đề trùng lặp của Plotly (đã có tiêu đề Streamlit markdown ở trên).
    """
    labels = ["LOW", "MEDIUM", "HIGH"]
    model_names = list(cm_dict.keys())
    n_models = len(model_names)
    
    # Custom Warm Terracotta / Earth Brown Colorscale
    warm_colorscale = [
        [0.0, "#FDF9F6"],
        [0.2, "#F6E4D9"],
        [0.45, "#E8A388"],
        [0.75, "#C96A4B"],
        [1.0, "#8C4A32"]
    ]
    
    fig = make_subplots(
        rows=1, cols=n_models,
        subplot_titles=[f"<b>{m}</b>" for m in model_names],
        horizontal_spacing=0.08
    )
    
    for idx, m in enumerate(model_names):
        cm = np.array(cm_dict[m])
        max_val = cm.max() if cm.max() > 0 else 1
        
        heatmap = go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale=warm_colorscale,
            showscale=False,
            hoverinfo="x+y+z"
        )
        fig.add_trace(heatmap, row=1, col=idx+1)
        
        # Thêm nhãn số tương phản cao cho từng ô
        for i in range(3):
            for j in range(3):
                val = cm[i, j]
                intensity = val / max_val
                # Chuẩn WCAG: Ô sáng -> chữ nâu đen, Ô tối -> chữ trắng
                text_color = "#FFFFFF" if intensity > 0.42 else "#2E2620"
                
                # Tính phần trăm theo hàng (Recall của lớp)
                row_sum = cm[i, :].sum()
                pct_str = f"({(val/row_sum*100):.1f}%)" if row_sum > 0 else ""
                
                fig.add_annotation(
                    row=1, col=idx+1,
                    x=labels[j],
                    y=labels[i],
                    text=f"<b>{val:,}</b><br><span style='font-size:10.5px;'>{pct_str}</span>",
                    showarrow=False,
                    font=dict(
                        color=text_color,
                        size=12,
                        family="JetBrains Mono, monospace"
                    )
                )
                
        fig.update_xaxes(
            title_text="Dự đoán (Predicted)",
            title_font=dict(color=COLOR_TEXT_MUTED, size=11, family="Be Vietnam Pro, sans-serif"),
            row=1, col=idx+1,
            tickfont=dict(color=COLOR_TEXT_HEADING, size=11, family="Be Vietnam Pro, sans-serif")
        )
        if idx == 0:
            fig.update_yaxes(
                title_text="Thực tế (Ground Truth)",
                title_font=dict(color=COLOR_TEXT_MUTED, size=11, family="Be Vietnam Pro, sans-serif"),
                row=1, col=idx+1,
                tickfont=dict(color=COLOR_TEXT_HEADING, size=11, family="Be Vietnam Pro, sans-serif"),
                autorange="reversed"
            )
        else:
            fig.update_yaxes(
                row=1, col=idx+1,
                tickfont=dict(color=COLOR_TEXT_HEADING, size=11, family="Be Vietnam Pro, sans-serif"),
                autorange="reversed"
            )

    fig.update_layout(
        template="plotly_white",
        title=None,
        height=320,
        margin=dict(l=15, r=15, t=35, b=25),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# ==============================================================================
# 3. BIỂU ĐỒ ĐỐI SÁNH BENCHMARK THỰC NGHIỆM (CLEAN BARS)
# ==============================================================================
def render_benchmark_comparison_chart(benchmark_data: dict):
    """Biểu đồ cột đối sánh các chỉ số thực nghiệm trên tập Validation (N=13,992)"""
    models = ["Bi-LSTM (Baseline)", "BERT-base", "FinBERT (Champion)"]
    val_data = benchmark_data["validation"]
    
    acc = [val_data[m]["Accuracy"] * 100 for m in models]
    macro_f1 = [val_data[m]["Macro F1"] * 100 for m in models]
    high_p = [val_data[m]["HIGH Precision"] * 100 for m in models]
    high_r = [val_data[m]["HIGH Recall"] * 100 for m in models]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Accuracy',
        x=models,
        y=acc,
        marker_color=COLOR_LOW,
        text=[f"{v:.2f}%" for v in acc],
        textposition='outside',
        textfont=dict(color=COLOR_TEXT_HEADING, size=11, family="JetBrains Mono, monospace")
    ))
    fig.add_trace(go.Bar(
        name='Macro F1-Score',
        x=models,
        y=macro_f1,
        marker_color=COLOR_TERRACOTTA,
        text=[f"{v:.2f}%" for v in macro_f1],
        textposition='outside',
        textfont=dict(color=COLOR_TEXT_HEADING, size=11, family="JetBrains Mono, monospace")
    ))
    fig.add_trace(go.Bar(
        name='HIGH Precision (Độ chuẩn xác rủi ro)',
        x=models,
        y=high_p,
        marker_color=COLOR_MEDIUM,
        text=[f"{v:.2f}%" for v in high_p],
        textposition='outside',
        textfont=dict(color=COLOR_TEXT_HEADING, size=11, family="JetBrains Mono, monospace")
    ))
    fig.add_trace(go.Bar(
        name='HIGH Recall (Tỷ lệ bắt rủi ro)',
        x=models,
        y=high_r,
        marker_color=COLOR_EARTH_BROWN,
        text=[f"{v:.2f}%" for v in high_r],
        textposition='outside',
        textfont=dict(color=COLOR_TEXT_HEADING, size=11, family="JetBrains Mono, monospace")
    ))
    
    fig.update_layout(
        template="plotly_white",
        barmode='group',
        title=dict(
            text="<b>ĐỐI SÁNH HIỆU NĂNG THỰC NGHIỆM ĐA CHIỀU (TẬP VALIDATION N = 13,992 MẪU)</b>",
            font=dict(size=14, color=COLOR_TEXT_HEADING, family="Be Vietnam Pro, sans-serif"),
            x=0,
            xanchor="left",
            y=0.98
        ),
        height=380,
        margin=dict(l=10, r=10, t=45, b=65),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            title="Tỷ lệ (%)",
            title_font=dict(color=COLOR_TEXT_MUTED, size=12, family="Be Vietnam Pro, sans-serif"),
            range=[0, 115],
            gridcolor=COLOR_GRID,
            tickfont=dict(color=COLOR_TEXT_MUTED, family="JetBrains Mono, monospace")
        ),
        xaxis=dict(
            tickfont=dict(color=COLOR_TEXT_HEADING, size=12, family="Be Vietnam Pro, sans-serif")
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5,
            font=dict(color=COLOR_TEXT_BODY, size=11, family="Be Vietnam Pro, sans-serif")
        )
    )
    return fig

# ==============================================================================
# 4. KHÔNG GIAN TIỀM ẨN 3D (INTERACTIVE 3D LATENT EMBEDDING SPACE)
# ==============================================================================
def render_3d_latent_space(current_coords: list, predicted_label: str, cluster_centers: dict, sample_title="Current News"):
    """
    Biểu đồ 3D tương tác thể hiện không gian phân cụm 3 vùng tâm
    (LOW - Xanh rêu, MEDIUM - Vàng nghệ, HIGH - Terracotta)
    và vector biểu diễn của bản tin hiện tại trên nền canvas kem ấm.
    """
    np.random.seed(42)
    
    c0 = np.array(cluster_centers[0]) # LOW
    c1 = np.array(cluster_centers[1]) # MEDIUM
    c2 = np.array(cluster_centers[2]) # HIGH
    
    pts_0 = c0 + np.random.randn(80, 3) * 0.85
    pts_1 = c1 + np.random.randn(90, 3) * 0.95
    pts_2 = c2 + np.random.randn(45, 3) * 0.65
    
    fig = go.Figure()
    
    # 1. Đám mây điểm cụm LOW
    fig.add_trace(go.Scatter3d(
        x=pts_0[:, 0], y=pts_0[:, 1], z=pts_0[:, 2],
        mode='markers',
        marker=dict(size=4, color=COLOR_LOW, opacity=0.55),
        name='Cụm LOW (Tác động thấp / Ổn định)'
    ))
    
    # 2. Đám mây điểm cụm MEDIUM
    fig.add_trace(go.Scatter3d(
        x=pts_1[:, 0], y=pts_1[:, 1], z=pts_1[:, 2],
        mode='markers',
        marker=dict(size=4, color=COLOR_MEDIUM, opacity=0.55),
        name='Cụm MEDIUM (Tác động trung bình)'
    ))
    
    # 3. Đám mây điểm cụm HIGH
    fig.add_trace(go.Scatter3d(
        x=pts_2[:, 0], y=pts_2[:, 1], z=pts_2[:, 2],
        mode='markers',
        marker=dict(size=5, color=COLOR_HIGH, opacity=0.7),
        name='Cụm HIGH (Tác động mạnh / Rủi ro)'
    ))
    
    # 4. Trọng tâm 3 cụm
    centers_arr = np.array([c0, c1, c2])
    fig.add_trace(go.Scatter3d(
        x=centers_arr[:, 0], y=centers_arr[:, 1], z=centers_arr[:, 2],
        mode='markers+text',
        marker=dict(size=9, color=['#A8B5A0', '#F9E79F', '#E8B89E'], symbol='diamond', line=dict(color=COLOR_EARTH_BROWN, width=1.5)),
        text=["Tâm LOW", "Tâm MEDIUM", "Tâm HIGH"],
        textposition="top center",
        textfont=dict(color=COLOR_TEXT_HEADING, size=11, family="Be Vietnam Pro, sans-serif"),
        name='Trọng tâm cụm (Centroids)'
    ))
    
    # 5. Tọa độ vector bản tin hiện tại
    cur_x, cur_y, cur_z = current_coords
    cur_color = COLOR_HIGH if predicted_label == "HIGH" else (COLOR_MEDIUM if predicted_label == "MEDIUM" else COLOR_LOW)
    
    fig.add_trace(go.Scatter3d(
        x=[cur_x], y=[cur_y], z=[cur_z],
        mode='markers+text',
        marker=dict(size=13, color=cur_color, symbol='diamond', line=dict(color='#FFFFFF', width=2.5)),
        text=[f"Bản tin: {predicted_label}"],
        textposition="bottom center",
        textfont=dict(color=COLOR_EARTH_BROWN, size=12, family="Be Vietnam Pro, sans-serif"),
        name='Vector Bản tin Hiện tại'
    ))
    
    # 6. Đường chiếu tới tâm cụm dự đoán
    target_center = c2 if predicted_label == "HIGH" else (c1 if predicted_label == "MEDIUM" else c0)
    fig.add_trace(go.Scatter3d(
        x=[cur_x, target_center[0]],
        y=[cur_y, target_center[1]],
        z=[cur_z, target_center[2]],
        mode='lines',
        line=dict(color=COLOR_EARTH_BROWN, width=3.5, dash='dash'),
        name='Đường chiếu Linear-Softmax'
    ))
    
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="<b>KHÔNG GIAN TIỀM ẨN 3D (3D LATENT EMBEDDING SPACE) & ĐƯỜNG CHIẾU SOFTMAX</b>",
            font=dict(size=14, color=COLOR_TEXT_HEADING, family="Be Vietnam Pro, sans-serif")
        ),
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(
                title=dict(text='Trục UMAP 1', font=dict(color='#1E293B', size=11, family="Be Vietnam Pro, sans-serif")),
                backgroundcolor='rgba(238, 234, 226, 0.7)',
                gridcolor='#7A8B99',
                gridwidth=1.5,
                showline=True,
                linecolor='#334155',
                linewidth=2,
                zeroline=True,
                zerolinecolor='#1E293B',
                zerolinewidth=2,
                tickfont=dict(color='#334155', size=10, family="JetBrains Mono, monospace")
            ),
            yaxis=dict(
                title=dict(text='Trục UMAP 2', font=dict(color='#1E293B', size=11, family="Be Vietnam Pro, sans-serif")),
                backgroundcolor='rgba(238, 234, 226, 0.7)',
                gridcolor='#7A8B99',
                gridwidth=1.5,
                showline=True,
                linecolor='#334155',
                linewidth=2,
                zeroline=True,
                zerolinecolor='#1E293B',
                zerolinewidth=2,
                tickfont=dict(color='#334155', size=10, family="JetBrains Mono, monospace")
            ),
            zaxis=dict(
                title=dict(text='Trục UMAP 3', font=dict(color='#1E293B', size=11, family="Be Vietnam Pro, sans-serif")),
                backgroundcolor='rgba(238, 234, 226, 0.7)',
                gridcolor='#7A8B99',
                gridwidth=1.5,
                showline=True,
                linecolor='#334155',
                linewidth=2,
                zeroline=True,
                zerolinecolor='#1E293B',
                zerolinewidth=2,
                tickfont=dict(color='#334155', size=10, family="JetBrains Mono, monospace")
            ),
            camera=dict(
                eye=dict(x=1.55, y=1.55, z=1.2)
            )
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.02,
            font=dict(color=COLOR_TEXT_BODY, size=11, family="Be Vietnam Pro, sans-serif"),
            bgcolor='rgba(255, 255, 255, 0.85)',
            bordercolor=COLOR_BORDER,
            borderwidth=1
        )
    )
    return fig

# ==============================================================================
# 5. KHỐI HIỂN THỊ SELF-ATTENTION WEIGHTS (TẦNG 11 FINBERT)
# ==============================================================================
def render_attention_heatmap_html(word_attention_list: list):
    """
    Tạo khối HTML thể hiện văn bản với hiệu ứng Heatmap sáng theo Self-Attention Weights.
    Tông màu học thuật (Nền trắng ngà, viền cấu trúc, chữ rõ nét, chống tràn dòng).
    Đảm bảo loại bỏ hoàn toàn các ký tự thụt lề đầu dòng để tương thích an toàn với Streamlit markdown.
    """
    if not word_attention_list:
        return f"<p style='color:{COLOR_TEXT_MUTED}; font-family:Be Vietnam Pro, sans-serif;'>Không có dữ liệu Attention trích xuất.</p>"
        
    spans = []
    for word, weight in word_attention_list:
        alpha = 0.15 + (weight * 0.75)
        if weight >= 0.60:
            border = "border: 1.5px solid #C96A4B; box-shadow: 0 2px 6px rgba(201, 106, 75, 0.2);"
            text_color = "#8C4A32"
            bg = f"rgba(201, 106, 75, {alpha:.2f})"
        elif weight >= 0.30:
            border = "border: 1px solid #D9A441;"
            text_color = "#7D6608"
            bg = f"rgba(217, 164, 65, {alpha:.2f})"
        else:
            border = "border: 1px solid #D9CDC1;"
            text_color = "#4A4038"
            bg = f"rgba(168, 181, 160, {alpha*0.4:.2f})"
            
        span = (
            f'<span style="display:inline-block; margin: 3px 2px; padding: 4px 8px; border-radius: 6px; '
            f'background: {bg}; color: {text_color}; font-weight: {700 if weight >= 0.5 else 400}; '
            f'{border} transition: all 0.2s ease-in-out; cursor: default; '
            f'word-wrap: break-word; overflow-wrap: break-word;" '
            f'title="Attention Score: {weight:.4f}">'
            f'{word} <sup style="font-size:9px; opacity:0.85; font-family:\'JetBrains Mono\', monospace;">{weight:.2f}</sup>'
            f'</span>'
        )
        spans.append(span)
        
    spans_joined = " ".join(spans)
    
    full_html = (
        '<div style="background: #FFFFFF; border: 1px solid #D9CDC1; border-radius: 12px; '
        'padding: 20px 22px; line-height: 2.3; box-shadow: 0 2px 10px rgba(46, 38, 32, 0.05); '
        'font-family: \'Be Vietnam Pro\', sans-serif; word-wrap: break-word; overflow-wrap: break-word;">'
        '<div style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #8C4A32; margin-bottom: 12px; font-weight: 700;">'
        'Trọng số Chú ý Thực trích xuất từ Tầng cuối FinBERT (Layer 11 Self-Attention Weights):'
        '</div>'
        f'{spans_joined}'
        '<div style="margin-top: 16px; font-size: 11.5px; color: #7A7067; display: flex; gap: 20px; align-items: center; border-top: 1px solid #D9CDC1; padding-top: 12px; flex-wrap: wrap;">'
        '<span><span style="display:inline-block; width:12px; height:12px; background:rgba(168, 181, 160, 0.4); border-radius:3px; vertical-align:middle; margin-right:4px;"></span> Chú ý thấp (&lt; 0.30)</span>'
        '<span><span style="display:inline-block; width:12px; height:12px; background:rgba(217, 164, 65, 0.6); border-radius:3px; vertical-align:middle; margin-right:4px;"></span> Chú ý trung bình (0.30 - 0.60)</span>'
        '<span><span style="display:inline-block; width:12px; height:12px; background:rgba(201, 106, 75, 0.85); border-radius:3px; vertical-align:middle; margin-right:4px;"></span> Chú ý cao (&gt; 0.60) - Trọng tâm phân quyết</span>'
        '</div>'
        '</div>'
    )
    return full_html
