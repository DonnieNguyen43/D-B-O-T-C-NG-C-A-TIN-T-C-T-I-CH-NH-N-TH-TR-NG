# SKILL: Thiết kế UI/UX theo phong cách TỰA — "Nét Vẽ Yêu Thương"

> **Mục đích:** File MD này tổng hợp toàn bộ phân tích UI/UX của website https://tuacharity.site để AI Agent có thể áp dụng lại phong cách thiết kế này cho dự án Machine Learning của bạn (báo cáo, dashboard, landing page, portfolio model...).
>
> **Tinh thần cốt lõi:** Ấm áp – Thủ công – Đáng tin – Tối giản có chiều sâu cảm xúc.

---

## 1. TỔNG QUAN PHONG CÁCH (Design Philosophy)

| Yếu tố | Đặc điểm |
|---|---|
| **Cảm hứng** | Thủ công (handmade), vẽ tay, charity, câu chuyện con người |
| **Tông cảm xúc** | Ấm áp, chân thành, không thương hại, truyền cảm hứng |
| **Mật độ thị giác** | Thoáng (airy), nhiều white space, không nhồi nhét |
| **Độ tương phản** | Mềm – không chói, ưu tiên tông đất và kem |
| **Typography** | Font có nét mềm / serif hoặc sans-serif tròn trịa, cá tính nhẹ |
| **Chuyển động** | Nhẹ nhàng, fade-in, không giật, không chớp |

**Nguyên tắc vàng:** Mỗi section phải kể được một phần của câu chuyện. Thiết kế phục vụ nội dung, không phô trương kỹ thuật.

---

## 2. HỆ THỐNG MÀU SẮC (Color System)

### Bảng màu chính (suy luận từ phong cách tổng thể)

```css
:root {
  /* Nền & bề mặt */
  --bg-primary:    #FAF6F0;   /* Kem ấm — nền chính */
  --bg-secondary:  #F3EAE0;   /* Be nhạt — section xen kẽ */
  --bg-card:       #FFFFFF;   /* Trắng ngà cho card */

  /* Màu chủ đạo (brand) */
  --brand-primary: #C96A4B;   /* Terracotta — CTA, điểm nhấn */
  --brand-deep:    #8C4A32;   /* Nâu đất — heading, hover */
  --brand-soft:    #E8B89E;   /* Đào nhạt — badge, highlight */

  /* Màu phụ trợ */
  --accent-sage:   #A8B5A0;   /* Xanh rêu nhạt — yếu tố thiên nhiên */
  --accent-mustard:#D9A441;   /* Vàng nghệ — progress, icon */

  /* Chữ */
  --text-heading:  #2E2620;   /* Nâu đen — tiêu đề */
  --text-body:     #4A4038;   /* Nâu xám — nội dung */
  --text-muted:    #8A7E72;   /* Xám ấm — caption, meta */

  /* Viền & đường kẻ */
  --border-soft:   #E5D9CC;
  --shadow-warm:   rgba(140, 74, 50, 0.10);
}
```

### Quy tắc phối màu
- **60-30-10:** 60% nền kem/be → 30% nâu đất & terracotta → 10% vàng nghệ/xanh rêu làm điểm nhấn.
- Không dùng màu lạnh gắt (xanh dương chói, tím neon).
- Mọi bóng đổ (shadow) phải **ấm** (nhuốm nâu), không dùng shadow xám lạnh.
- Độ tương phản chữ/nền đảm bảo **WCAG AA** (≥ 4.5:1 cho body text).

---

## 3. TYPOGRAPHY (Hệ thống chữ)

```css
/* Heading — có cá tính, nét hơi mềm */
--font-heading: "Be Vietnam Pro", "Playfair Display", serif;

/* Body — dễ đọc, thân thiện */
--font-body: "Be Vietnam Pro", "Inter", sans-serif;

/* Nhấn mạnh / chữ ký tay (dùng tiết chế) */
--font-accent: "Dancing Script", cursive;
```

### Thang cỡ chữ (Type Scale)

| Cấp | Size | Weight | Line-height | Dùng cho |
|---|---|---|---|---|
| Display | 48–56px | 700 | 1.15 | Hero title |
| H1 | 36px | 700 | 1.25 | Tiêu đề section |
| H2 | 28px | 600 | 1.3 | Sub-section |
| H3 | 22px | 600 | 1.4 | Card title |
| Body | 16–17px | 400 | 1.7 | Đoạn văn (line-height rộng = thoáng) |
| Caption | 14px | 400 | 1.5 | Meta, ngày tháng |
| Quote | 18px | 400 italic | 1.6 | Trích dẫn lời nhắn |

**Lưu ý cho AI Agent:** Line-height body **1.7** là chìa khóa tạo cảm giác "thở". Không bao giờ để body text dưới 16px.

---

## 4. LAYOUT & GRID (Bố cục)

- **Max-width container:** 1120–1200px, căn giữa.
- **Padding ngang mobile:** 20–24px.
- **Khoảng cách section dọc:** 80–120px (thoáng!).
- **Grid card:** 1 cột (mobile) → 2 cột (tablet) → 3 cột (desktop), gap 24–32px.
- **Căn lề:** Chủ yếu **căn trái** cho nội dung dài; **căn giữa** cho hero, CTA, tiêu đề section.

### Cấu trúc trang mẫu (Landing/Report page)

```
[Sticky header mỏng — logo + nav nhẹ]
        ↓
[HERO — nền kem, title lớn, subtitle cảm xúc, 2 CTA, 3 stat cards]
        ↓
[STORY / CÂU CHUYỆN — text + bullet có icon, breathing space lớn]
        ↓
[SOCIAL PROOF / VINH DANH — progress bar + danh sách donor cards]
        ↓
[CTA cuối — kêu gọi hành động, dịu không gắt]
        ↓
[Footer tối giản]
```

---

## 5. THÀNH PHẦN UI CHI TIẾT (Component Specs)

### 5.1 Header
- Chiều cao mỏng (~64px), nền trong suốt hoặc kem, **sticky**.
- Logo chữ (wordmark) thay vì hình phức tạp.
- Nav link nhỏ, weight 500, màu `--text-body`, hover → `--brand-primary`.

### 5.2 Hero Section
- **Title lớn** (Display), có thể xuống 2 dòng, màu `--text-heading`.
- **Subtitle** 1–2 câu, kể lợi ích/cảm xúc, màu `--text-muted`.
- **2 nút CTA** cạnh nhau:
  - Primary: nền `--brand-primary`, chữ trắng, border-radius **12–16px** (bo tròn mềm, không pill hoàn toàn).
  - Secondary: outline `--brand-primary`, nền trong suốt.
- **Stat cards** nổi bật: 3 thẻ nhỏ hiển thị con số lớn (100%, 5.000.000đ...) + label ngắn. Số dùng font heading, màu brand.

### 5.3 Progress Bar (Gây quỹ)
- Track: nền `--bg-secondary`, bo tròn hoàn toàn.
- Fill: gradient nhẹ `--brand-primary` → `--accent-mustard`.
- Hiển thị: số đã đạt (đậm) + mục tiêu (nhạt) + % ở cuối.
- Chiều cao 10–14px, có animation fill mượt khi scroll tới.

### 5.4 Donor / Contribution Card
```html
<div class="donor-card">
  <div class="donor-top">
    <span class="donor-name">Tên người ủng hộ</span>
    <span class="donor-amount">72.000đ</span>
  </div>
  <p class="donor-message">"Lời nhắn..."</p>
  <span class="donor-date">09/10/2026</span>
</div>
```
- Nền trắng ngà, border `--border-soft` 1px, radius 16px.
- Shadow mềm: `0 4px 16px var(--shadow-warm)`.
- **Lời nhắn** in nghiêng, màu muted, đặt trong dấu ngoặc kép.
- Tên đậm hơn số tiền; số tiền màu `--brand-primary`.

### 5.5 Story Block
- Đoạn văn max-width ~**65ch** để dễ đọc.
- Bullet có **icon nhỏ** (tim, lá, cọ vẽ) thay vì dấu chấm.
- Xen kẽ **pull-quote** hoặc câu hỏi tu từ để giữ nhịp cảm xúc.
- Khoảng trắng giữa các đoạn lớn (margin-bottom 24px).

### 5.6 Buttons

| Loại | Nền | Chữ | Border | Radius | Hover |
|---|---|---|---|---|---|
| Primary | `--brand-primary` | Trắng | none | 14px | Tối 8% + nhấc 2px |
| Secondary | Trong suốt | `--brand-primary` | 1.5px solid | 14px | Nền `--brand-soft` 30% |
| Ghost | none | `--text-body` | none | 8px | Gạch chân |

- Transition: `all 0.25s ease`.
- Focus ring rõ ràng cho accessibility (outline 2px `--accent-mustard` offset 2px).

### 5.7 Badge / Pill
- Nền `--brand-soft`, chữ `--brand-deep`, radius full (999px), padding 6px 14px, font 13px weight 600.

---

## 6. HIỆU ỨNG & CHUYỂN ĐỘNG (Motion)

- **Nguyên tắc:** Tinh tế, chậm, không phô. Thời lượng 200–400ms, easing `ease-out`.
- **Fade-up khi scroll:** `opacity 0→1`, `translateY 20px→0`, stagger 80ms giữa các card.
- **Hover card:** nhấc lên 2–4px + shadow đậm nhẹ.
- **Progress bar:** fill từ 0 → giá trị thật khi vào viewport.
- **Tuyệt đối tránh:** parallax nặng, autoplay video ồn, animation lặp vô hạn gây mất tập trung.
- Tôn trọng `prefers-reduced-motion`.

---

## 7. ACCESSIBILITY & UX WRITING (Giọng văn)

### Accessibility
- Contrast ≥ AA, focus state rõ, alt text đầy đủ, kích thước chạm ≥ 44px.
- Form label luôn hiển thị (không chỉ placeholder).

### Giọng văn (áp dụng cho nội dung ML của bạn)
- **Xưng "chúng mình/mình"** — gần gũi, không quan liêu.
- Câu ngắn, giàu hình ảnh, tránh thuật ngữ khô khan khi giải thích cho người không chuyên.
- Cấu trúc: **Vấn đề cảm xúc → Số liệu cụ thể → Giải pháp → Kêu gọi nhẹ nhàng**.
- Ví dụ chuyển thể cho ML:
  > "Bạn có biết: một mô hình dự đoán tốt có thể giúp bác sĩ phát hiện sớm bệnh chỉ trong vài giây?"
  > Thay vì: "Accuracy đạt 94.2% trên tập test."

---

## 8. CHECKLIST ÁP DỤNG CHO DỰ ÁN MACHINE LEARNING

Khi AI Agent thiết kế trang báo cáo/demo ML theo phong cách này, hãy đảm bảo:

- [ ] Nền kem ấm `#FAF6F0`, không dùng nền trắng tinh hay xám lạnh.
- [ ] Palette terracotta + nâu đất + vàng nghệ, tỷ lệ 60-30-10.
- [ ] Font heading có cá tính, body line-height 1.7, body ≥16px.
- [ ] Hero có title cảm xúc + 3 stat card (ví dụ: Accuracy, F1, Dataset size) trình bày như "con số biết kể chuyện".
- [ ] Có section **"Câu chuyện"** giải thích vì sao mô hình quan trọng (bằng ngôn ngữ đời thường).
- [ ] Biểu đồ/kết quả trình bày dạng **card bo tròn 16px, shadow ấm**, không dùng khung bảng khô cứng.
- [ ] Progress bar cho metric (ví dụ độ chính xác) với gradient brand.
- [ ] Nút CTA bo 14px, màu terracotta, hover nhấc nhẹ.
- [ ] Animation fade-up khi scroll, tôn trọng reduced-motion.
- [ ] Mọi section cách nhau 80–120px, container 1200px.
- [ ] Giọng văn "chúng mình", kể chuyện, tránh jargon thuần túy.
- [ ] Đảm bảo contrast AA và focus state rõ.

---

## 9. PROMPT MẪU CHO AI AGENT

> "Hãy thiết kế một trang báo cáo Machine Learning theo phong cách **TỰA – Nét Vẽ Yêu Thương**: nền kem ấm `#FAF6F0`, palette terracotta `#C96A4B` + nâu đất `#8C4A32` + vàng nghệ `#D9A441` (tỷ lệ 60-30-10), font heading cá tính, body 17px line-height 1.7. Bố cục: Hero với tiêu đề cảm xúc + 3 stat card (Accuracy/F1/Dataset), section 'Câu chuyện mô hình' kể bằng ngôn ngữ đời thường với bullet có icon, card kết quả bo tròn 16px shadow ấm, progress bar gradient cho metric chính, CTA dịu. Khoảng cách section 100px, container 1200px, animation fade-up nhẹ tôn trọng reduced-motion. Giọng văn xưng 'chúng mình', tránh jargon khô khan."

---

## 10. GHI CHÚ QUAN TRỌNG

- Phân tích trên dựa trên **nội dung và phong cách tổng thể** của trang (cấu trúc section, hệ thống thông điệp, cách trình bày stat/progress/donor card). Các mã màu cụ thể là **suy luận theo tinh thần thiết kế** (ấm, thủ công, charity) — bạn nên tinh chỉnh khi đối chiếu trực tiếp bằng DevTools trên trang gốc.
- Điểm mạnh nhất cần "ăn theo": **sự hài hòa giữa câu chuyện cảm xúc và con số cụ thể**, cùng nhịp thở rộng rãi giữa các section — đây là thứ tạo nên cảm giác "đẹp và dễ chịu" mà bạn cảm nhận được.

---
