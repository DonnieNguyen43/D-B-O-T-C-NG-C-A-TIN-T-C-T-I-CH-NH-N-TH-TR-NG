"""
news_scraper.py
Module thu thập tin tức tài chính thời gian thực từ Google News Financial, Yahoo Finance, CNBC Markets.
Thiết kế với cơ chế đa nguồn, timeout 5 giây, bảo vệ try-except an toàn tuyệt đối và Fallback Cache 10 bản tin.
"""

import time
import requests
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime

# Fallback database chất lượng cao chuẩn tài chính vĩ mô (10 bản tin đại diện)
FALLBACK_NEWS_DATABASE = [
    {
        "title": "Federal Reserve unexpectedly hikes benchmark interest rate by 75 bps to counter inflation surge",
        "description": "US central bank delivers aggressive rate increase as consumer prices hit 40-year high, triggering widespread bond selloff and equity volatility across global indices.",
        "source": "CNBC Markets",
        "published": "Vừa cập nhật (Fallback)",
        "expected_tier": "HIGH"
    },
    {
        "title": "Global banking crisis fears escalate as major regional lender halts stock trading amid liquidity run",
        "description": "Regulators step in after deposit outflows accelerate beyond emergency reserves, prompting systemic contagion warnings across North American and European financial institutions.",
        "source": "Yahoo Finance",
        "published": "15 phút trước (Fallback)",
        "expected_tier": "HIGH"
    },
    {
        "title": "Crude oil plunges 8% as OPEC+ unexpectedly discusses production quota increase amidst demand slowdown",
        "description": "Energy commodity futures collapse following surprise alliance announcements, putting intense downward pressure on multinational energy sector equities.",
        "source": "Google News Financial",
        "published": "42 phút trước (Fallback)",
        "expected_tier": "HIGH"
    },
    {
        "title": "Middle East geopolitical tensions spike following military confrontation near key maritime trade straits",
        "description": "Commercial shipping routes face immediate rerouting, pushing marine insurance premiums up 400% and threatening critical global energy supply corridors.",
        "source": "CNBC Markets",
        "published": "55 phút trước (Fallback)",
        "expected_tier": "HIGH"
    },
    {
        "title": "Apple reports quarterly earnings: Services revenue reaches record $24.2 billion while iPhone shipments steady",
        "description": "Fiscal second quarter net income met Wall Street expectations with modest gross margin expansion driven by recurring subscription ecosystems.",
        "source": "Yahoo Finance",
        "published": "1 giờ trước (Fallback)",
        "expected_tier": "MEDIUM"
    },
    {
        "title": "Semiconductor manufacturing equipment suppliers maintain conservative annual revenue guidance",
        "description": "Chip tooling orders remain resilient in server hardware while consumer electronics segment experiences slight sequential inventory adjustments.",
        "source": "CNBC Markets",
        "published": "2 giờ trước (Fallback)",
        "expected_tier": "MEDIUM"
    },
    {
        "title": "Commercial real estate developers adjust debt refinancing timelines ahead of central bank symposium",
        "description": "Metropolitan office occupancy rates level off at 64% as corporate tenants finalize hybrid working protocols for the coming fiscal calendar.",
        "source": "Google News Financial",
        "published": "3 giờ trước (Fallback)",
        "expected_tier": "MEDIUM"
    },
    {
        "title": "Company files routine Form 8-K announcing date for upcoming Annual General Meeting of Shareholders",
        "description": "The board of directors submitted standard regulatory documentation with the Securities and Exchange Commission regarding proxy ballot procedures.",
        "source": "Google News Financial",
        "published": "4 giờ trước (Fallback)",
        "expected_tier": "LOW"
    },
    {
        "title": "Regional enterprise software vendor issues ordinary notification of corporate office address relocation",
        "description": "Administrative headquarters will transition to adjacent commercial complex next month with zero operational disruption to client services.",
        "source": "Yahoo Finance",
        "published": "5 giờ trước (Fallback)",
        "expected_tier": "LOW"
    },
    {
        "title": "Audit committee reappoints independent accounting firm for statutory annual compliance verification",
        "description": "Standard corporate governance oversight reaffirms existing external auditing engagement following customary triennial review committee session.",
        "source": "CNBC Markets",
        "published": "6 giờ trước (Fallback)",
        "expected_tier": "LOW"
    }
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5"
}

def clean_html(raw_html: str) -> str:
    """Lọc bỏ toàn bộ thẻ HTML khỏi tóm tắt RSS"""
    if not raw_html:
        return ""
    try:
        soup = BeautifulSoup(str(raw_html), "html.parser")
        return soup.get_text(separator=" ").strip()
    except Exception:
        return str(raw_html).strip()

def fetch_google_news(limit: int = 10):
    """Thu thập từ Google News Financial RSS với timeout=5s và try-except"""
    url = "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en"
    articles = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=5)
        if resp.status_code == 200:
            feed = feedparser.parse(resp.content)
            for entry in feed.entries[:limit]:
                title = entry.get("title", "").strip()
                if " - " in title:
                    title = title.rsplit(" - ", 1)[0]
                summary = clean_html(entry.get("summary", ""))
                pub_date = entry.get("published", datetime.now().strftime("%Y-%m-%d %H:%M"))
                articles.append({
                    "title": title,
                    "description": summary if summary else title,
                    "source": "Google News Financial",
                    "published": pub_date,
                    "link": entry.get("link", "#")
                })
    except Exception as e:
        print(f"[news_scraper] Google News error (handled): {e}")
    return articles

def fetch_yahoo_finance(limit: int = 10):
    """Thu thập từ Yahoo Finance RSS với timeout=5s và try-except"""
    url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US"
    articles = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=5)
        if resp.status_code == 200:
            feed = feedparser.parse(resp.content)
            for entry in feed.entries[:limit]:
                title = entry.get("title", "").strip()
                summary = clean_html(entry.get("summary", ""))
                pub_date = entry.get("published", datetime.now().strftime("%Y-%m-%d %H:%M"))
                articles.append({
                    "title": title,
                    "description": summary if summary else title,
                    "source": "Yahoo Finance",
                    "published": pub_date,
                    "link": entry.get("link", "#")
                })
    except Exception as e:
        print(f"[news_scraper] Yahoo Finance error (handled): {e}")
    return articles

def fetch_cnbc_markets(limit: int = 10):
    """Thu thập từ CNBC Markets RSS với timeout=5s và try-except"""
    url = "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"
    articles = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=5)
        if resp.status_code == 200:
            feed = feedparser.parse(resp.content)
            for entry in feed.entries[:limit]:
                title = entry.get("title", "").strip()
                summary = clean_html(entry.get("summary", ""))
                pub_date = entry.get("published", datetime.now().strftime("%Y-%m-%d %H:%M"))
                articles.append({
                    "title": title,
                    "description": summary if summary else title,
                    "source": "CNBC Markets",
                    "published": pub_date,
                    "link": entry.get("link", "#")
                })
    except Exception as e:
        print(f"[news_scraper] CNBC error (handled): {e}")
    return articles

def fetch_all_financial_news(limit_per_source: int = 6):
    """
    Thu thập tổng hợp từ 3 nguồn. Nếu mạng lỗi hoặc rỗng, tự động trả về FALLBACK_NEWS_DATABASE.
    Đảm bảo 100% không bao giờ làm crash ứng dụng.
    """
    all_news = []
    try:
        # 1. Google News
        g_news = fetch_google_news(limit=limit_per_source)
        all_news.extend(g_news)
        
        # 2. Yahoo Finance
        y_news = fetch_yahoo_finance(limit=limit_per_source)
        all_news.extend(y_news)
        
        # 3. CNBC Markets
        c_news = fetch_cnbc_markets(limit=limit_per_source)
        all_news.extend(c_news)
    except Exception as e:
        print(f"[news_scraper] Global fetch error (handled): {e}")
        return FALLBACK_NEWS_DATABASE.copy(), True
        
    if not all_news:
        print("[news_scraper] Live feeds unavailable. Activating Fallback Cache.")
        return FALLBACK_NEWS_DATABASE.copy(), True
        
    return all_news, False
