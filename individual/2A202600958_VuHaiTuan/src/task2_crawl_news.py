"""Task 2 — Crawl news articles related to drug-law incidents.

Default mode writes offline crawled-article fixtures with the same metadata shape
as Crawl4AI output. Set `ENABLE_LIVE_CRAWL=1` and fill `ARTICLE_URLS` with real
URLs to crawl live pages when network access is available.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .common import LANDING_DIR, ensure_dir, write_json
except ImportError:  # pragma: no cover
    from common import LANDING_DIR, ensure_dir, write_json


DATA_DIR = LANDING_DIR / "news"

ARTICLE_URLS = [
    "https://example.com/phap-luat/nghe-si-va-nguyen-tac-suy-doan-vo-toi",
    "https://example.com/giai-tri/kiem-chung-tin-nghe-si-lien-quan-ma-tuy",
    "https://example.com/phap-luat/xu-ly-hanh-vi-su-dung-trai-phep-chat-ma-tuy",
    "https://example.com/xa-hoi/trach-nhiem-truyen-thong-khi-dua-tin-ma-tuy",
    "https://example.com/phap-luat/phan-biet-tin-dieu-tra-khoi-to-xet-xu",
]


OFFLINE_ARTICLES = [
    {
        "title": "Nguyên tắc suy đoán vô tội khi đưa tin nghệ sĩ liên quan ma túy",
        "url": ARTICLE_URLS[0],
        "source_name": "Offline Legal News Fixture",
        "content_markdown": """
# Nguyên tắc suy đoán vô tội khi đưa tin nghệ sĩ liên quan ma túy

Khi một nghệ sĩ hoặc người nổi tiếng bị nhắc đến trong vụ việc liên quan đến ma
túy, thông tin cần được trình bày theo nguyên tắc suy đoán vô tội. Tin tức nên
phân biệt rõ thông tin ban đầu, thông tin xác minh, quyết định khởi tố, cáo
trạng, phiên tòa và bản án có hiệu lực pháp luật. Việc khẳng định một cá nhân
có tội trước khi có bản án có thể gây ảnh hưởng nghiêm trọng đến danh dự, quyền
riêng tư và quyền được xét xử công bằng.

Độc giả nên kiểm tra nguồn tin chính thống, đối chiếu phát ngôn từ cơ quan có
thẩm quyền và tránh chia sẻ thông tin chưa được xác minh. Với RAG chatbot, phần
citation phải chỉ rõ nguồn và năm để người dùng biết câu trả lời dựa trên tài
liệu nào.
""",
    },
    {
        "title": "Kiểm chứng nguồn tin trong các vụ việc nghệ sĩ bị nghi liên quan chất cấm",
        "url": ARTICLE_URLS[1],
        "source_name": "Offline Legal News Fixture",
        "content_markdown": """
# Kiểm chứng nguồn tin trong các vụ việc nghệ sĩ bị nghi liên quan chất cấm

Các vụ việc liên quan đến nghệ sĩ và chất cấm thường lan truyền nhanh trên mạng
xã hội. Người đọc cần kiểm tra ngày đăng, cơ quan phát hành, trích dẫn từ cơ
quan chức năng và tình trạng pháp lý của vụ việc. Tin đồn, bài tổng hợp chưa rõ
nguồn hoặc ảnh chụp màn hình không đủ để kết luận một cá nhân vi phạm pháp luật.

Khi xây dựng hệ thống truy xuất, bài báo loại này nên được đánh nhãn là news,
lưu URL gốc, tiêu đề, ngày crawl và nội dung markdown. Retrieval pipeline cần
ưu tiên văn bản pháp luật khi câu hỏi yêu cầu quy định pháp lý, và dùng tin tức
như nguồn bối cảnh khi câu hỏi liên quan truyền thông hoặc kiểm chứng thông tin.
""",
    },
    {
        "title": "Xử lý hành vi sử dụng trái phép chất ma túy theo hướng quản lý và cai nghiện",
        "url": ARTICLE_URLS[2],
        "source_name": "Offline Legal News Fixture",
        "content_markdown": """
# Xử lý hành vi sử dụng trái phép chất ma túy theo hướng quản lý và cai nghiện

Hành vi sử dụng trái phép chất ma túy có thể kéo theo biện pháp quản lý, xác
định tình trạng nghiện và áp dụng hình thức cai nghiện phù hợp. Luật Phòng,
chống ma túy 2021 quy định các hình thức cai nghiện tự nguyện tại gia đình,
cộng đồng, cơ sở cai nghiện và cai nghiện bắt buộc tại cơ sở cai nghiện.

Trong trường hợp người được truyền thông nhắc đến là nghệ sĩ, việc đưa tin cần
tránh mô tả giật gân và cần nhấn mạnh quyền tiếp cận hỗ trợ, điều trị, tư vấn.
Các nội dung chưa được xác minh từ cơ quan chức năng không nên được xem là căn
cứ kết luận.
""",
    },
    {
        "title": "Trách nhiệm truyền thông khi đưa tin về ma túy và người nổi tiếng",
        "url": ARTICLE_URLS[3],
        "source_name": "Offline Legal News Fixture",
        "content_markdown": """
# Trách nhiệm truyền thông khi đưa tin về ma túy và người nổi tiếng

Truyền thông có trách nhiệm đưa tin thận trọng, không quy kết tội danh khi chưa
có kết luận của cơ quan có thẩm quyền. Đối với chủ đề ma túy, bài viết nên ưu
tiên bối cảnh pháp luật, nguy cơ sức khỏe, trách nhiệm phòng ngừa và thông tin
kiểm chứng thay vì khai thác đời tư người nổi tiếng.

Một hệ thống hỏi đáp dựa trên tài liệu cần trả lời có giới hạn: nếu dữ liệu chỉ
nói về nguyên tắc đưa tin, hệ thống không được suy đoán ai có tội. Nếu nguồn
hiện có không chứa thông tin về một cá nhân cụ thể, câu trả lời phải nói rõ
không thể xác minh từ nguồn hiện có.
""",
    },
    {
        "title": "Phân biệt thông tin điều tra, khởi tố, xét xử và bản án trong vụ việc ma túy",
        "url": ARTICLE_URLS[4],
        "source_name": "Offline Legal News Fixture",
        "content_markdown": """
# Phân biệt thông tin điều tra, khởi tố, xét xử và bản án trong vụ việc ma túy

Trong một vụ việc liên quan đến ma túy, thông tin điều tra là dữ liệu ban đầu
được cơ quan chức năng xác minh. Quyết định khởi tố thể hiện việc cơ quan tố
tụng xem xét dấu hiệu tội phạm, nhưng chưa đồng nghĩa với kết luận có tội. Cáo
trạng, xét xử sơ thẩm, phúc thẩm và bản án có hiệu lực là các giai đoạn pháp lý
khác nhau.

Đối với các vụ việc được công chúng chú ý vì có nghệ sĩ hoặc người nổi tiếng,
việc phân biệt giai đoạn tố tụng giúp tránh đưa tin sai lệch. Đây cũng là lý do
hệ thống RAG cần hiển thị citation, source document và relevance score để người
dùng tự kiểm tra căn cứ của câu trả lời.
""",
    },
]


def setup_directory() -> Path:
    return ensure_dir(DATA_DIR)


def _fixture_for_url(url: str) -> dict[str, Any]:
    for article in OFFLINE_ARTICLES:
        if article["url"] == url:
            return article
    return OFFLINE_ARTICLES[0]


async def crawl_article(url: str) -> dict[str, Any]:
    """Crawl one article or return an offline fixture.

    Live crawling is opt-in because automated tests and classroom demos should
    not depend on network stability.
    """

    if os.getenv("ENABLE_LIVE_CRAWL") == "1":
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            metadata = result.metadata or {}
            return {
                "url": url,
                "title": metadata.get("title") or "Untitled article",
                "source_name": metadata.get("site_name") or "Crawl4AI",
                "date_crawled": datetime.now(timezone.utc).isoformat(),
                "content_markdown": result.markdown or "",
            }

    article = _fixture_for_url(url).copy()
    article["date_crawled"] = datetime.now(timezone.utc).isoformat()
    return article


async def crawl_all(urls: list[str] | None = None, force: bool = False) -> list[Path]:
    """Crawl or create all article files in `data/landing/news/`."""

    setup_directory()
    saved: list[Path] = []
    for index, url in enumerate(urls or ARTICLE_URLS, start=1):
        output_path = DATA_DIR / f"article_{index:02d}.json"
        if output_path.exists() and not force:
            saved.append(output_path)
            continue
        article = await crawl_article(url)
        write_json(output_path, article)
        saved.append(output_path)
    return saved


def write_offline_articles(force: bool = False) -> list[Path]:
    """Synchronous helper used by tests/bootstrap."""

    return asyncio.run(crawl_all(force=force))


if __name__ == "__main__":
    files = asyncio.run(crawl_all(force=True))
    print(f"Created/verified {len(files)} news articles in {DATA_DIR}")
