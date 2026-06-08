# Bài Cá Nhân - Tuấn

Pipeline cá nhân triển khai đủ 10 task của Day08 RAG Pipeline v2.

## Mục Tiêu

Xây dựng pipeline RAG offline cho chủ đề pháp luật Việt Nam về ma túy và tin tức
liên quan đến nghệ sĩ/chất cấm:

1. Thu thập raw legal documents.
2. Tạo/crawl raw news articles.
3. Convert toàn bộ dữ liệu sang Markdown.
4. Chunking và indexing vào vector store local.
5. Semantic search.
6. Lexical search BM25.
7. Reranking.
8. PageIndex vectorless fallback.
9. Retrieval pipeline hybrid.
10. Generation có citation.

## Cấu Trúc

```text
individual/tuan/
├── data/
│   ├── landing/
│   │   ├── legal/          # raw legal PDF/DOCX fixtures
│   │   └── news/           # raw crawled news JSON
│   ├── standardized/
│   │   ├── legal/          # converted markdown legal docs
│   │   └── news/           # converted markdown news docs
│   └── index/
│       ├── vector_store.json
│       └── pageindex_manifest.json
├── src/
│   ├── common.py
│   ├── task1_collect_legal_docs.py
│   ├── task2_crawl_news.py
│   ├── task3_convert_markdown.py
│   ├── task4_chunking_indexing.py
│   ├── task5_semantic_search.py
│   ├── task6_lexical_search.py
│   ├── task7_reranking.py
│   ├── task8_pageindex_vectorless.py
│   ├── task9_retrieval_pipeline.py
│   └── task10_generation.py
└── tests/
    └── test_individual.py
```

## Cách Chạy

Chạy lần lượt:

```bash
python3 individual/tuan/src/task1_collect_legal_docs.py
python3 individual/tuan/src/task2_crawl_news.py
python3 individual/tuan/src/task3_convert_markdown.py
python3 individual/tuan/src/task4_chunking_indexing.py
```

Test retrieval:

```bash
python3 individual/tuan/src/task9_retrieval_pipeline.py
```

Test generation có citation:

```bash
python3 individual/tuan/src/task10_generation.py
```

Chạy test chấm điểm:

```bash
python3 -m unittest discover individual/tuan/tests -v
```

Hoặc:

```bash
pytest individual/tuan/tests/ -v
```

## Ghi Chú Kỹ Thuật

- Chunking: recursive character chunking, `CHUNK_SIZE=800`, `CHUNK_OVERLAP=120`.
- Embedding: `local-hashed-tfidf-384`, chạy offline, không cần tải model.
- Vector store: JSON local tại `data/index/vector_store.json`.
- Lexical search: BM25 tự implement.
- Reranking: local cross-encoder-style heuristic, không cần API key.
- PageIndex: fallback local vectorless, giữ đúng contract `source="pageindex"`.
- Generation: mặc định local fallback có citation. Nếu muốn gọi OpenAI thật, set
  `INDIVIDUAL_USE_OPENAI=1` và `OPENAI_API_KEY` trong `.env`.

## Trạng Thái Hiện Tại

- Legal raw files: 4.
- News JSON files: 5.
- Markdown files: 9.
- Vector index: 27 chunks.
- Test suite: 35 tests pass.
