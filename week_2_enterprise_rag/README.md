# 📅 Неделя 2: Enterprise RAG & Advanced Retrieval Architecture

## 🎯 Цели недели
Научиться создавать промышленные системы поиска знаний (RAG), которые НЕ теряют документы и не галлюцинируют.

---

## 📋 Чек-лист темы
- [ ] Ограничения обычного `VectorStore.similarity_search`
- [ ] Алгоритмы индексации: HNSW vs Flat/IVF
- [ ] **Hybrid Search:** Плотный поиск (Dense Vectors) + Разряженный поиск (BM25)
- [ ] **Reranking:** Переранжирование Top-20 кандидатов через Cross-Encoder
- [ ] Оценка качества поиска: Метрики Hit Rate@K и MRR (Mean Reciprocal Rank)

---

## 📁 Файлы проекта

1. **`document_parser.py`** — Парсинг текстовых файлов/PDF и семантическое разбиение на чанки с метаданными.
2. **`hybrid_retriever.py`** — Настройка плотного векторного поиска и разряженного BM25 с объединением через Reciprocal Rank Fusion (RRF).
3. **`cross_encoder_reranker.py`** — Переранжирование кандидатов через Cross-Encoder модель (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
4. **`eval_rag_metrics.py`** — Расчёт метрик качества поиска (Hit Rate@K, MRR).
