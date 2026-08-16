"""
Файл: cross_encoder_reranker.py
Неделя 2: Финальное переранжирование через Cross-Encoder Reranker.

ЧТО ЗДЕСЬ ДОЛЖНО БЫТЬ РЕАЛИЗОВАНО:
1. Класс/Функция Reranker:
   - Загружает модель Cross-Encoder (например, `cross-encoder/ms-marco-MiniLM-L-6-v2` или `BAAI/bge-reranker-base`).
   - Объяснение разницы: Bi-Encoder генерирует векторы отдельно, а Cross-Encoder пропускает парную связку (Query, Document) через все слои Attention одновременно для максимальной точности.

2. Функция rerank(query, candidate_docs, top_n=5):
   - Формирует пары [(query, doc1), (query, doc2), ..., (query, doc20)].
   - Пропускает пары через Cross-Encoder и получает точный скор релевантности для каждой пары.
   - Сортирует список документов по убыванию релевантности.
   - Срезает результат до Top-N (например, Top-5 самых точных контекстов) для передачи в LLM.
"""
