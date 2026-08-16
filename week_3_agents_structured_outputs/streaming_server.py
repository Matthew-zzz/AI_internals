"""
Файл: streaming_server.py
Неделя 3: FastAPI сервер с трансляцией цепочки размышлений агента по SSE.

ЧТО ЗДЕСЬ ДОЛЖНО БЫТЬ РЕАЛИЗОВАНО:
1. Создание FastAPI приложения.

2. Эндпоинт POST /api/v1/agent/query:
   - Принимает { "query": "Покажи топ-5 продаваемых товаров за июль" }
   - Возвращает StreamingResponse с медиатипом `text/event-stream`.

3. Генератор событий SSE (Server-Sent Events):
   - Транслирует шаги агента клиентскому приложению в реальном времени:
     * event: "thought", data: {"step": "Исследую схему БД..."}
     * event: "sql", data: {"query": "SELECT * FROM sales..."}
     * event: "status", data: {"message": "SQL выполнен успешно"}
     * event: "result", data: {"report": "..."}
"""
