"""
ПРОМЫШЛЕННЫЙ МИНИ-ПРОЕКТ НЕДЕЛИ 3: Autonomous Multi-Step Database & Analytics Agent with Constrained Decoding (Pydantic v2 + SSE Streaming).

ОТРАСЛЕВАЯ ЗАДАЧА:
Разработать защищенного автономного ИИ-аналитика с поддержкой грамматического вывода Pydantic v2,
самовосстановлением при SQL-ошибках и асинхронным стримингом цепочки мыслей в веб-интерфейс.

АРХИТЕКТУРА И ШАГИ РЕАЛИЗАЦИИ:

1. СХЕМЫ ДАННЫХ (`PydanticV2Schemas`):
   - `AgentPlan`: {thought_process: str, required_tables: List[str], strategy: str}
   - `SQLQueryOutput`: {sql_query: str, explanation: str, is_read_only: bool}
   - `SQLErrorCorrection`: {failed_sql: str, error_message: str, corrective_action: str, new_sql: str}
   - `FinalAnalyticsReport`: {summary: str, key_metrics: Dict[str, Any], chart_config: Dict[str, Any]}

2. КЛАСС `SafeSQLExecutor`:
   - Загружает базу данных e-commerce (`orders`, `customers`, `products`, `refunds`).
   - Проверяет SQL-запрос на отсутствие мутирующих команд (DROP, DELETE, UPDATE, INSERT, ALTER).
   - Выполняет запрос в режиме Read-Only и возвращает результат в виде списка словарей или подтягивает текст ошибки.

3. КЛАСС `AutonomousAnalyticsAgent`:
   - Содержит State Machine (Состояния: `PLANNING`, `GENERATING_SQL`, `EXECUTING`, `CORRECTING`, `SYNTHESIZING`).
   - Изучает схему БД через `PRAGMA table_info()`.
   - Генерирует детерминированные Pydantic-структуры через Instructor / Outlines.
   - Метод `run_with_self_correction(user_query, max_retries=3)`:
     * Если `SafeSQLExecutor` возвращает ошибку, переходит в состояние `CORRECTING`.
     * Формирует объект `SQLErrorCorrection` и исправляет SQL-запрос.

4. МОДУЛЬ `FastAPISSEServer`:
   - Создает асинхронный эндпоинт `POST /api/v1/analytics/stream`.
   - Использует `async generator` для трансляции SSE-событий клиентов в формате:
     data: {"state": "PLANNING", "thought": "Изучаю схему таблиц orders и customers..."}
     data: {"state": "EXECUTING", "sql": "SELECT ..."}
     data: {"state": "CORRECTING", "error": "Unknown column ...", "fix": "..."}
     data: {"state": "COMPLETED", "report": {...}}
"""
