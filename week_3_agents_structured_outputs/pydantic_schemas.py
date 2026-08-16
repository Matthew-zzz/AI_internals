"""
Файл: pydantic_schemas.py
Неделя 3: Схемы Pydantic v2 для детерминированного контроля вывода LLM.

ЧТО ЗДЕСЬ ДОЛЖНО БЫТЬ РЕАЛИЗОВАНО:
1. Pydantic Модель для SQL Агента:
   - SQLQueryAnalysis:
     * thought_process: str (Ход мыслей агента и обоснование выбора таблиц)
     * sql_query: str (Сгенерированный SQL-запрос)
     * expected_columns: List[str] (Ожидаемый список колонок)

2. Pydantic Модель для ответа пользователю:
   - FinalAnalyticsReport:
     * summary: str (Краткий бизнес-вывод по найденным данным)
     * data_table: List[Dict[str, Any]] (Данные из базы)
     * chart_type: Literal["bar", "line", "pie", "none"] (Тип графика для дашборда)

3. Настройка интеграции с Instructor / Outlines:
   - Валидация вывода через Pydantic.Если модель отдает невалидный тип данных (например, строку вместо числа), Pydantic выбрасываетValidationError, и агент уходит на автоматический повторный запрос (Retry loop).
"""
