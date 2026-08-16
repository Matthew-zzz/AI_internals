"""
Файл: dataset_generator.py
Неделя 4: Создание и подготовка обучающего датасета.

ЧТО ЗДЕСЬ ДОЛЖНО БЫТЬ РЕАЛИЗОВАНО:
1. Функция generate_synthetic_dataset(num_samples=500):
   - Создает или генерирует специфические бизнес-обращения/логи (например, классификация категорий поддержки клиентов).
   - Структура в формате ChatML / OpenAI Messages:
     [
       {"messages": [
         {"role": "system", "content": "Ты - классификатор обращений..."},
         {"role": "user", "content": "Не могу войти в личный кабинет"},
         {"role": "assistant", "content": "{\"category\": \"AUTH_ERROR\", \"priority\": \"HIGH\"}"}
       ]}
     ]

2. Функция train_test_split(dataset, ratio=0.8):
   - Разделяет датасет на train (80%) для обучения и test (20%) для валидации через Evals.
   - Сохраняет результаты в `dataset_train.jsonl` и `dataset_test.jsonl`.
"""
