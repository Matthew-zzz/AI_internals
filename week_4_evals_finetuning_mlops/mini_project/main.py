"""
ПРОМЫШЛЕННЫЙ МИНИ-ПРОЕКТ НЕДЕЛИ 4: Enterprise QLoRA Fine-Tuning & MLOps CI/CD Evaluation Pipeline for Customer Support Intent Classification.

ОТРАСЛЕВАЯ ЗАДАЧА:
Спроектировать и выполнить полный MLOps-пайплайн дообучения локальной модели (Qwen 2.5 3B / Llama 3.2 3B)
с использованием QLoRA, её вывода в продакшн через vLLM и автоматизированным заполнением отчёта Evals.

АРХИТЕКТУРА И ШАГИ РЕАЛИЗАЦИИ:

1. МОДУЛЬ `DatasetPipeline`:
   - Формирует датасет из 500+ обращений клиентов с финтех-классификацией (`category`, `urgency`, `department_routing`).
   - Конвертирует в формат ChatML (`messages: [{role: system/user/assistant}]`).
   - Сохраняет `dataset_train.jsonl` (400 примеров) и `dataset_eval.jsonl` (100 примеров).

2. МОДУЛЬ `QLoRATrainer`:
   - Настраивает 4-битное квантование NF4 через BitsAndBytesConfig.
   - Подключает LoRA-адаптеры к слоям Attention и MLP (`r=16`, `lora_alpha=32`, `lora_dropout=0.05`).
   - Запускает `SFTTrainer` из библиотеки TRL (3 эпохи, fp16/bf16, paged_adamw_8bit).
   - Сохраняет скомпилированные веса LoRA в папочку `./models/support_classifier_lora`.

3. МОДУЛЬ `vLLMInferenceServer`:
   - Инициализирует движок vLLM: `LLM(model="Qwen/Qwen2.5-3B-Instruct", enable_lora=True)`.
   - Загружает обученный адаптер через `LoRARequest`.
   - Выполняет пакетный инференс тестового набора с PagedAttention и непрерывным батчингом.

4. МОДУЛЬ `CIEvalPipeline`:
   - Прогоняет `dataset_eval.jsonl` через 2 модели:
     * Model 1: Базовая Qwen 2.5 3B (Zero-Shot промпт).
     * Model 2: Наша QLoRA Fine-Tuned модель.
   - Подсчитывает метрики:
     * Accuracy & Micro/Macro F1-Score по категориям.
     * LLM-as-a-Judge score (оценка корректности аргументов через GPT-4o / Claude).
     * P95 & P99 Latency (задержка ответа в миллисекундах).
   - Генерирует сводную сравнительную таблицу и сохраняет отчёт `eval_report.md`.
"""
