# 📅 Неделя 4: Evals, QLoRA Fine-Tuning и MLOps Инференса

## 🎯 Цели недели
Научиться измерять качество систем ИИ (Evals), дообучать локальные модели под узкие бизнес-задачи (QLoRA) и разворачивать высокоскоростной инференс на vLLM.

---

## 📋 Чек-лист темы
- [ ] Оценка систем ИИ (Evals): Синтетические тесты, LLM-as-a-Judge, Ragas
- [ ] Механика LoRA / QLoRA (4-bit квантование, адаптеры, ранг $R$)
- [ ] Обучение через Unsloth / PEFT + TRL (SFTTrainer)
- [ ] Оптимизация инференса на vLLM (PagedAttention & Continuous Batching)

---

## 📁 Файлы проекта

1. **`dataset_generator.py`** — Генерация и форматирование датасета из 500+ записей в формат Alpaca / ChatML.
2. **`qlora_finetune.py`** — Дообучение локальной модели (Qwen 2.5 3B / Llama 3.2 3B) с помощью QLoRA.
3. **`vllm_deploy.py`** — Развертывание инференса дообученной модели в движке vLLM.
4. **`eval_pipeline.py`** — Автоматическая проверка качества дообученной модели против базовой (Accuracy, F1-Score, Latency, LLM-as-a-Judge).
