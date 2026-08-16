# ЭТОТ README.md оформлен нейросетью.
# 🧠 AI Internals & LLM Engineering Track

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![vLLM](https://img.shields.io/badge/vLLM-0080FF?style=for-the-badge&logo=fastapi&logoColor=white)](https://vllm.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

Полный практический курс по внутреннему устройству больших языковых моделей (LLM), продвинутым архитектурам поиска (Enterprise RAG), детерминированным агентам и MLOps инференса.

---

## 🎯 Озимут Проекта

Данный репозиторий содержит практические реализации фундаментальных алгоритмов ИИ без использованием абстрактных библиотек "высокого уровня". Каждая неделя посвящена отдельному слою современной AI-инженерии: от сырого внимания в PyTorch до оптимизированного инференса на GPU.

```
       +-------------------------------------------------------+
       |                  AI INTERNALS TRACK                   |
       +-------------------------------------------------------+
                                   |
        +--------------------------+--------------------------+
        |                                                     |
  [Week 1: LLM Mechanics]                            [Week 2: Enterprise RAG]
  * Custom Autoregressive Loop                       * Semantic Chunking & Parsing
  * Manual KV-Cache & Profiling                      * Hybrid Search (Dense + BM25)
  * Sampling (Temp/Top-P/Top-K)                      * Reranking & Hit@K/MRR Evals
        |                                                     |
        +--------------------------+--------------------------+
                                   |
        +--------------------------+--------------------------+
        |                                                     |
  [Week 3: Agents & Schemas]                         [Week 4: Evals & Fine-Tuning]
  * Pydantic v2 Grammar Constraints                   * Synthetic Dataset Generation
  * Self-Correcting SQL Agent                        * 4-bit QLoRA Adaptation
  * FastAPI SSE Streaming                            * vLLM Deployment & LLM Judge
        +-----------------------------------------------------+
```

---

## 📚 Структура Курса

### 📅 [Неделя 1: Фундамент ИИ и механика Transformer](./week_1_llm_mechanics)
**Тема:** Инференс под капотом, авторегрессионный цикл и оптимизация памяти.
- `mini_llm_generator.py` — Ручной инференс-цикл на PyTorch/Transformers: `Forward Pass` $\to$ `Logits` $\to$ `Temperature Scaling` $\to$ `Top-P/Top-K` $\to$ `Categorical Sampling`.
- `kv_cache_profiler.py` — Профайлинг скорости генерации (TPS) и потребления VRAM/RAM с ручной поддержкой `past_key_values`.
- `mini_project/` — Готовый генератор текста с настраиваемыми стратегиями сэмплинга.

### 📅 [Неделя 2: Enterprise RAG & Advanced Retrieval Architecture](./week_2_enterprise_rag)
**Тема:** Промышленные системы поиска знаний с минимальным процентом галлюцинаций.
- `document_parser.py` — Семантический парсинг и гибкое разбиение документов на чанки.
- `hybrid_retriever.py` — Гибридный поиск: Плотные векторы (FAISS/Dense) + Разряженный поиск (BM25) с объединением через **Reciprocal Rank Fusion (RRF)**.
- `cross_encoder_reranker.py` — Переранжирование кандидатов с помощью Cross-Encoder (`ms-marco-MiniLM-L-6-v2`).
- `eval_rag_metrics.py` — Подсчет метрик Hit Rate@K и MRR (Mean Reciprocal Rank).
- `mini_project/` — Завершенная служба поиска знаний.

### 📅 [Неделя 3: Автономные Агенты и Structured Outputs](./week_3_agents_structured_outputs)
**Тема:** Принуждение вероятностных моделей к вызову строгого JSON и рекурсивному мышлению.
- `pydantic_schemas.py` — Детерминированные Pydantic v2 схемы для генерации структуры данных.
- `sql_agent.py` — Автономный SQL-агент с циклом самоисправления ошибок (Self-Correction Loop).
- `streaming_server.py` — FastAPI сервис с потоковой трансляцией мыслей агента через **Server-Sent Events (SSE)**.
- `mini_project/` — Интерактивный агент-ассистент.

### 📅 [Неделя 4: Evals, QLoRA Fine-Tuning и MLOps Инференса](./week_4_evals_finetuning_mlops)
**Тема:** Измерение качества, дообучение локальных моделей и хайлоад инференс.
- `dataset_generator.py` — Генерация синтетических данных для SFT в формате ChatML / Alpaca.
- `qlora_finetune.py` — Дообучение 4-bit квантованных моделей (Qwen / Llama) с помощью PEFT и LoRA-адаптеров.
- `vllm_deploy.py` — Высокоскоростной инференс с движком **vLLM** (PagedAttention, Continuous Batching).
- `eval_pipeline.py` — Автоматический Evals пайплайн: Accuracy, F1-Score, Latency и методология **LLM-as-a-Judge**.
- `mini_project/` — Финальный пайплайн тонкой настройки и деплоя.

---

## 🛠️ Технологический Стек

- **Deep Learning / Core:** PyTorch, Transformers, HuggingFace Hub, PEFT, TRL, BitsAndBytes.
- **RAG & Search:** Rank-BM25, Sentence-Transformers, Cross-Encoders, FAISS / Vector Indexing.
- **Agents & Structured IO:** Pydantic v2, FastAPI, Server-Sent Events (SSE).
- **MLOps & Evals:** vLLM, Ragas, Unsloth, Synthetic Data Pipelines.

---

## 🚀 Быстрый Старт

### 1. Клонирование репозитория
```bash
git clone https://github.com/Matthew-zzz/AI_internals.git
cd AI_internals
```

### 2. Создание и активация виртуального окружения
```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1
```

### 3. Установка зависимостей
```bash
pip install --upgrade pip
pip install torch transformers sentence-transformers pydantic fastapi uvicorn rank-bm25
```

---

## 📁 Дерево Проекта

```
AI_internals/
├── README.md
├── .gitignore
├── week_1_llm_mechanics/
│   ├── kv_cache_profiler.py
│   ├── mini_llm_generator.py
│   ├── README.md
│   └── mini_project/
│       ├── main.py
│       └── README.md
├── week_2_enterprise_rag/
│   ├── cross_encoder_reranker.py
│   ├── document_parser.py
│   ├── eval_rag_metrics.py
│   ├── hybrid_retriever.py
│   ├── README.md
│   └── mini_project/
│       ├── main.py
│       └── README.md
├── week_3_agents_structured_outputs/
│   ├── pydantic_schemas.py
│   ├── sql_agent.py
│   ├── streaming_server.py
│   ├── README.md
│   └── mini_project/
│       ├── main.py
│       └── README.md
└── week_4_evals_finetuning_mlops/
    ├── dataset_generator.py
    ├── eval_pipeline.py
    ├── qlora_finetune.py
    ├── vllm_deploy.py
    ├── README.md
    └── mini_project/
        ├── main.py
        └── README.md
```

---

## 📄 Лицензия

Проект распространяется под лицензией [MIT](LICENSE).
