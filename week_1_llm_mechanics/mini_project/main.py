"""
ПРОМЫШЛЕННЫЙ МИНИ-ПРОЕКТ НЕДЕЛИ 1: Real-Time Tokenizer & Custom KV-Cache Inference Engine.

ОТРАСЛЕВАЯ ЗАДАЧА:
Создать изолированный инференс-движок на PyTorch для локальных моделей (Qwen 2.5 / Llama 3.2),
обеспечивающий детерминированное сэмплирование, стриминг и точный профайлинг расхода памяти KV-Cache.

АРХИТЕКТУРА И ШАГИ РЕАЛИЗАЦИИ:

1. КЛАСС `CustomInferenceEngine`:
   - Метод `__init__(model_name, device, torch_dtype)`:
     * Загружает модель и токенизатор в режиме eval().
     * Переводит веса в float16 или bfloat16 для оптимизации VRAM.
   
   - Метод `sample_logits(logits, temperature, top_p, min_p, repetition_penalty, generated_tokens)`:
     * Применяет Repetition Penalty к логитам ранее сгенерированных токенов.
     * Применяет масштабирование по температуре (logits / temperature).
     * Фильтрует логиты по Min-P (отсечение токенов с p < min_p * max_p).
     * Применяет Top-P (Nucleus) сэмплирование.
     * Нормализует вероятности через Softmax и сэмплирует ID следующего токена через torch.multinomial.

   - Метод `generate_stream(prompt, max_tokens, sampling_params)`:
     * Выполняет первый forward pass и инициализирует past_key_values (KV-Cache).
     * Замеряет время до первого токена (TTFT - Time To First Token).
     * В цикле пошагово передает только ПОСЛЕДНИЙ сгенерированный токен и обновленный past_key_values.
     * Выполняет декодирование и потоковый вывод токена в консоль (Streaming).

2. КЛАСС `KVCacheProfiler`:
   - Метод `profile_memory_and_speed(engine, prompt_lengths=[512, 2048, 8192])`:
     * Рассчитывает теоретический размер KV-Cache по формуле:
       Bytes = 2 * 2 * num_layers * num_heads * head_dim * seq_len * sizeof(dtype)
     * Измеряет реальное потребление памяти GPU (через torch.cuda.max_memory_allocated) и время работы.
     * Рассчитывает реальный TPS (Tokens Per Second) в двух режимах: With KV-Cache vs Without KV-Cache.
     * Формирует и выводит сравнительную сводную таблицу в консоль.

3. ТОЧКА ВХОДА (main):
   - Инициализирует движок, выполняет пробный генерационный стрим и запускает полный цикл профайлинга.
"""
