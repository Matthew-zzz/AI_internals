"""
Файл: kv_cache_profiler.py
Неделя 1: Профайлинг производительности и ручная поддержка KV-Cache.

ЧТО ЗДЕСЬ ДОЛЖНО БЫТЬ РЕАЛИЗОВАНО:
  1. Функция profile_kv_cache(model_name, context_lengths):
    - Загружает авторегрессионную модель и переводит на GPU/CPU.
    - Для каждого значения длины контекста (512, 2048, 8192 токенов):
      * Создает длинный тестовый промпт.

  2. Сравнение 2 подходов генерации 30 токенов:
    - ВАРИАНТ А (Без KV-Cache):
      * При вызове model(curr_ids, use_cache=False) передается весь накопившийся контекст.
    - ВАРИАНТ Б (С KV-Cache):
      * На первом шаге сохраняется past_key_values из вывода модели.
      * На последующих шагах в модель передается только 1 ПОСЛЕДНИЙ токен и скопленный past_key_values.
      * Замеряется время и TPS.

  3. Вывод результатов:
    - Сравнение времени выполнения (Без KV-Cache vs С KV-Cache).
    - Расчёт коэффициента ускорения (Speedup Factor).
"""

import time
from loguru import logger
import torch
from transformers import AutoModelForCausalLM, DynamicCache


def log_benchmark_result(label: str, result: dict):
    logger.info(
        f"{label} | "
        f"Общее время={result['elapsed']:.2f}s | "
        f"TPS={result['tps']:.2f} | "
        f"Peak VRAM={result['peak_memory_gb']:.3f}GB | "
        f"Prefill={result['prefill_elapsed']:.2f}s | "
        f"Decode={result['decode_elapsed']:.2f}s"
    )


def kv_cache_structure(kv_cache: DynamicCache):
    layer_o_kv = kv_cache.layers[0]
    keys_tensor = layer_o_kv.keys
    values_tensor = layer_o_kv.values

    logger.info("=== РАЗМЕРНОСТИ ТЕНЗОРОВ В КЭШЕ ===")
    # форма - (batch_size, num_kv_heads, sequence_length, head_dim)
    logger.info("Форма тензора Key (Ключи):  ", keys_tensor.shape)
    logger.info("Форма тензора Value (Значения):", values_tensor.shape)

    logger.info("\n=== КАК ВЫГЛЯДЯТ САМИ ЧИСЛА (кусочек матриц) ===")

    logger.info(keys_tensor[0, 0, 0, :10])


def profile_kv_cache(
    model,
    context_length: list[int] = [512, 1024, 2048, 8192],
    generate_tokens_count: int = 30,
):

    model_name = model.name_or_path
    device = model.device

    max_pos = getattr(
        model.config,
        "max_position_embeddings",
        getattr(model.config, "n_positions", 2048),
    )

    for length in context_length:
        if length > max_pos:
            logger.info(
                f"Пропускаем контекстное окно из {length} токенов: Превышение max_position_embeddings ({max_pos}) для модели {model_name}\n"
            )
            return

        logger.info(
            f"Контекстное окно из {length - generate_tokens_count if length == max_pos else length} токенов"
        )

        input_ids = torch.randint(
            low=0,
            high=model.config.vocab_size,
            size=(1, (length - generate_tokens_count) if length == max_pos else length),
        ).to(device)

        with_kv_cache = check_time_kv_cache(
            model,
            input_ids,
            use_cache=True,
            generate_tokens_count=generate_tokens_count,
        )
        log_benchmark_result("С KV кэшем", with_kv_cache)

        without_kv_cache = check_time_kv_cache(
            model,
            input_ids,
            use_cache=False,
            generate_tokens_count=generate_tokens_count,
        )
        log_benchmark_result("Без KV кэша", without_kv_cache)

        speedup_factor = without_kv_cache["elapsed"] / with_kv_cache["elapsed"]
        logger.success(f"Коэффициент ускорения: {speedup_factor:.2f}x")


def check_time_kv_cache(
    model,
    input_ids: torch.Tensor,
    use_cache: bool = True,
    generate_tokens_count: int = 30,
):
    device = input_ids.device

    copy_input_ids = input_ids.clone()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        if use_cache:
            cache = DynamicCache(config=model.config)

            prefill_start = time.perf_counter()

            output = model(input_ids, past_key_values=cache, use_cache=True)

            if device.type == "cuda":
                torch.cuda.synchronize()

            prefill_elapsed = time.perf_counter() - prefill_start

            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            for _ in range(generate_tokens_count - 1):
                output = model(next_token, past_key_values=cache, use_cache=True)

                next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        else:
            for _ in range(generate_tokens_count):
                output = model(copy_input_ids, use_cache=False)

                next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)

                copy_input_ids = torch.cat([copy_input_ids, next_token], dim=1)

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    else:
        peak_memory = 0

    elapsed = time.perf_counter() - start

    tps = generate_tokens_count / elapsed

    return {
        "elapsed": elapsed,
        "tps": tps,
        "peak_memory_gb": peak_memory,
        "prefill_elapsed": prefill_elapsed if use_cache else 0,
        "decode_elapsed": elapsed - prefill_elapsed if use_cache else 0,
    }


def checking_kv_cache(model_name: str):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    for index in range(5):
        logger.info(f"Start of {index} iteration.")
        profile_kv_cache(model)


checking_kv_cache("Qwen/Qwen2.5-0.5B")
