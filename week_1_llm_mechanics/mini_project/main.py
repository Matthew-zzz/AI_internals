"""
ПРОМЫШЛЕННЫЙ МИНИ-ПРОЕКТ НЕДЕЛИ 1: Real-Time Tokenizer & Custom KV-Cache Inference Engine.

ОТРАСЛЕВАЯ ЗАДАЧА:
Создать изолированный инференс-движок на PyTorch для локальных моделей (Qwen 2.5 / Llama 3.2),
обеспечивающий детерминированное сэмплирование, стриминг и точный профайлинг расхода памяти KV-Cache.

АРХИТЕКТУРА И ШАГИ РЕАЛИЗАЦИИ:

1. КЛАСС `CustomInferenceEngine`:
   - Метод `__init__(model_name, device, torch_dtype)`:
     * Загружает модель(в режиме eval()) и токенизатор.
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
   - Метод `profile_memory_and_speed(engine, prompt_lengths)`:
     * Рассчитывает теоретический размер KV-Cache по формуле:
       Bytes = 2 * num_layers * num_heads * head_dim * seq_len * sizeof(dtype)
     * Измеряет реальное потребление памяти GPU (через torch.cuda.max_memory_allocated) и время работы.
     * Рассчитывает реальный TPS (Tokens Per Second) в двух режимах: With KV-Cache vs Without KV-Cache.
     * Формирует и выводит сравнительную сводную таблицу в консоль.

3. ТОЧКА ВХОДА (main):
   - Инициализирует движок, выполняет пробный генерационный стрим и запускает полный цикл профайлинга.
"""

from dataclasses import dataclass
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    SentencePieceBackend,
    TokenizersBackend,
)


@dataclass
class SamplingParams:
    # Параметры для сэмплирования

    top_p: float
    min_p: float
    repetition_penalty: float
    temperature: float
    use_cache: bool


@dataclass
class GenerateResult:
    # Результаты генерации

    time_to_first_token: float
    tokens_per_second: float
    kv_cache_prefill: float
    kv_cache_total: float
    model_answer: str


class CustomInferenceEngine:
    def __init__(
        self,
        model_name: str,
        device,
        torch_dtype=torch.float16,
        tokenizer_name: str = None,
    ):
        self.device: torch.device = device

        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
            .to(self.device)
            .eval()
        )

        self.tokenizer: TokenizersBackend | SentencePieceBackend = (
            AutoTokenizer.from_pretrained(
                tokenizer_name if tokenizer_name else model_name
            )
        )

    def top_p(self, logits: torch.Tensor, top_p: float):
        # Сортируем в порядке убывания
        sorted_values, sorted_indices = torch.sort(logits, descending=True)

        # Переводим в вероятности (0.0 - 1.0)
        probs = torch.softmax(sorted_values, dim=-1)

        # Кумулятивная сумма
        cumulative_sum = probs.cumsum(dim=-1)

        # Формируем маску только из значений, которые меньше кумулятивной суммы
        mask = cumulative_sum > top_p

        # Всегда берем первый элемент. Зачем ? Ответ в файле mini_llm_generator.py
        mask[..., 0] = False

        # Создаем пустой тензор с размерностью маски
        new_mask = torch.zeros_like(mask)

        # Заполняем новый тензор значениями из mask с индексами sorted_indices
        new_mask = torch.scatter(new_mask, dim=-1, index=sorted_indices, src=mask)

        # Возвращаем отфильтрованный тензор
        return torch.masked_fill(logits, new_mask, -torch.inf)

    def min_p(self, logits: torch.Tensor, min_p: float):
        # Формируем вероятности. Зачем ? Ответ в файле mini_llm_generator.py
        probs = torch.softmax(logits, dim=-1)
        # Ищем самый вероятный токен
        max_prob = probs.max(dim=-1, keepdim=True).values

        # Формируем границу
        threshold = min_p * max_prob

        mask = probs < threshold

        return torch.masked_fill(logits, mask, -torch.inf)

    def sample_logits(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        min_p: float,
        repetition_penalty: float,
        generated_tokens: torch.Tensor = None,
    ):
        if generated_tokens is not None:
            # Если размерность несоответствует размерности для работы с логитами, то добавляем новую ось в начало
            if generated_tokens.ndim == 1:
                generated_tokens = generated_tokens.unsqueeze(0)

            # Берем значения, указанных индексов
            generated_logits = torch.gather(logits, dim=-1, index=generated_tokens)

            # Штрафуем логиты
            penalty_logits = torch.where(
                generated_logits > 0,
                generated_logits / repetition_penalty,
                generated_logits * repetition_penalty,
            )

            # Возвращаем все индексы с нвоыми значениями на место
            logits = torch.scatter(
                input=logits, dim=-1, index=generated_tokens, src=penalty_logits
            )

        # Масштабирование под температуру
        scaled = logits / temperature

        filtered_scaled = self.min_p(scaled, min_p)

        filtered_scaled = self.top_p(filtered_scaled, top_p)

        # Формируем итоговые вероятности
        probs = torch.softmax(filtered_scaled, dim=-1)

        # Берем токен с помощью детерменированного сэмплирования
        next_token = torch.multinomial(probs, 1)

        return next_token

    def generate_stream(self, prompt, max_tokens, sampling_params: SamplingParams):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        with torch.no_grad():

            input_ids_prompt = (
                self.tokenizer(prompt, return_tensors="pt").to(self.device).input_ids
            )

            time_to_first_token = None
            kv_cache_prefill = None
            kv_cache_total = None

            # Используем KV-кэш
            if sampling_params.use_cache:
                cache = DynamicCache(config=self.model.config)

                # KV-кэш промпта
                kv_cache_prefill = KVCacheProfiler.profile_memory_and_speed(
                    self, (input_ids_prompt.shape[-1])
                )

                # KV-кэш промпта + сгенерированные токены
                kv_cache_total = KVCacheProfiler.profile_memory_and_speed(
                    self, (input_ids_prompt.shape[-1] + max_tokens)
                )

                start_ttft = time.perf_counter()

                output = self.model(
                    input_ids_prompt, use_cache=True, past_key_values=cache
                )

                next_token = self.sample_logits(
                    logits=output.logits[:, -1, :],
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                    min_p=sampling_params.min_p,
                    repetition_penalty=sampling_params.repetition_penalty,
                )

                generated_tokens = next_token

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                time_to_first_token = time.perf_counter() - start_ttft

                start_tps = time.perf_counter()

                for _ in range(max_tokens - 1):
                    output = self.model(
                        next_token, use_cache=True, past_key_values=cache
                    )

                    next_token = self.sample_logits(
                        logits=output.logits[:, -1, :],
                        temperature=sampling_params.temperature,
                        top_p=sampling_params.top_p,
                        min_p=sampling_params.min_p,
                        repetition_penalty=sampling_params.repetition_penalty,
                        generated_tokens=generated_tokens,
                    )
                    generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

                    word = self.tokenizer.decode(next_token.item())
                    print(f"[{_}]Слово: {word}")

                finish_tps = time.perf_counter() - start_tps
                model_answer = self.tokenizer.decode(generated_tokens)
            # Не используем KV-кэш
            else:
                output = self.model(input_ids_prompt, use_cache=False)

                start_tps = time.perf_counter()

                next_token = self.sample_logits(
                    output.logits[:, -1, :],
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                    min_p=sampling_params.min_p,
                    repetition_penalty=sampling_params.repetition_penalty,
                )

                generated_tokens = next_token

                for _ in range(max_tokens):

                    output = self.model(
                        torch.cat([input_ids_prompt, generated_tokens], dim=-1),
                        use_cache=False,
                    )

                    next_token = self.sample_logits(
                        output.logits[:, -1, :],
                        temperature=sampling_params.temperature,
                        top_p=sampling_params.top_p,
                        min_p=sampling_params.min_p,
                        repetition_penalty=sampling_params.repetition_penalty,
                        generated_tokens=generated_tokens,
                    )

                    generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

                finish_tps = time.perf_counter() - start_tps

                model_answer = self.tokenizer.decode(generated_tokens)

        return GenerateResult(
            time_to_first_token=(time_to_first_token if time_to_first_token else 0),
            tokens_per_second=max_tokens / finish_tps,
            kv_cache_prefill=kv_cache_prefill,
            kv_cache_total=kv_cache_total,
            model_answer=model_answer,
        )


class KVCacheProfiler:
    @staticmethod
    def profile_memory_and_speed(engine: CustomInferenceEngine, prompt_lengths: int):
        """Фильтрует логиты(сырые веса), оставляя только кандидатов значения которых не ниже доли от значений лучшего кандидата

        Args:
            engine (CustomInferenceEngine): Класс с нашими внутренними функциями, отуда потом возьмем конфиг модели
            prompt_lengths (int): Длина промта

        Returns:
            int: Размер KV кэша в байтах

        Examples:
            >>> kv_cache_size = profile_memory_and_speed(engine, 1024)
            >>> print(f"{kv_cache_size} байт")
            4325234 байт
        """

        # Конфиг выбранной LLM модели, содержит в себе основные характеристики модели
        model_config = engine.model.config

        # Формула расчета Key-Value кжэша
        kv_cache_size = (
            # 2 - пара key, value
            2
            # Количество слоев трансформера
            * model_config.num_hidden_layers
            # Количество голов key,value для каждого слоя
            * model_config.num_key_value_heads
            # Размер вектора каждой головы внимания => общий размер вектора / количество голов внимания
            * (model_config.hidden_size / model_config.num_attention_heads)
            # Длина промпта
            * prompt_lengths
            # 2 - размер значения dtype в байтах. Если в битах, то поделить на 8.
            # В данном случае используется float16 - 16 бит - 2 байта
            * 2
        )
        return kv_cache_size


def main(samping_params: SamplingParams):
    max_tokens = 30
    prompt = "Hello! How are you ?"
    model_name = "Qwen/Qwen2.5-0.5B"

    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    custom_engine = CustomInferenceEngine(
        model_name, torch.device("cuda"), torch.bfloat16
    )

    generate_result: GenerateResult = custom_engine.generate_stream(
        prompt, max_tokens, samping_params
    )

    finish_time = time.perf_counter() - start_time

    print(f"Входящий промпт: {prompt}")
    print(f"Ответ модели({model_name}): {generate_result.model_answer}")

    print("=" * 50)
    print("ВРЕМЯ")
    print("=" * 50)
    if generate_result.time_to_first_token:
        print(f"Время до первого токена: {generate_result.time_to_first_token:.2f}")
    print(f"TPS: {generate_result.tokens_per_second:.0f} токенов/сек")
    print(f"Времени потрачено: {finish_time:.2f} сек")

    max_vram_peak = torch.cuda.max_memory_allocated() / 1024**3
    max_vram_reserved = torch.cuda.max_memory_reserved() / 1024**3

    print("=" * 50)
    print("ПАМЯТЬ")
    print("=" * 50)
    print(f"Максимально использовалось памяти: {max_vram_peak:.2f} GB")
    print(f"Максимально выделилось памяти: {max_vram_reserved:.2f} GB")
    if generate_result.kv_cache_total and generate_result.kv_cache_prefill:
        print(
            f"KV кэш - prefill: {generate_result.kv_cache_prefill} байт | {(generate_result.kv_cache_prefill / 1024**2):.2f} мб"
        )
        print(
            f"KV кэш - общий: {generate_result.kv_cache_total} байт | {(generate_result.kv_cache_total / 1024**2):.2f} мб"
        )


print("=" * 50)
print(f"С ИСПОЛЬЗОВАНИЕМ KV-КЭША")
print("=" * 50)

samping_params = SamplingParams(
    top_p=0.6, min_p=0.05, repetition_penalty=1.5, temperature=0.8, use_cache=True
)

main(samping_params)

print("=" * 50)
print(f"БЕЗ ИСПОЛЬЗОВАНИЯ KV-КЭША")
print("=" * 50)

samping_params = SamplingParams(
    top_p=0.6, min_p=0.05, repetition_penalty=1.5, temperature=0.8, use_cache=False
)

main(samping_params)
