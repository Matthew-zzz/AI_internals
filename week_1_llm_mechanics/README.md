# 📅 Неделя 1: Фундамент ИИ и механика Transformer (Under the Hood)

## 🎯 Цели недели
Перестать видеть в LLM "черный ящик". Понять, как цифры превращаются в смысл и текст через авторегрессионный инференс, сэмплирование и KV-Cache.

---

## 📋 Чек-лист темы
- [ ] Понять авторегрессию: P(w_n | w_1, ..., w_n-1)
- [ ] Разобрать устройство Transformer Decoder (Q, K, V матрицы, Self-Attention)
- [ ] Опробовать стратегии сэмплинга: Temperature, Top-K, Top-P (Nucleus), Min-P
- [ ] Замерить потребление VRAM/RAM и TPS при поддержке KV-Cache

---

## 📁 Файлы проекта

1. **`mini_llm_generator.py`**
   - Написание собственного инференс-цикла на `PyTorch` + `transformers` без утилиты `generate()`.
   - Задачи: `Forward Pass -> Logits -> Temperature Scaling -> Top-P/Top-K Filtering -> Softmax -> Categorical Sampling`.

2. **`kv_cache_profiler.py`**
   - Ручная реализация KV-Cache и профайлинг производительности.
   - Задачи: Сравнение скорости (Tokens Per Second) и расхода памяти GPU/RAM при контексте 512, 2048 и 8192 токенов.

---

## 📚 Рекомендуемые материалы
- **Andrej Karpathy:** YouTube *"Neural Networks: Zero to Hero"* (особенно "Let's build GPT from scratch").
- **Hugging Face Docs:** `past_key_values` в Transformers.
