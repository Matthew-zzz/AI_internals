"""
ЗАДАНИЕ ОТ ИИ
Файл: mini_llm_generator.py
Неделя 1: Ручной инференс-цикл генерации токенов (БЕЗ использования model.generate()).

ЧТО ЗДЕСЬ ДОЛЖНО БЫТЬ РЕАЛИЗОВАНО:
1. Функция sample_next_token(logits, temperature, top_k, top_p, min_p):
   - Применяет масштабирование логитов по температуре (logits / temperature).
   - Фильтрует логиты по Min-P, Top-K и Top-P (Nucleus Sampling).
   - Применяет Softmax для получения распределения вероятностей.
   - Выполняет сэмплирование следующего токена через torch.multinomial (или argmax при temp=0).

2. Функция generate_text_custom(prompt, max_new_tokens, temperature):
   - Загружает модель и токенизатор (например, Qwen/Qwen2.5-0.5B).
   - Переводит модель в режим model.eval().
   - В цикле for step in range(max_new_tokens):
     * Делает forward pass: outputs = model(input_ids)
     * Извлекает логиты последнего токена
     * Сэмплирует ID следующего токена
     * Печатает декодированное слово в консоль (стриминг)
     * Добавляет полученный токен к текущему контексту (torch.cat)
     * Останавливает генерацию, если выпал токен EOS.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

def func_top_k(tensor: torch.Tensor, topk: int = 40):
   """ Фильтрует логиты(сырые веса), оставляя только топ k наибольших значений
   
   Args:
      tensor (torch.Tensor): Тензор над которым будет происходить операция, shape=(..., vocab_size)
      topk (int): Количество лучших кандидатов, которых нужно оставить

   Returns:
      torch.Tensor: Отфильтрованный тензор, в котором значения не прошедшие фильтрацию заменены на -∞

   Examples:
      >>> logits = torch.tensor([[2.3, 4.5, 1.2, -1.5]])
      >>> filtered_logits = func_top_k(logits, 3)
      >>> print(filtered_logits)
      [[2.3000, 4.5000, 1.2000, -inf]]
   """

   # Проверяем чтобы topk не был больше размера словаря и меньше 0
   if topk > tensor.shape[-1] or topk <= 0:
      return tensor

   # Используем встроенную функцию torch для отбора лучших кандидатов
   top_k_values, _ = torch.topk(tensor, topk, dim=-1)

   # (...) Из каждой предыдущей размерности берем последнее значение (-1) и добавляем единичную ось в конец (None), чтобы превратить вектор в столбец
   k_th_value = top_k_values[..., -1, None]

   # Формируем маску
   mask = tensor < k_th_value

   # Заменяем все значения True на -∞
   # Очень важно ! Можно заменять и на 0.0, но если функция softmax была использована до фильтрации, так как e^0 = 1.0,
   #  поэтому лучше использовать -∞ => e^-∞ = 0.0
   return tensor.masked_fill(mask, -torch.inf)

def func_min_p(tensor: torch.Tensor, min_p: float = 0.05):
   """ Фильтрует логиты(сырые веса), оставляя только кандидатов значения которых не ниже доли от значений лучшего кандидата
      
   Args:
      tensor (torch.Tensor): Тензор над которым будет происходить операция, shape=(..., vocab_size)
      min_p (float): Минимальная допустимая доля от вероятности 
         лидера (в диапазоне [0.0, 1.0]). По умолчанию 0.05

   Returns:
      torch.Tensor: Отфильтрованный тензор, в котором значения не прошедшие фильтрацию заменены на -∞

   Examples:
      >>> logits = torch.tensor([[2.3, 4.5, 1.1, -1.5, 0.02]])
      >>> filtered_logits = func_min_p(logits, 0.05)
      >>> print(filtered_logits)
      [[2.3000, 4.5000, -inf, -inf, -inf]]
   """
   
   if min_p < 0.0:
      return tensor

   # Применяем softmax, для получения вероятностей от 0.0 до 1.0, чтобы при формировании threshold не получилось минусовое значение (-1.5*0.05=-0.075)
   probs = F.softmax(tensor, dim=-1)

   # Выбираем максимальную вероятность для последующей фильтрации
   max_prob = probs.max(dim=-1, keepdim=True).values

   # Устанавливаем порог
   threshold = max_prob * min_p

   mask = probs < threshold

   return tensor.masked_fill(mask, -torch.inf)
   
def func_top_p(tensor: torch.Tensor, top_p:float=0.95):
   """ Фильтрует логиты(сырые веса), оставляя только кандидатов кумулятивная сумма которых не превышает top_p
         
   Args:
      tensor (torch.Tensor): Тензор над которым будет происходить операция, shape=(..., vocab_size)
      top_p (float): Значение порога после которого сумма значений элементов не учитывается

   Returns:
      torch.Tensor: Отфильтрованный тензор, в котором значения не прошедшие фильтрацию заменены на -∞

   Examples:
      >>> logits = torch.tensor([[2.3, 4.5, 1.1, -1.5, 0.02]])
      >>> filtered_logits = func_min_p(logits, 0.05)
      >>> print(filtered_logits)
      [[2.3000, 4.5000, -inf, -inf, -inf]]
   """
   
   if not (0.0 <= top_p <= 1.0):
      return tensor

   # Обязательно сортируем по убыванию, для правильной работы функции(Считаем сумму самых лучших первых кандидатов) 
   sorted_logits, sorted_indices = tensor.sort(descending=True)

   # Получаем вероятности
   sorted_probs = torch.softmax(sorted_logits, dim=-1)
   
   # Считаем кумулятивную сумму элементов тензора
   cumulative_probs = sorted_probs.cumsum(dim=-1)

   masked_probs = cumulative_probs > top_p
   # Оставляем лучшего кандидата. Может произойти так что [0] = 0.98, а top_p=0.95, из за этого маска отсеет его сразу же и забракует все элементы словаря
   
   masked_probs[..., 1:] = masked_probs[..., :-1].clone()

   # Делаем его автоматически успешно отфильтрованным
   masked_probs[..., 0] = False

   # После сортировки индексы элементов перемешались и поэтому их надо вернуть на исходные места, чтобы маска не забраковала нужные элементы.
   return_tensor_indices = masked_probs.scatter(dim=-1, index=sorted_indices, src=masked_probs)

   return tensor.masked_fill(return_tensor_indices, -torch.inf)

def sample_text_token(logits: torch.Tensor, temperature: float, top_k: int =50, top_p: float =0.9, min_p: float =0.05):
   """ Фильтрует логиты(сырые веса), применяя масштабируемость к температуре и проходя через фильтры min_p > top_k > top_p
            
   Args:
      logits (torch.Tensor): Тензор над которым будет происходить операция, shape=(..., vocab_size)
      temperature (float): Параметр температуры для контроля случайности
      top_k (int, optional): Максимальное количество лучших кандидатов. По умолчанию 50
      top_p (float, optional): Порог кумулятивной вероятности. По умолчанию 0.9
      min_p (float, optional): Минимальная допустимая доля от вероятности кандидата
         По умолчанию 0.05.

   Returns:
      torch.Tensor: Отфильтрованный тензор, с ID выбранного токена

   Examples:
      >>> logits = torch.tensor([[2.3, 4.5, 1.1, -1.5, 0.02]])
      >>> token_id = sample_text_token(logits, temperature=0.7, top_k=3, top_p=0.9, min_p=0.05)
      >>> print(token_id.shape)
      >>> print("Индекс элемента: ", token_id.item())
      torch.Size([1, 1])
      Индекс элемента: 334
   """

   # Масштабируем под температуру все логиты
   scaled = logits / temperature
    
   filtered_logits = func_min_p(scaled, min_p)

   filtered_logits = func_top_k(filtered_logits, top_k)

   filtered_logits = func_top_p(filtered_logits, top_p)

   probs = filtered_logits.softmax(dim=-1)

   # Выбираем случайный элемент, учитывая вероятности элементов
   next_token = torch.multinomial(probs, num_samples=1)
   return next_token
   
def generate_text_custom(prompt, max_new_tokens, temperature):
   """Генерирует текст авторегрессионным способом без использования model.generate()

   Токенизирует входной промпт, в цикле выполняет forward pass модели, 
   сэмплирует новые токены и добавляет их в контекст до достижения лимита или EOS-токена

   Args:
      prompt (str): Входной текстовый запрос для модели
      max_new_tokens (int, optional): Максимальное число генерируемых токенов
      temperature (float, optional): Температура сэмплирования

   Returns:
      str: Полный сгенерированный ответ модели

   Examples:
      >>> response = generate_text_custom("Hello! How are you?", max_new_tokens=20)
      >>> isinstance(response, str)
      True
    """
   # При наличии cuda переносим вычисления на нее
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   model_name = "Qwen/Qwen2.5-0.5B"
   model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
   tokenizer = AutoTokenizer.from_pretrained(model_name)

   # Говорим модели что мы не учимся, выключает DropOut, меняет батч нормализацию
   model.eval()

   llm_answer = ""
   input_ids = tokenizer((prompt + llm_answer), return_tensors="pt").to(device).input_ids

   # Отлючаем сохранение градиентов для ускорения работы и экономии памяти
   with torch.no_grad():
      for _ in range(max_new_tokens):
         print(input_ids.shape)

         output = model(
            input_ids
         )

         next_token = sample_text_token(output.logits[:, -1, :], temperature)

         # Проверяем, чтобы токен не был токеном конца ответа модели
         if next_token.item() == tokenizer.eos_token_id:
            break

         word = tokenizer.decode(next_token.item())
         llm_answer += word
         
         print(f"Сгенерированный токен: {word}")

         if next_token.ndim == 1:
            next_token = next_token.unsqueeze(0)

         # Соединяем токен слова с токенами общего ответа
         input_ids = torch.cat([input_ids, next_token], dim=-1)

   print(f"Ответ модели: {llm_answer}")

generate_text_custom("Hello! How are you ?", 30, 0.6)