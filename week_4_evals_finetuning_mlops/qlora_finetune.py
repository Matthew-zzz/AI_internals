"""
Файл: qlora_finetune.py
Неделя 4: Дообучение модели (QLoRA Fine-Tuning).

ЧТО ЗДЕСЬ ДОЛЖНО БЫТЬ РЕАЛИЗОВАНО:
1. Загрузка базовой модели в 4-битном квантовании (BitsAndBytesConfig):
   - model_name = "Qwen/Qwen2.5-3B-Instruct" или "meta-llama/Llama-3.2-3B-Instruct"
   - load_in_4bit = True, bnb_4bit_quant_type = "nf4"

2. Настройка LoRA Конфигурации (LoraConfig от PEFT / Unsloth):
   - r = 16 (ранг матриц адаптера)
   - lora_alpha = 32
   - target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
   - lora_dropout = 0.05

3. Запуск SFTTrainer (из библиотеки TRL):
   - Передача train_dataset и параметров обучения (learning_rate = 2e-4, num_train_epochs = 3, per_device_train_batch_size = 4).
   - Сохранение обученных LoRA адаптеров в папку `./saved_lora_model`.
"""
