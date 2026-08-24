import pickle

# Открываем файл в режиме чтения байтов
with open("profile.pkl", "rb") as f:
    data = pickle.load(f)

# Выводим содержимое на экран
print(data)
