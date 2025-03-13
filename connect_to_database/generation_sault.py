import bcrypt

# Хэширование пароля с солью
password = "nid20242025"
salt = bcrypt.gensalt()  # Генерация случайной соли
print('SALT= ',salt)
hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)

print("Соль встроена в хэшированный пароль:", hashed_password)

# Проверка пароля
is_valid = bcrypt.checkpw(password.encode('utf-8'), hashed_password)
print("Пароль верный:", is_valid)