import paramiko

# Данные для подключения
hostname = '79.134.209.114'  # IP или домен сервера
port = 22555                # Порт SSH (по умолчанию 22)
username = 'leovaldaev'          # Логин
password = 'nid20242025'      # Пароль

try:
    # Создание SSH-клиента
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Доверие к новому серверу

    # Подключение к серверу
    ssh.connect(hostname, port, username, password)
    print("✅ Подключение успешно!")

    # Выполнение команды на сервере
    # scp - P 22555 06.11.2024.zip leovaldaev@79.134.209.114:/mnt/sda/


    # Закрытие соединения
    ssh.close()

except Exception as e:
    print(f"❌ Ошибка подключения: {e}")