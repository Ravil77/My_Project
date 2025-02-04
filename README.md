
### Описание проекта

Приложение разработано для анализа изображений с использованием Hugging Face. Приложение предоставляет пользователю возможность загружать изображения, вводить текстовые запросы и получать ответы от модели в удобном интерфейсе. За основу принят код с использованием модели **Paligemma** через API Hugging Face. Интерфейс реализован с помощью **Streamlit**.

---

### Установка и запуск

Для локального использования выполните следующие шаги:

#### 1. Клонирование репозитория

Клонируйте данный репозиторий на локальный компьютер:

```bash
git clone https://github.com/Ravil77/My_Project.git
```

Перейдите в директорию проекта:

```bash
cd My_Project
```

#### 2. Установка зависимостей для приложения

Убедитесь, что у вас установлен Python не ниже версии 3.7. Установите зависимости из файла `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### 3. Запуск приложения

Для запуска Streamlit-приложения выполните команду:

```bash
streamlit run streamlit_app.py
```


### Основные функции

- Загрузка изображений через файловый проводник.
- Ввод текстового запроса для анализа изображения.
- Вывод результатов анализа 
- Удобный интерфейс для взаимодействия с моделью.

---

### Структура проекта

- **`streamlit_app.py`**: Основной файл приложения Streamlit.
- **`requirements.txt`**: Список зависимостей, необходимых для работы приложения.
- **Дополнительные файлы**: Код и вспомогательные модули для обеспечения работы приложения.

---

### Требования к приложению

- Python > 3.10
- Установленные зависимости из `requirements.txt`

