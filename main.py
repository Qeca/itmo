import asyncio
from quart import Quart, request, jsonify
import openai
import httpx
from bs4 import BeautifulSoup
import re
import logging
import urllib.parse
import xmltodict
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from cachetools import TTLCache, cached

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,  # Измените на WARNING или ERROR для продуктивной среды
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

app = Quart(__name__)

# Конфигурация с жестко закодированными API-ключами
OPENAI_API_KEY = FJ932uQtKGzh7G4eaizsiDh6SqKh8m89fJrisDLFzqRvp3IYYfbRXbkO2WTO35h4rkJzwiSBS40A'
YANDEX_API_KEY =
YANDEX_FOLDER_ID =
YANDEX_DOMAIN = 'ru'  # Например: ru, com, com.tr

EMBEDDING_MODEL = 'text-embedding-3-large'
CHUNK_SIZE = 512

# Инициализация асинхронного клиента OpenAI
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Инициализация LangChain компонентов
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=100,
    length_function=len
)

# Инициализация FAISS без загрузки с диска
vector_store = FAISS.from_texts(
    texts=["Initial document"],
    embedding=embeddings,
    metadatas=[{"url": "none"}]
)

url_cache = set()

# Кэширование поиска через Yandex
search_cache = TTLCache(maxsize=1000, ttl=3600)  # Кэширование на 1 час

@cached(cache=search_cache)
async def yandex_search(query, num=3, page=0):
    """Асинхронный поиск через Yandex Search API v1 с кэшированием"""
    try:
        base_url = f"https://yandex.{YANDEX_DOMAIN}/search/xml"
        # Параметры запроса
        params = {
            'folderid': YANDEX_FOLDER_ID,
            'apikey': YANDEX_API_KEY,
            'query': query,
            'lr': 213,  # Идентификатор региона поиска (Санкт-Петербург)
            'l10n': 'ru',  # Язык уведомлений
            'sortby': 'rlv',  # Сортировка по релевантности
            'filter': 'strict',  # Фильтр семейного контента
            'groupby': 'attr=d.mode=deep.groups-on-page=5.docs-in-group=3',
            'maxpassages': 3,  # Количество пассажей
            'page': page  # Номер страницы
        }

        # Экранирование специальных символов в параметрах
        encoded_params = {k: urllib.parse.quote_plus(str(v)) for k, v in params.items()}

        # Формирование полного URL с параметрами
        query_string = '&'.join([f"{k}={v}" for k, v in encoded_params.items()])
        request_url = f"{base_url}?{query_string}"

        logging.info(f"Отправка запроса к Yandex Search API: {request_url}")

        async with httpx.AsyncClient(timeout=0) as client:
            response = await client.get(request_url)
            response.raise_for_status()

            # Парсинг XML-ответа
            data = xmltodict.parse(response.text)
            links = []

            try:
                groups = data['yandexsearch']['response']['results']['grouping']['group']
                for group in groups:
                    docs = group.get('doc', [])
                    if isinstance(docs, dict):
                        docs = [docs]
                    for doc in docs:
                        url = doc.get('url')
                        if url and not url.endswith('.pdf'):
                            links.append(url)
                            if len(links) >= num:
                                break
                    if len(links) >= num:
                        break
            except KeyError as e:
                logging.error(f"Ошибка при парсинге XML-ответа: {e}")
                return []

            logging.info(f"Найдено {len(links)} ссылок через Yandex Search.")
            return links
    except Exception as e:
        logging.error(f"Search error: {e}")
        return []

def parse_question(query):
    """Парсинг вопроса и вариантов ответов"""
    lines = query.split('\n')
    question = lines[0].strip()
    options = {}
    pattern = re.compile(r'^(\d+)\.\s+(.+)$')

    for line in lines[1:]:
        match = pattern.match(line.strip())
        if match:
            options[int(match.group(1))] = match.group(2)

    return question, options

async def scrape_webpage(url):
    """Асинхронный парсинг веб-страниц"""
    if url in url_cache:
        logging.info(f"URL уже в кэше: {url}")
        return ""

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        async with httpx.AsyncClient(timeout=0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            for elem in soup(['script', 'style', 'nav', 'footer']):
                elem.decompose()

            text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])
            url_cache.add(url)
            logging.info(f"Текст успешно получен с {url}")
            return text
    except Exception as e:
        logging.error(f"Error parsing {url}: {e}")
        return ""

def add_to_vector_db(text_chunks, url):
    """Добавление данных в векторную БД"""
    if not text_chunks:
        return

    docs = [
        Document(
            page_content=chunk,
            metadata={"url": url}
        ) for chunk in text_chunks
    ]

    global vector_store
    vector_store.add_documents(docs)
    logging.info(f"Добавлено {len(text_chunks)} чанков из {url} в FAISS.")

def search_similar_chunks(query, k=6):
    """Поиск релевантных фрагментов (без оценки)"""
    return vector_store.similarity_search(query, k=k)

def search_similar_chunks_with_score(query, k=3):
    """Поиск релевантных фрагментов вместе с оценкой схожести"""
    # Предполагается, что vector_store имеет метод similarity_search_with_score
    return vector_store.similarity_search_with_score(query, k=k)

async def get_answer_from_llm(question, context=""):
    """Генерация ответа через OpenAI API"""
    try:
        logging.info("Отправка запроса к OpenAI API.")
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "Ты помощник по информации об ИТМО. Отвечай только на основе предоставленного контекста. Полноценно и структурированно отвечай на вопрос"},
                {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"}
            ]
        )
        logging.info("Ответ от OpenAI получен.")
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return None

async def scrape_and_add(url):
    """Функция для асинхронного парсинга веб-страницы и добавления в FAISS"""
    logging.info(f"Обрабатывается URL: {url}")
    text = await scrape_webpage(url)
    if text:
        chunks = text_splitter.split_text(text)
        add_to_vector_db(chunks, url)
    else:
        logging.info(f"Текст не получен для {url}")

@app.route('/api/request', methods=['POST'])
async def handle_request():
    data = await request.get_json()
    query = data.get('query', '')
    req_id = data.get('id', 0)

    question_text, options = parse_question(query)
    is_multiple_choice = len(options) > 0

    # Проверяем, есть ли в FAISS контекст с релевантностью >= 0.8
    similar_results = search_similar_chunks_with_score(question_text, k=1)
    if similar_results and similar_results[0][1] >= 0.8:
        logging.info("Найден релевантный контекст в FAISS с оценкой >= 0.8. Пропускаем внешнее гугление.")
        relevant_chunks = [similar_results[0][0]]
    else:
        # Если подходящего контекста нет, выполняем поиск через Yandex
        search_query = f"{question_text}"
        additional_sources = await yandex_search(search_query, num=3, page=0)
        logging.info(f"Найдено дополнительных ссылок: {len(additional_sources)}")

        sources = additional_sources
        logging.info(f"Общее количество источников: {len(sources)}")

        # Обработка контента (парсинг сайтов)
        tasks = []
        for url in sources:
            if url not in url_cache:
                tasks.append(scrape_and_add(url))
        # Запускаем все задачи параллельно
        await asyncio.gather(*tasks)
        # Выполняем семантический поиск в обновлённом FAISS
        relevant_chunks = search_similar_chunks(question_text)

    # Формируем контекст для LLM
    context = '\n'.join([
        f"Источник: {chunk.metadata['url']}\n{chunk.page_content[:1000]}"  # Ограничение длины
        for chunk in relevant_chunks
    ])

    # Генерация ответа
    llm_answer = await get_answer_from_llm(question_text, context)

    # Определение номера ответа для вариантов (если есть)
    answer_number = None
    if is_multiple_choice and llm_answer:
        for num, text in options.items():
            if text.lower() in llm_answer.lower():
                answer_number = num
                break

    # Формирование ответа
    response = {
        "id": req_id,
        "answer": answer_number if is_multiple_choice else None,
        "reasoning": f"{llm_answer} (Ответ с gpt-4o-mini)",
        "sources": list(set([chunk.metadata['url'] for chunk in relevant_chunks]))[:3]
    }

    return jsonify(response)

if __name__ == '__main__':
    # Запуск приложения с помощью Hypercorn
    # Пример команды для запуска через Hypercorn:
    # hypercorn main:app --bind 0.0.0.0:8081
    app.run(host='0.0.0.0', port=8081)