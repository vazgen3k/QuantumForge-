from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent

FAISS_INDEX_DIR = PROJECT_DIR / "task3" / "faiss_index"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

LLM_MODEL_NAME = "qwen2.5:3b"


def create_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def load_vectorstore(embeddings_model: HuggingFaceEmbeddings) -> FAISS:

    return FAISS.load_local(
        folder_path=str(FAISS_INDEX_DIR),
        embeddings=embeddings_model,
        allow_dangerous_deserialization=True
    )


def search_relevant_chunks(
    vectorstore: FAISS,
    query: str,
    k: int = 4
):
    return vectorstore.similarity_search(query, k=k)


def format_context(chunks) -> str:
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata

        source_file = metadata.get("source_file", "unknown")
        chunk_id = metadata.get("chunk_id", "unknown")

        context_parts.append(
            f"[Источник {i}: {source_file}, chunk_id={chunk_id}]\n"
            f"{chunk.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)


def build_prompt(query: str, context: str) -> str:
    """
    Формирует prompt для LLM на основе вопроса пользователя,
    найденных чанков и few-shot примеров.
    """
    prompt = f"""
Ты — внутренний справочный RAG-ассистент QuantumForge Software.

Отвечай только на основе предоставленного контекста.

Не используй внешние знания.

Если в контексте нет информации для ответа, честно ответь: "Я не знаю".

Всегда отвечай в такой структуре:

1. Шаги:

- Кратко укажи, какие найденные фрагменты относятся к вопросу.
- Кратко объясни, какая информация из них важна.
- Не добавляй факты, которых нет в контексте.

2. Ответ:

- Дай итоговый ответ на вопрос пользователя.
- Если возможно, укажи источник в формате: [source_file, chunk_id].

Примеры правильного поведения:

Q: Что такое Грецкис?
A: 1. Шаги:
- В найденном фрагменте `Arrakis.txt`, chunk_id=0, сказано, что Грецкис — это мир, также известный как Айсберг.
- Там же указано, что Грецкис является единственным источником конфета, или карамеланжа.
- Следовательно, важность Грецкиса связана с контролем над конфетом и межзвёздными перелётами.
2. Ответ:
Грецкис — это мир, также известный как Айсберг. Он важен тем, что является единственным источником конфета, или карамеланжа, необходимого для межзвёздных перелётов. [Arrakis.txt, chunk_id=0]

Q: Сколько спутников у Грецкис?
A: 
1. Шаги
- В найденном фрагменте `Arrakis.txt`, chunk_id=0, сказано, что Грецкис имеет два спутника.
- Следовательно,  Грецкиса имеет два спутника.
2. Ответ:
Грецкис имеет два спутника

Теперь ответь на вопрос пользователя.

Контекст:
{context}

Q: {query}
A:
"""
    return prompt.strip()

def create_llm() -> ChatOllama:

    """
    LLM через Ollama.
    """
    return ChatOllama(
        model=LLM_MODEL_NAME,
        temperature=0.2
    )

def generate_answer(llm: ChatOllama, prompt: str) -> str:
    response = llm.invoke(prompt)
    return response.content


def main() -> None:
    print("Загрузка embedding-модели...")
    embeddings_model = create_embedding_model()
    print("Загрузка FAISS-индекса...")
    vectorstore = load_vectorstore(embeddings_model)
    print("Загрузка локальной LLM...")
    llm = create_llm()
    print("\nбот запущен.")
    print("Введите вопрос или напишите 'exit', 'quit', 'выход' для завершения.")
    while True:
        query = input("\nВы: ").strip()
        if query.lower() in {"exit", "quit", "выход"}:
            print("Бот: Завершаю работу.")
            break
        if not query:
            print("Бот: Введите вопрос.")
            continue
        chunks = search_relevant_chunks(vectorstore, query, k=4)
        context = format_context(chunks)
        prompt = build_prompt(query, context)
        print("\nБот думает...")
        answer = generate_answer(llm, prompt)
        print("\nБот:")
        print(answer)



if __name__ == "__main__":
    main()