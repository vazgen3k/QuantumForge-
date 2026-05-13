from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


BASE_DIR = Path(__file__).parent

FAISS_INDEX_DIR = BASE_DIR / "faiss_index"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


TEST_QUERIES = [
    "Кто такой Вазген Калантарян?",
    "Что такое Грецкис и почему он важен?",
    "Кто такие аксакалы и как они связаны с сахарницей?"
]


def create_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def load_vectorstore(embeddings_model: HuggingFaceEmbeddings) -> FAISS:
    if not FAISS_INDEX_DIR.exists():
        raise FileNotFoundError(
            f"Папка с FAISS-индексом не найдена: {FAISS_INDEX_DIR}. "
            "Сначала запустите build_index.py"
        )

    return FAISS.load_local(
        folder_path=str(FAISS_INDEX_DIR),
        embeddings=embeddings_model,
        allow_dangerous_deserialization=True
    )


def print_search_results(query: str, results: list, top_k: int) -> None:
    print("=" * 100)
    print(f"Запрос: {query}")
    print(f"Найдено чанков: {len(results)} из top_k={top_k}")
    print("=" * 100)

    for index, doc in enumerate(results, start=1):
        metadata = doc.metadata

        print(f"\nРезультат #{index}")
        print(f"Файл: {metadata.get('source_file')}")
        print(f"Заголовок: {metadata.get('title')}")
        print(f"Chunk ID: {metadata.get('chunk_id')}")
        print(f"Word count: {metadata.get('word_count')}")
        print("-" * 100)
        print(doc.page_content[:1000])
        print("-" * 100)


def main() -> None:
    embeddings_model = create_embedding_model()
    vectorstore = load_vectorstore(embeddings_model)

    top_k = 3

    for query in TEST_QUERIES:
        results = vectorstore.similarity_search(
            query=query,
            k=top_k
        )

        print_search_results(query, results, top_k)


if __name__ == "__main__":
    main()