from pathlib import Path
import json
import time

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


BASE_DIR = Path(__file__).parent

KNOWLEDGE_BASE_DIR = BASE_DIR.parent / "task2" / "knowledge_base"
OUTPUT_DIR = BASE_DIR / "output"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"

CHUNKS_PATH = OUTPUT_DIR / "chunks.jsonl"
EMBEDDING_INFO_PATH = OUTPUT_DIR / "embedding_info.json"
BUILD_INFO_PATH = OUTPUT_DIR / "build_info.json"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def count_words(text: str) -> int:
    return len(text.split())


def load_and_split_documents() -> list[Document]:
    """
    Загружает .txt файлы из knowledge_base и разбивает их на чанки.
    Каждый чанк получает метаданные: файл, путь, заголовок, chunk_id, word_count.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
        length_function=count_words,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    documents: list[Document] = []

    txt_files = sorted(KNOWLEDGE_BASE_DIR.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(
            f"В папке {KNOWLEDGE_BASE_DIR} не найдены .txt файлы"
        )

    for file_path in txt_files:
        text = file_path.read_text(encoding="utf-8")

        source_document = Document(
            page_content=text,
            metadata={
                "source_file": file_path.name,
                "source_path": str(file_path),
                "title": file_path.stem
            }
        )

        chunks = text_splitter.split_documents([source_document])

        for chunk_id, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["word_count"] = count_words(chunk.page_content)
            documents.append(chunk)

    return documents


def create_embedding_model() -> HuggingFaceEmbeddings:
    """
    Создаёт локальную embedding-модель.
    """
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def save_chunks_debug_file(documents: list[Document]) -> None:
    """
    Сохраняет чанки без эмбеддингов в jsonl.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with CHUNKS_PATH.open("w", encoding="utf-8") as file:
        for document in documents:
            record = {
                "text": document.page_content,
                "metadata": document.metadata
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")



def check_embeddings(
    documents: list[Document],
    embeddings_model: HuggingFaceEmbeddings
) -> dict:
    """
    Генерирует эмбеддинги для всех чанков и возвращает информацию о результате.
    """
    texts = [document.page_content for document in documents]

    start_time = time.time()
    vectors = embeddings_model.embed_documents(texts)
    generation_time = time.time() - start_time

    if not vectors:
        raise RuntimeError("Эмбеддинги не были сгенерированы")

    chunks_count = len(documents)
    embeddings_count = len(vectors)
    embedding_dimension = len(vectors[0])


    info = {
        "embedding_model": MODEL_NAME,
        "embedding_dimension": embedding_dimension,
        "chunks_count": chunks_count,
        "embeddings_count": embeddings_count,
        "generation_time_seconds": round(generation_time, 2),
        "knowledge_base_dir": str(KNOWLEDGE_BASE_DIR),
        "chunks_file": str(CHUNKS_PATH)
    }

    with EMBEDDING_INFO_PATH.open("w", encoding="utf-8") as file:
        json.dump(info, file, ensure_ascii=False, indent=2)

    return info


def build_faiss_index(
    documents: list[Document],
    embeddings_model: HuggingFaceEmbeddings
) -> dict:
    """
    Создаёт FAISS-индекс из чанков и сохраняет его на диск.
    """
    start_time = time.time()

    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings_model
    )

    build_time = time.time() - start_time

    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_DIR))

    info = {
        "vector_db": "FAISS",
        "index_dir": str(FAISS_INDEX_DIR),
        "index_files": [
            "index.faiss",
            "index.pkl"
        ],
        "chunks_count": len(documents),
        "build_time_seconds": round(build_time, 2)
    }

    with BUILD_INFO_PATH.open("w", encoding="utf-8") as file:
        json.dump(info, file, ensure_ascii=False, indent=2)

    return info


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_and_split_documents()

    save_chunks_debug_file(documents)

    embeddings_model = create_embedding_model()

    embedding_info = check_embeddings(documents, embeddings_model)

    print("\nЭмбеддинги успешно сгенерированы")
    print(f"Модель: {embedding_info['embedding_model']}")
    print(f"Размер эмбеддинга: {embedding_info['embedding_dimension']}")
    print(f"Количество чанков: {embedding_info['chunks_count']}")
    print(f"Время генерации: {embedding_info['generation_time_seconds']} секунд")
    print(f"Информация сохранена в: {EMBEDDING_INFO_PATH}")

    print("\nСоздание FAISS-индекса...")
    index_info = build_faiss_index(documents, embeddings_model)

    print("\nFAISS-индекс успешно создан")
    print(f"Папка индекса: {index_info['index_dir']}")
    print(f"Файлы индекса: {', '.join(index_info['index_files'])}")
    print(f"Количество чанков в индексе: {index_info['chunks_count']}")
    print(f"Время построения индекса: {index_info['build_time_seconds']} секунд")
    print(f"Информация сохранена в: {BUILD_INFO_PATH}")


if __name__ == "__main__":
    main()