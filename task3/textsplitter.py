import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


KNOWLEDGE_BASE_DIR = Path("../task2/knowledge_base")


def count_words(text: str) -> int:
    return len(text.split())


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    length_function=count_words,
    separators=["\n\n", "\n", ". ", " ", ""]
)


documents = []
for file_path in KNOWLEDGE_BASE_DIR.glob("*.txt"):
    print(file_path)
    text = file_path.read_text(encoding="utf-8")

    document = Document(
        page_content=text,
        metadata={
            "source_file": file_path.name,
            "source_path": str(file_path),
            "title": file_path.stem
        }
    )

    chunks = text_splitter.split_documents([document])

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["word_count"] = count_words(chunk.page_content)
        documents.append(chunk)

CHUNKS_PATH = Path("chunks.jsonl")

# with CHUNKS_PATH.open("w", encoding="utf-8") as f:
#
#     for doc in documents:
#
#         record = {
#
#             "text": doc.page_content,
#
#             "metadata": doc.metadata
#
#         }
#
#         f.write(json.dumps(record, ensure_ascii=False) + "\n")


print(f"Всего чанков: {len(documents)}")

