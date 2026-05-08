from pathlib import Path
import json


BASE_DIR = Path(__file__).parent

SOURCE_DIR = BASE_DIR / "source"
OUTPUT_DIR = BASE_DIR / "knowledge_base"
TERMS_MAP_PATH = BASE_DIR / "terms_map.json"


def load_terms_map(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Файл словаря не найден: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def replace_terms(text: str, terms_map: dict[str, str]) -> str:

    for old_term in sorted(terms_map.keys(), key=len, reverse=True):
        new_term = terms_map[old_term]
        text = text.replace(old_term, new_term)

    return text


def process_files() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Папка source не найдена: {SOURCE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    terms_map = load_terms_map(TERMS_MAP_PATH)

    txt_files = list(SOURCE_DIR.glob("*.txt"))

    if not txt_files:
        return

    for source_file in txt_files:
        original_text = source_file.read_text(encoding="utf-8")
        replaced_text = replace_terms(original_text, terms_map)

        output_file = OUTPUT_DIR / source_file.name
        output_file.write_text(replaced_text, encoding="utf-8")

        print(f"Готово: {source_file.name} -> {output_file}")

    print(f"\nОбработано файлов: {len(txt_files)}")
    print(f"Результат сохранён в папку: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_files()