
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TEXT_EXTENSIONS = {
    ".md", ".txt", ".py", ".ipynb",
    ".html", ".css", ".json", ".yaml",
    ".yml", ".toml"
}

REPLACEMENTS = {
    "â€”": ", ",
    "â€“": ", ",
    "â€˜": "'",
    "â€™": "'",
    "â€œ": '"',
    "â€": '"',
    "â€¦": "...",
    "&#x20;": " ",
    "&#32;": " ",
    "&nbsp;": " ",
    "\u2014": ", ",
    "\u2013": ", ",
}

SKIP_DIRS = {
    ".git", "__pycache__", ".venv",
    "venv", "node_modules"
}

files_changed = 0
replacements_made = 0

for path in ROOT.rglob("*"):
    if not path.is_file():
        continue

    if path.name == Path(__file__).name:
        continue

    if any(part in SKIP_DIRS for part in path.parts):
        continue

    if path.suffix.lower() not in TEXT_EXTENSIONS:
        continue

    try:
        original = path.read_text(encoding="utf-8-sig")
    except (UnicodeDecodeError, OSError):
        print(f"Skipped unreadable file: {path.relative_to(ROOT)}")
        continue

    cleaned = original

    for old, new in REPLACEMENTS.items():
        count = cleaned.count(old)
        if count:
            replacements_made += count
            cleaned = cleaned.replace(old, new)

    if cleaned != original:
        path.write_text(cleaned, encoding="utf-8", newline="")
        files_changed += 1
        print(f"Cleaned: {path.relative_to(ROOT)}")

print()
print(f"Files cleaned: {files_changed}")
print(f"Text replacements: {replacements_made}")
print("Text cleanup complete.")