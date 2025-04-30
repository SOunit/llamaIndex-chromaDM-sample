# utils/gitignore_directory_reader.py

import os
import pathspec
from llama_index.core import SimpleDirectoryReader, Document

# 読み込みたい拡張子だけをセット（ここ自由に拡張できる）
TARGET_EXTENSIONS = {'.js', '.jsx', '.ts', '.tsx', '.cs', '.html', '.css', '.py', '.java', ".md"}

class GitIgnoreDirectoryReader(SimpleDirectoryReader):
    def __init__(self, input_dir: str, recursive: bool = True):
        super().__init__(input_dir=input_dir, recursive=recursive)

        # .gitignoreを読み込む
        gitignore_path = os.path.join(input_dir, ".gitignore")
        if os.path.exists(gitignore_path):
            with open(gitignore_path) as f:
                self.spec = pathspec.PathSpec.from_lines("gitwildmatch", f)
        else:
            self.spec = None

    def _should_include_file(self, file_path: str) -> bool:
        _, ext = os.path.splitext(file_path)
        return ext.lower() in TARGET_EXTENSIONS

    def _should_exclude_file(self, file_path: str) -> bool:
        if self.spec:
            relative_path = os.path.relpath(file_path, start=self.input_dir)
            return self.spec.match_file(relative_path)
        return False

    def load_data(self):
        documents = []
        for dirpath, _, filenames in os.walk(self.input_dir):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)

                if not self._should_include_file(file_path):
                    continue
                if self._should_exclude_file(file_path):
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    documents.append(Document(text=content, metadata={"file_path": file_path}))
                except Exception as e:
                    print(f"❌ Failed to load {file_path}: {e}")
        return documents
