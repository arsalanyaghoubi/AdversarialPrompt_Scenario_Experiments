import os

OUTPUT_FILE = "all_project_code.txt"
IGNORE_DIRS = {".git", "node_modules", "venv", ".venv", "bin", "obj", "dist", "build", "__pycache__", ".idea",
               ".vscode"}
ALLOWED_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".cs", ".cpp", ".c", ".h", ".java", ".html", ".css", ".json",
                      ".sql", ".md"}


def extract_code(root_dir):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Exclude specified directories in-place
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]

            for file in filenames:
                ext = os.path.splitext(file)[1].lower()
                if ext in ALLOWED_EXTENSIONS and file != OUTPUT_FILE:
                    file_path = os.path.join(dirpath, file)
                    outfile.write(f"{'=' * 50}\nFILE: {file_path}\n{'=' * 50}\n")
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
                            outfile.write(infile.read())
                        outfile.write("\n\n")
                    except Exception as e:
                        outfile.write(f"[Error reading file: {e}]\n\n")


if __name__ == "__main__":
    extract_code(".")
    print(f"Extraction complete! Saved to {OUTPUT_FILE}")