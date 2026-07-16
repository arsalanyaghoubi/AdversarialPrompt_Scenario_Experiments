import pathlib
import random
from transformers import pipeline
from markdown_tree_parser.parser import parse_string
import re

BASE_DIR = pathlib.Path(__file__).parent.parent
CONTEXT_DIR = BASE_DIR / "Context"

client = pipeline(
    "text-generation",
    model=pathlib.Path("/home1/shared/Models/Llama/Llama-3.1-8B-Instruct"),
    device=1
)

def find_consent_forms(context_dir):
    consent_forms = []
    for f in context_dir.glob("*.txt"):
        if not f.name.startswith("SUM_") and not f.name.startswith("PAR_"):
            consent_forms.append(f)
    return consent_forms

def generate_summary(cf_content, client):
    response = client(
        [
            {"role": "system", "content": "You are a medical research assistant. Summarize the following clinical trial consent form clearly and concisely."},
            {"role": "user", "content": cf_content}
        ],
        max_new_tokens=1024,
        truncation=True,
    )
    return response[0]["generated_text"][-1]["content"]

def decomposer(sections, n=15):
    def count_sentences(text):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return len([s for s in sentences if s.strip()])
    scored = [(section, count_sentences(section)) for section in sections]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_n = [section for section, count in scored[:n]]
    return random.choice(top_n)

def generate_paragraph(cf_content):
    parsed = parse_string(cf_content)
    children = parsed.children
    sections = []
    current_parent = None
    for heading in children:
        source = (heading.source or "").strip()
        if not source:
            current_parent = heading
            continue
        if heading.text.isupper():
            current_parent = None
        if current_parent is not None:
            content = f"## {current_parent.text}\n\n### {heading.text}\n{heading.source}"
        else:
            content = f"## {heading.text}\n{heading.source}"
        sections.append(content)
    if not sections:
        paragraphs = [p.strip() for p in cf_content.split("\n\n") if p.strip()]
        if paragraphs:
            return decomposer(paragraphs, n=15)
        return ""
    return decomposer(sections, n=15)

def fix_bold_headings(cf_content):
    lines = cf_content.split("\n")
    fixed_lines = []
    for line in lines:
        # Matches: optional leading #s, then **text** → ## text
        match = re.match(r'^#+\s*\*\*(.*?)\*\*\s*$', line)
        standalone = re.match(r'^\*\*(.*?)\*\*\s*$', line)
        if match:
            text = match.group(1).strip()
            fixed_lines.append(f"## {text}")
        elif standalone:
            text = standalone.group(1).strip()
            fixed_lines.append(f"## {text}")
        else:
            fixed_lines.append(line)
    return "\n".join(fixed_lines)

def save_file(content, output_path):
    with open(output_path, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    consent_forms = find_consent_forms(CONTEXT_DIR)
    for cf_file in consent_forms:
        print(f"Processing {cf_file.name}...")
        with open(cf_file, 'r') as f:
            cf_content = f.read()
        cf_content = fix_bold_headings(cf_content)
        summary_content = generate_summary(cf_content, client)
        paragraph_content = generate_paragraph(cf_content)
        summary_filename = CONTEXT_DIR / f"SUM_{cf_file.name}"
        paragraph_filename = CONTEXT_DIR / f"PAR_{cf_file.name}"
        save_file(summary_content, summary_filename)
        save_file(paragraph_content, paragraph_filename)
        print(f"Processed {cf_file.name}: Summary saved to {summary_filename}, Paragraph saved to {paragraph_filename}")