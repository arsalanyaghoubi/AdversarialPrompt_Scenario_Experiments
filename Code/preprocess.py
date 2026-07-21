import pathlib
import random
from transformers import pipeline
from markdown_tree_parser.parser import parse_string
import re
import os

N_SECTIONS = 15
BASE_DIR = pathlib.Path(__file__).parent.parent
CONTEXT_DIR = BASE_DIR / "Context"

def find_consent_forms(context_dir):
    consent_forms = []
    for d in context_dir.iterdir():
        if d.is_dir():
            cf_file = d / f"{d.name}.txt"
            if cf_file.exists():
                consent_forms.append(cf_file)
    return consent_forms

def generate_summary(cf_content, client):
    response = client(
        [
            {"role": "system", "content": "As an intelligent principal investigator of a clinical trial, you must provide a clear summary using the consent form text.\nThe summary should include a statement that explain the Purpose of the Research, Procedures, Risks or Discomforts, Benefits, Alternatives, Confidentiality, Compensation & Treatment for Injury, Voluntariness & Withdrawal, Contact Information, Eligibility, Unknown Risks, Termination Criteria, Significant New Findings, Costs to Subject, Use of Data/Samples in Future Research, Compensation for Participation and Time.\n\nUse the following guidelines to create a summary in a paragraph style:\n- Summary must be at most 150 words\n- Simplify any complex terms or concepts\n- Make the summary highly understandable, recommended for eighth-grade level audience\n- Use respectful and empowering language for patients\n- Spell out acronyms upon first use\n- Include all relevant information without adding extra details\n- Use active voice\n- Keep the summary concise and to the point"},
            {"role": "user", "content": cf_content}
        ],
        max_new_tokens=1024,
        truncation=True,
    )
    return response[0]["generated_text"][-1]["content"]

def decomposer(sections, n=N_SECTIONS):
    def count_sentences(text):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        non_empty = []
        for s in sentences:
            if s.strip():
                non_empty.append(s)
        return len(non_empty)
    scored = []
    for section in sections:
        count = count_sentences(section)
        scored.append((section, count))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_n = []
    for section, count in scored[:n]:
        top_n.append(section)
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
        paragraphs = []
        for p in cf_content.split("\n\n"):
            p = p.strip()
            if p:
                paragraphs.append(p)
        if paragraphs:
            return decomposer(paragraphs, n=N_SECTIONS)
        return ""
    return decomposer(sections, n=N_SECTIONS)

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
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    device = int(os.environ.get("DEVICE", 1))
    client = pipeline(
    "text-generation",
    model = "/home1/shared/Models/Llama/Llama-3.1-8B-Instruct",
    device=device
    )
    consent_forms = find_consent_forms(CONTEXT_DIR)
    for cf_file in consent_forms:
        print(f"Processing {cf_file.name}...")
        with open(cf_file, 'r', encoding='utf-8') as f:
            cf_content = f.read()
        cf_content = fix_bold_headings(cf_content)
        try:
            summary_content = generate_summary(cf_content, client)
        except Exception as e:
            print(f"Error generating summary for {cf_file.name}: {e}")
            summary_content = None
        try:
            paragraph_content = generate_paragraph(cf_content)
        except Exception as e:
            print(f"Error generating paragraph for {cf_file.name}: {e}")
            paragraph_content = None
        summary_filename = cf_file.parent / f"{cf_file.stem}.SUM.txt"
        paragraph_filename = cf_file.parent / f"{cf_file.stem}.PAR.txt"
        if summary_content:
            save_file(summary_content, summary_filename)
        if paragraph_content:
            save_file(paragraph_content, paragraph_filename)
        else:
            print(f"Skipping empty paragraph for {cf_file.name}")
        print(f"Processed {cf_file.name}: Summary saved to {summary_filename}, Paragraph saved to {paragraph_filename}")