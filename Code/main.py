import pathlib
import json
import pandas as pd
import os
from transformers import pipeline as hf_pipeline
from preprocess import find_consent_forms, fix_bold_headings, generate_summary, generate_paragraph, save_file, CONTEXT_DIR
from autoPrompt_generation import find_consent_form_pairs, validate_scenario, accumulate_prompts, save_results, SCENARIOS, CRITERIA

class GroqClient:
    def __init__(self, model="llama-3.1-8b-instant"):
        from groq import Groq
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model

    def __call__(self, messages, max_new_tokens=2048, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens
        )
        return [{"generated_text": messages + [{"role": "assistant", "content": response.choices[0].message.content}]}]

def run_preprocess(client):
    consent_forms = find_consent_forms(CONTEXT_DIR)
    for cf_file in consent_forms:
        print(f"Preprocessing {cf_file.name}...")
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
        if summary_content:
            save_file(summary_content, cf_file.parent / f"{cf_file.stem}.SUM.txt")
        if paragraph_content:
            save_file(paragraph_content, cf_file.parent / f"{cf_file.stem}.PAR.txt")
        else:
            print(f"Skipping empty paragraph for {cf_file.name}")

def run_autoprompt(client):
    pairs = find_consent_form_pairs(CONTEXT_DIR)
    for pair in pairs:
        cf_content = pair["cf"].read_text(encoding="utf-8") if pair["cf"] else None
        cf_filename = pair["cf"].name if pair["cf"] else None
        cf_stem = pair["cf"].stem if pair["cf"] else None
        summary_content = pair["summary"].read_text(encoding="utf-8") if pair["summary"] else None
        paragraph_content = pair["paragraph"].read_text(encoding="utf-8") if pair["paragraph"] else None
        for scenario in SCENARIOS:
            if not validate_scenario(scenario, pair):
                continue
            for criterion in CRITERIA:
                results = accumulate_prompts(scenario, criterion, cf_content, summary_content, paragraph_content, cf_filename, client)
                save_results(results, scenario, criterion, cf_stem)

def encoder_(folder_link):
    folder_link = pathlib.Path(folder_link)
    # 1. Clear out old CSV files first
    for file in folder_link.iterdir():
        if file.is_file() and file.suffix == ".csv":
            print(f"Deleting file: {file.name}")
            file.unlink()
    target_csv_file = ""
    for file in folder_link.iterdir():
        if file.is_file() and file.name.endswith("Results.txt"):
            if target_csv_file == "":
                target_csv_file = folder_link / f"{file.name.strip('.txt').replace('C1','').replace('C2','').replace('C3', '')}.csv"
            print(40*"*")
            print(f"Processing {file.name}")
            print(f"File directory: {file}")
            print(f"Saving to: {target_csv_file.name}")
            process_and_append_to_csv(file, target_csv_file)

def process_and_append_to_csv(txt_file_path, output_csv_path):
    txt_file_path = pathlib.Path(txt_file_path)
    output_csv_path = pathlib.Path(output_csv_path)
    json_data = None
    if txt_file_path.is_file():
        with open(txt_file_path, 'r', encoding='utf-8-sig') as f:
            json_data = json.load(f)
    if json_data is None:
        print(f"Skipping empty or invalid file: {txt_file_path.name}")
        return
    df_new = pd.DataFrame(json_data)
    if 'criteria_targeted' in df_new.columns:
        df_new['criteria_targeted'] = df_new['criteria_targeted'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else x)
    if output_csv_path.exists():
        df_new.to_csv(output_csv_path, mode='a', index=False, header=False, encoding='utf-8-sig')
        print(f"Successfully appended {len(df_new)} items to existing {output_csv_path.name}\n")
    else:
        df_new.to_csv(output_csv_path, mode='w', index=False, header=True, encoding='utf-8')
        print(f"Created a brand new file and saved data to {output_csv_path.name}\n")

if __name__ == '__main__':
    print("What would you like to do?")
    print("1. Run full pipeline (preprocess + generate prompts)")
    print("2. Encode results to CSV")
    print("3. Run both")
    choice = input("Enter 1, 2, or 3: ").strip()
    if choice in ("1", "3"):
        print("Which model?")
        print("1. Llama 8B (local GPU)")
        print("2. Groq (Llama 8B via API)")
        model_choice = input("Enter 1 or 2: ").strip()
        if model_choice == "1":
            device = int(os.environ.get("DEVICE", 1))
            client = hf_pipeline(
                "text-generation",
                model="/home1/shared/Models/Llama/Llama-3.1-8B-Instruct",
                device=device
            )
        else:
            client = GroqClient()
        print("=== Step 1: Preprocessing ===")
        run_preprocess(client)
        print("=== Step 2: Generating Adversarial Prompts ===")
        run_autoprompt(client)
        print("=== Pipeline Done ===")
    if choice in ("2", "3"):
        pathdir = input("Please enter the directory path: \n")
        encoder_(pathdir)