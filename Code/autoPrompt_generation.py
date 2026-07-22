import pathlib
import json
# from anthropic import Anthropic (Commenting this out because I don't have access to the Anthropic API key)
from transformers import pipeline
import re
import os

BASE_DIR = pathlib.Path(__file__).parent.parent
CONTEXT_DIR = BASE_DIR / "Context"

with open(BASE_DIR / "config.json") as f:
    CONFIG = json.load(f)

CF_TARGET = CONFIG["targets"]["CF"]
SUM_TARGET = CONFIG["targets"]["SUM"]
PAR_TARGET = CONFIG["targets"]["PAR"]
SUM_PAR_TARGET = CONFIG["targets"]["SUM_PAR"]
N_PARAGRAPHS = CONFIG["n_paragraphs"]

SCENARIOS = [
    {
        "dir": "1_OrigCF_Context",
        "scenario_id": "ConsentForm",        
        "needs_cf": True,
        "needs_summary": False,
        "needs_paragraph": False,
        "prompt_prefix": "CF_Adversarial_Prompt",      
        "result_prefix": "OrigCF_Context", 
        "target": CF_TARGET,            
    },
    {
        "dir": "2_SumCF_Context",
        "scenario_id": "CF_SUM",             
        "needs_cf": False,
        "needs_summary": True,
        "needs_paragraph": False,
        "prompt_prefix": "CF_SUM_Adversarial_Prompt",  
        "result_prefix": "SumCF_Context",
        "target": SUM_TARGET,   
    },
    {
        "dir": "3_SUM_PARAG_Context",
        "scenario_id": "SUM & PAR",          
        "needs_cf": False,
        "needs_summary": False,
        "needs_paragraph": False,
        "needs_sum_par": True,
        "prompt_prefix": "SUM_PAR_Adversarial_Prompt", 
        "result_prefix": "SUM_PARAG_Context",
        "target": SUM_PAR_TARGET,          
    },
    {
        "dir": "4_PARAG_Context",            
        "scenario_id": "PAR",               
        "needs_cf": False,
        "needs_summary": False,
        "needs_paragraph": True,
        "prompt_prefix": "PARAG_Adversarial_Prompt",   
        "result_prefix": "PARAG_Context",
        "target": PAR_TARGET,  
    },
]

CRITERIA = ["C1", "C2", "C3"]

def find_consent_form_pairs(base_dir):
    pairs = []
    for d in base_dir.iterdir():
        if d.is_dir():
            cf_file = d / f"{d.name}.txt"
            if not cf_file.exists():
                continue
            stem = d.name
            summary_file = d / f"{stem}.SUM.txt"
            paragraphs = []
            sum_paragraphs = []
            for i in range (1,N_PARAGRAPHS + 1):
                sum_par_file = d / f"{stem}.SUM_PAR{i}.txt"
                if sum_par_file.exists():
                    sum_paragraphs.append(sum_par_file)
                par_file = d / f"{stem}.PAR{i}.txt"
                if par_file.exists():
                    paragraphs.append(par_file)
            pairs.append({
                "cf": cf_file,
                "summary": summary_file if summary_file.exists() else None,
                "paragraphs": paragraphs,
                "sum_paragraphs": sum_paragraphs
            })
    return pairs

def validate_scenario(scenario, pair):
    if scenario["needs_cf"] and pair["cf"] is None:
        print(f"Scenario {scenario['scenario_id']} requires a consent form, but none was found for {pair}.")
        return False
    if scenario["needs_summary"] and pair["summary"] is None:
        print(f"Scenario {scenario['scenario_id']} requires a summary, but none was found for {pair}.")
        return False
    if scenario["needs_paragraph"] and not pair["paragraphs"]:
        print(f"Scenario {scenario['scenario_id']} requires a paragraph, but none was found for {pair}.")
        return False
    if scenario.get("needs_sum_par") and not pair["sum_paragraphs"]:
        print(f"Scenario {scenario['scenario_id']} requires a summary paragraph, but none was found for {pair}.")
        return False
    return True

def load_system_prompt(scenario, criterion):
    prompt_file = BASE_DIR / scenario["dir"] / f"{scenario['prompt_prefix']}_{criterion}.txt"
    if not prompt_file.exists():
        print(f"Prompt file {prompt_file} does not exist.")
        return None
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    return content.replace("{n_prompts}", str(scenario["target"]))

def build_user_message(scenario, cf_content, summary_content, paragraph_content, cf_filename):
    if scenario["needs_cf"] and scenario["needs_summary"]:
        return f"Consent Form File: {cf_filename}\n\nConsent Form:\n{cf_content}\n\nSummary:\n{summary_content}"
    elif scenario["needs_summary"] and scenario["needs_paragraph"]:
        return f"Consent Form File: {cf_filename}\n\nSummary:\n{summary_content}\n\nParagraph:\n{paragraph_content}"
    elif scenario["needs_cf"]:
        return f"Consent Form File: {cf_filename}\n\nConsent Form:\n{cf_content}"
    elif scenario["needs_summary"]:
        return f"Consent Form File: {cf_filename}\n\nSummary:\n{summary_content}"
    elif scenario["needs_paragraph"]:
        return f"Consent Form File: {cf_filename}\n\nParagraph:\n{paragraph_content}"
    elif scenario.get("needs_sum_par"):
        return f"Consent Form File: {cf_filename}\n\nContext:\n{summary_content}"
    else:
        return "Generate the adversarial prompts without context."

def generate_batch(scenario, criterion, cf_content, summary_content, paragraph_content, cf_filename, client): # this is going to be changed to use Llama model instead of Claude model. The code is written for Claude model, but I will change it to use Llama model.
    system_prompt = load_system_prompt(scenario, criterion)
    if system_prompt is None:
        return []
    user_message = build_user_message(scenario, cf_content, summary_content, paragraph_content, cf_filename)
    # response = client.messages.create(
    #     model="claude-sonnet-4-6",
    #     max_tokens = 2048,
    #     system = system_prompt,
    #     messages = [{"role": "user", "content": user_message}]
    # )
    response = client(
        [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ],
        max_new_tokens = 2048
    )
    # result_text = response.content[0].text.strip() # changing for the Llama model
    result_text = response[0]["generated_text"][-1]["content"].strip()
    results = []
    start = result_text.find('[')
    end = result_text.rfind(']')
    if start != -1 and end != -1:
        try:
            results = json.loads(result_text[start:end+1])
        except json.JSONDecodeError:
            pass
    if not results:
        for match in re.finditer(r'\{.*?\}', result_text, re.DOTALL):
            try:
                results.append(json.loads(match.group()))
            except json.JSONDecodeError:
                continue
    return results

def accumulate_prompts(scenario, criterion, cf_content, summary_content, paragraph_content, cf_filename, client):
    accumulated_prompts = []
    max_retries = 20
    retries = 0
    while len(accumulated_prompts) < scenario["target"]:
        if retries >= max_retries:
            print(f"Max retries reached for {scenario['scenario_id']} {criterion}, skipping...")
            break
        new_prompts = generate_batch(scenario, criterion, cf_content, summary_content, paragraph_content, cf_filename, client)
        if not new_prompts:
            retries += 1
        else:
            retries = 0
            accumulated_prompts.extend(new_prompts)
    return accumulated_prompts[:scenario["target"]]

def save_results(results, scenario, criterion, cf_stem, paragraph_index=None):
    result_dir = BASE_DIR / scenario["dir"] / cf_stem
    result_dir.mkdir(parents=True, exist_ok=True)
    if paragraph_index is not None:
        result_file = result_dir / f"{cf_stem}_PAR{paragraph_index}_{scenario['result_prefix']}{criterion}Results.txt"
    else:
        result_file = result_dir / f"{cf_stem}_{scenario['result_prefix']}{criterion}Results.txt"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {result_file}")

if __name__ == "__main__":
    # client = Anthropic() (Commenting this out because I don't have access to the Anthropic API key)
    device = int(os.environ.get("DEVICE", 1))
    client = pipeline(
    "text-generation",
    model="/home1/shared/Models/Llama/Llama-3.1-8B-Instruct",
    device=device
    ) # this is just for testing purposes. Will be changed later.
    pairs = find_consent_form_pairs(CONTEXT_DIR)
    for pair in pairs:
        if pair["cf"]:
            cf_content = pair["cf"].read_text(encoding="utf-8")
            cf_filename = pair["cf"].name
            cf_stem = pair["cf"].stem
        else:
            cf_content = None
            cf_filename = None
            cf_stem = None

        if pair["summary"]:
            summary_content = pair["summary"].read_text(encoding="utf-8")
        else:
            summary_content = None
        for scenario in SCENARIOS:
            if not validate_scenario(scenario, pair):
                continue
            if scenario["needs_paragraph"]:
                for i, par_file in enumerate(pair["paragraphs"], start=1):
                    paragraph_content = par_file.read_text(encoding="utf-8")
                    for criterion in CRITERIA:
                        results = accumulate_prompts(scenario, criterion, cf_content, summary_content, paragraph_content, cf_filename, client)
                        save_results(results, scenario, criterion, cf_stem, paragraph_index=i)
            elif scenario.get("needs_sum_par"):
                for i, sum_par_file in enumerate(pair["sum_paragraphs"], start=1):
                    sum_par_content = sum_par_file.read_text(encoding="utf-8")
                    for criterion in CRITERIA:
                        results = accumulate_prompts(scenario, criterion, cf_content, sum_par_content, None, cf_filename, client)
                        save_results(results, scenario, criterion, cf_stem, paragraph_index=i)
            else:
                for criterion in CRITERIA:
                    results = accumulate_prompts(scenario, criterion, cf_content, summary_content, None, cf_filename, client)
                    save_results(results, scenario, criterion, cf_stem)