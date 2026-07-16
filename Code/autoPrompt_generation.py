import pathlib
import json
# from anthropic import Anthropic (Commenting this out because I don't have access to the Anthropic API key)
from transformers import pipeline

BASE_DIR = pathlib.Path(__file__).parent.parent

SCENARIOS = [
    {
        "dir": "0_Non_Context",
        "scenario_id": "None",
        "needs_cf": False,
        "needs_summary": False,
        "prompt_prefix": "None_Adversarial_Prompt",
        "result_prefix": "non_context",
    },
    {
    "dir": "1_OrigCF_Context",
    "scenario_id": "ConsentForm",        
    "needs_cf": True,
    "needs_summary": False,
    "prompt_prefix": "CF_Adversarial_Prompt",      
    "result_prefix": "OrigCF_Context",             
},
{
    "dir": "2_SumCF_Context",
    "scenario_id": "CF_SUM",             
    "needs_cf": False,
    "needs_summary": True,
    "prompt_prefix": "CF_SUM_Adversarial_Prompt",  
    "result_prefix": "SumCF_Context",              
},
{
    "dir": "3_SUM_PARAG_Context",
    "scenario_id": "SUM & PAR",          
    "needs_cf": True,
    "needs_summary": True,
    "prompt_prefix": "SUM_PAR_Adversarial_Prompt", 
    "result_prefix": "SUM_PARAG_Context",          
},
{
    "dir": "4_PARAG_Context",            
    "scenario_id": "PAR",               
    "needs_cf": True,
    "needs_summary": False,
    "prompt_prefix": "PARAG_Adversarial_Prompt",   
    "result_prefix": "PARAG_Context",  
},
]

CRITERIA = ["C1", "C2", "C3"]
TARGET = 3

def find_consent_form_pairs(base_dir):
    pairs = []
    for f in base_dir.glob("*.txt"):
        if not f.name.startswith("SUM_"):
            summary_file = base_dir / f"SUM_{f.name}"
            pairs.append({
                "cf": f,
                "summary": summary_file if summary_file.exists() else None
            })
    return pairs


def validate_scenario(scenario, pair):
    if scenario["needs_cf"] and pair["cf"] is None:
        print(f"Scenario {scenario['scenario_id']} requires a consent form, but none was found for {pair}.")
        return False
    if scenario["needs_summary"] and pair["summary"] is None:
        print(f"Scenario {scenario['scenario_id']} requires a summary, but none was found for {pair}.")
        return False
    return True

def load_system_prompt(scenario, criterion):
    prompt_file = BASE_DIR / scenario["dir"] / f"{scenario['prompt_prefix']}_{criterion}.txt"
    if not prompt_file.exists():
        print(f"Prompt file {prompt_file} does not exist.")
        return None
    with open(prompt_file, 'r') as f:
        return f.read()

def build_user_message(scenario, cf_content, summary_content, cf_filename):
    if scenario["needs_cf"] and scenario["needs_summary"]:
        return f"Consent Form File: {cf_filename}\n\nConsent Form:\n{cf_content}\n\nSummary:\n{summary_content}"
    elif scenario["needs_cf"]:
        return f"Consent Form File: {cf_filename}\n\nConsent Form:\n{cf_content}"
    elif scenario["needs_summary"]:
        return f"Consent Form File: {cf_filename}\n\nSummary:\n{summary_content}"
    else:
        return "Generate the adversarial prompts without context."

def generate_batch(scenario, criterion, cf_content, summary_content, cf_filename, client): # this is going to be changed to use Llama model instead of Claude model. The code is written for Claude model, but I will change it to use Llama model.
    system_prompt = load_system_prompt(scenario, criterion)
    if system_prompt is None:
        return []
    user_message = build_user_message(scenario, cf_content, summary_content, cf_filename)
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
    result_text = result_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        results = json.loads(result_text)
    except json.JSONDecodeError:
        print("Invalid JSON, skipping batch...")
        results = []
    return results

def accumulate_prompts(scenario, criterion, cf_content, summary_content, cf_filename, client):
    accumulated_prompts = []
    max_retries = 20
    retries = 0
    while len(accumulated_prompts) < TARGET:
        if retries >= max_retries:
            print(f"Max retries reached for {scenario['scenario_id']} {criterion}, skipping...")
            break
        new_prompts = generate_batch(scenario, criterion, cf_content, summary_content, cf_filename, client)
        if not new_prompts:
            retries += 1
        else:
            retries = 0
            accumulated_prompts.extend(new_prompts)
    return accumulated_prompts[:TARGET]

def save_results(results, scenario, criterion):
    result_dir = BASE_DIR / scenario["dir"]
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{scenario['result_prefix']}{criterion}Results.txt"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {result_file}")


if __name__ == "__main__":
    # client = Anthropic() (Commenting this out because I don't have access to the Anthropic API key)
    client = pipeline(
    "text-generation",
    model=pathlib.Path("/home1/shared/Models/Llama/Llama-3.1-8B-Instruct"),
    device=1
    ) # this is just for testing purposes. Will be changed later.
    pairs = find_consent_form_pairs(BASE_DIR)
    for pair in pairs:
        cf_content = pair["cf"].read_text(encoding = "utf-8") if pair["cf"] else None
        cf_filename = pair["cf"].name if pair["cf"] else None
        summary_content = pair["summary"].read_text(encoding = "utf-8") if pair["summary"] else None
        for scenario in SCENARIOS:
            if not validate_scenario(scenario, pair):
                continue
            for criterion in CRITERIA:
                results = accumulate_prompts(scenario, criterion, cf_content, summary_content, cf_filename, client)
                save_results(results, scenario, criterion)
   