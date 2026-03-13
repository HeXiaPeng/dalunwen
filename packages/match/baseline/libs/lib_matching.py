import json
import os
import time
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from aliy_client import AliyClient

def qwen_match_batch(client, model, patient_text, batch_trials):
    sys_prompt = """
You are a clinical trial matching expert.
Your task is to evaluate a patient's eligibility for a list of clinical trials based on the patient's note and each trial's SPECIFIC criteria.

Input:
- A patient note describing their condition, history, and status.
- A list of clinical trials. Each trial has an ID, title, summary, and a LIST of criteria.

Output Requirement:
Return a STRICT JSON object with a single key "results", which is a list. Each item in the list must correspond to a trial in the input.

For EACH trial, you must iterate through its provided "criteria" list.
For EVERY single criterion in that list, you must provide an evaluation. Do not skip any criterion.

Structure for each trial result:
{
  "trial_id": "The Trial ID",
  "criteria": [
    {
      "index": 0, // The index of the criterion in the input list
      "text": "The exact text of the criterion being evaluated",
      "type": "inclusion" or "exclusion",
      "label": "include" | "not_included" | "exclude" | "not_excluded" | "not_relevant" | "not_enough_information",
      "evidence": "Reasoning citing the patient note"
    },
    ... (Repeat for ALL criteria in the input)
  ],
  "relevance": 0-100,
  "eligibility": -100 to 100
}

Label Definitions:
- "include": Patient MEETS this inclusion criterion.
- "not_included": Patient does NOT meet this inclusion criterion.
- "exclude": Patient MEETS this exclusion criterion (should be excluded).
- "not_excluded": Patient does NOT meet this exclusion criterion (safe).

CRITICAL RULES:
- If a trial has 10 criteria in the input, your output MUST have 10 evaluations in the "criteria" list.
- Be precise with "New Diagnosis" vs "Recurrent".
- Be precise with "Anatomy" (Supratentorial vs Spinal).
- Output valid JSON only.
"""
    
    # Pre-process batch_trials to ensure criteria are clean lists
    # The build_batches function already does some splitting, but let's ensure it's robust.
    # In the current code, build_batches calls extract_criteria which returns lists.
    # So batch_trials items already have "criteria": [{"type":..., "text":...}, ...]
    
    payload = {"patient": patient_text[:1500], "trials": batch_trials}
    
    # Use a simpler format for the user message to save tokens and confusion
    messages=[
        {"role":"system","content":sys_prompt},
        {"role":"user","content":json.dumps(payload, ensure_ascii=False)}
    ]
    
    try:
        content = client.chat_completions(messages=messages, model=model)
        # Clean up markdown code blocks if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content.rsplit("\n", 1)[0]
        content = content.strip()
        return json.loads(content)
    except Exception as e:
        print(f"Error in qwen_match_batch: {e}")
        # Print first 200 chars and last 200 chars of content for debugging
        if 'content' in locals():
            print(f"Content preview: {content[:200]} ... {content[-200:]}")
        return {"results": []}

def clean_split_criteria(text):
    if not text: return []
    lines = []
    # Split by newline
    raw_lines = text.split('\n')
    for l in raw_lines:
        l = l.strip()
        if not l: continue
        # Remove headers if present
        lower_l = l.lower()
        if lower_l.startswith("inclusion criteria") or lower_l.startswith("exclusion criteria"):
            # Try to strip the header
            if ":" in l:
                l = l.split(":", 1)[1].strip()
            else:
                continue # Skip the header line itself
        elif l.startswith(":"):
            l = l[1:].strip()
            
        if not l: continue
        if len(l) < 3: continue 
        lines.append(l)
    return lines

def build_batches(id2doc, candidates, patient_text, chunk=1, qid=None, base_dir=None):
    items=[]
    for cid, _ in candidates:
        trial=id2doc[cid]
        # We now have specific fields in id2doc
        inc_text = trial.get("inclusion_criteria", "")
        exc_text = trial.get("exclusion_criteria", "")
        
        # If specific fields are empty, fall back to parsing 'text' (backward compatibility)
        if not inc_text and not exc_text:
             inc, exc = extract_criteria(trial.get("text", ""))
        else:
             inc = clean_split_criteria(inc_text)
             exc = clean_split_criteria(exc_text)

        items.append({
            "trial_id": cid,
            "title": trial.get("brief_title", "") or trial.get("title", ""),
            "brief_title": trial.get("brief_title", ""),
            "phase": trial.get("phase", ""),
            "enrollment": trial.get("enrollment", ""),
            "drugs": trial.get("drugs_list", []),
            "diseases": trial.get("diseases_list", []),
            "brief_summary": trial.get("brief_summary", ""),
            "inclusion_criteria": inc_text,
            "exclusion_criteria": exc_text,
            "criteria": [{"type":"inclusion","text":l} for l in inc[:30]] + [{"type":"exclusion","text":l} for l in exc[:30]]
        })
    batches=[items[i:i+chunk] for i in range(0, len(items), chunk)]
    ids=[f"batch-{qid}-{i}-{int(time.time())}" for i in range(len(batches))]
    if base_dir:
        # Create a dedicated "inputs" subdirectory for inspection
        input_debug_dir = os.path.join(base_dir, "inputs")
        os.makedirs(input_debug_dir, exist_ok=True)
        
        for bid, trials_chunk in zip(ids, batches):
            # Save the exact JSON payload sent to LLM
            with open(os.path.join(input_debug_dir, f"{bid}_input.json"), "w", encoding="utf-8") as f:
                json.dump({"patient": patient_text, "trials": trials_chunk}, f, ensure_ascii=False, indent=2)
    
    return ids, batches


def extract_criteria(text, max_inc=30, max_exc=30):
    t = text or ""
    # Simple heuristic to split sections
    # Often criteria are bulleted or numbered.
    
    # Normalize headers
    low = t.lower()
    inc_start = low.find("inclusion criteria")
    exc_start = low.find("exclusion criteria")
    
    inc_text = ""
    exc_text = ""
    
    if inc_start != -1:
        if exc_start != -1 and exc_start > inc_start:
            inc_text = t[inc_start:exc_start]
            exc_text = t[exc_start:]
        elif exc_start != -1 and exc_start < inc_start:
            exc_text = t[exc_start:inc_start]
            inc_text = t[inc_start:]
        else:
            inc_text = t[inc_start:]
    elif exc_start != -1:
        exc_text = t[exc_start:]
    else:
        # No clear headers, treat whole text as inclusion? Or just summary.
        # TrialGPT uses a more complex parser, but for now let's try to split by newlines/bullets.
        inc_text = t
        
    def clean_split(block):
        lines = []
        # Remove the header line itself
        raw_lines = block.split('\n')
        for l in raw_lines:
            l = l.strip()
            if not l: continue
            if "inclusion criteria" in l.lower() or "exclusion criteria" in l.lower(): continue
            # Basic filtering of short noise
            if len(l) < 5: continue
            lines.append(l)
        return lines

    inc_lines = clean_split(inc_text)
    exc_lines = clean_split(exc_text)
    
    return inc_lines[:max_inc], exc_lines[:max_exc]
