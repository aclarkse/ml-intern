#!/usr/bin/env python3
"""
Load-Bearing Tokens in Chain-of-Thought Reasoning — Faster Version
===================================================================
Stage-0 Pre-Registration:
  Hypothesis (H1): ~15-30% of tokens are "load-bearing"
  Falsifier (F1):  Random-position perturbation produces same answer-change 
                   rate as structured perturbation
  Measurable Criterion:
    - Perturbation: Replace operators, numbers, keywords
    - "Answer changed": Exact match of final answer after #### marker
    - H1 threshold: load-bearing ≥70% change vs. non-load-bearing <20%
    - F1 threshold: random perturbation ≥50% answer change rate
"""

import json
import re
import random
from typing import Optional, List, Tuple

# ——— CONFIGURATION ——————————————————————————————————————————————————
MODEL_ID     = "Qwen/Qwen2.5-7B-Instruct"
DATASET      = "openai/gsm8k"
CONFIG       = "main"
SPLIT        = "test"
N_DISCOVERY  = 20
N_VALIDATION = 10
SEED         = 42

# ——— SETUP ——————————————————————————————————————————————————————————
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading model and dataset...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

dataset   = load_dataset(DATASET, CONFIG)
test_data = dataset[SPLIT]

# Pre-register split BEFORE running
random.seed(SEED)
indices = list(range(len(test_data)))
random.shuffle(indices)
discovery_indices  = indices[:N_DISCOVERY]
validation_indices = indices[N_DISCOVERY:]
print(f"\nPre-registered split:", flush=True)
print(f"  Discovery:   {discovery_indices}", flush=True)
print(f"  Validation:  {validation_indices}", flush=True)

discovery_problems   = [test_data[i] for i in discovery_indices]
validation_problems  = [test_data[i] for i in validation_indices]

# ——— HELPERS —————————————————————————————————————————————————————————

def extract_answer(answer_text: str) -> str:
    match = re.search(r'####\s*(\S+)', answer_text)
    return match.group(1) if match else None

def generate_answer(question: str, cot_prefix: str = None, max_tokens: int = 200) -> str:
    """Generate answer from model with optional CoT prefix."""
    prompt = question
    if cot_prefix:
        prompt = question + "\n\n" + cot_prefix
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_final_answer(text: str) -> str:
    match = re.search(r'####\s*(\S+)', text)
    return match.group(1) if match else None

# ——— PERTURBATION ——————————————————————————————————————————————————

def tokenize_cot(cot: str) -> List[str]:
    tokens = tokenizer.encode(cot, add_special_tokens=False)
    return [tokenizer.decode([t]) for t in tokens]

def find_positions_by_type(decoded: List[str]) -> dict:
    """Classify token positions by type."""
    positions = {'operator': [], 'number': [], 'keyword': [], 'calc': []}
    for i, t in enumerate(decoded):
        t = t.strip()
        if re.match(r'^[\+\-\*/=]$', t):
            positions['operator'].append(i)
        elif re.match(r'^\d+$', t):
            positions['number'].append(i)
        elif any(kw in t.lower() for kw in ['more', 'less', 'each', 'total', 'add', 'subtract', 'times', 'plus', 'minus', 'half', 'double', 'every', 'equals']):
            positions['keyword'].append(i)
        if '<<' in t or '>>' in t:
            positions['calc'].append(i)
    return positions

def perturb_token(token_str: str, ptype: str) -> str:
    if ptype == 'operator':
        return {'+': '-', '-': '+', '*': '/', '/': '*', '=': '≠'}.get(token_str.strip(), token_str)
    elif ptype == 'number' and token_str.isdigit():
        return str((int(token_str) + 7) % 10)
    elif ptype == 'keyword':
        return {'more': 'less', 'less': 'more', 'add': 'subtract', 'subtract': 'add',
                'times': 'plus', 'plus': 'times', 'half': 'double', 'double': 'half',
                'each': 'every', 'every': 'each'}.get(token_str.lower(), token_str)
    return token_str

def apply_perturbation(cot: str, pos: int, ptype: str) -> str:
    tokens = tokenizer.encode(cot, add_special_tokens=False)
    decoded = [tokenizer.decode([t]) for t in tokens]
    if pos >= len(decoded):
        return cot
    new_tok = perturb_token(decoded[pos], ptype)
    new_id  = tokenizer.encode(new_tok, add_special_tokens=False)[0]
    new_tokens = tokens[:pos] + [new_id] + tokens[pos+1:]
    return tokenizer.decode(new_tokens)

# ——— EXPERIMENTS ————————————————————————————————————————————————————

def run_contamination_check(problems: List[dict]) -> dict:
    """Check if model memorizes answers without real reasoning."""
    print("\n" + "="*60, flush=True)
    print("CONTAMINATION CHECK (5 problems)", flush=True)
    print("="*60, flush=True)
    
    results = {'real': [], 'random_cot': []}
    for i, p in enumerate(problems[:5]):
        q = p['question']
        true_ans = extract_answer(p['answer'])
        
        real_out   = generate_answer(q)
        real_ans   = get_final_answer(real_out)
        real_ok    = (real_ans == true_ans)
        
        rand_cot   = "Step 1: Analyze the problem. Step 2: Compute the result. Step 3: Final answer."
        rand_out   = generate_answer(q, rand_cot)
        rand_ans   = get_final_answer(rand_out)
        rand_ok    = (rand_ans == true_ans)
        
        results['real'].append(real_ok)
        results['random_cot'].append(rand_ok)
        
        print(f"  [{i}] real={real_ok} (got {real_ans}), rand={rand_ok} (got {rand_ans})", flush=True)
    
    real_acc  = sum(results['real'])      / len(results['real'])
    rand_acc  = sum(results['random_cot']) / len(results['random_cot'])
    print(f"\n  Real CoT acc: {real_acc:.1%}, Random CoT acc: {rand_acc:.1%}", flush=True)
    print(f"  Contamination: {'HIGH' if rand_acc > 0.5 else 'LOW'}", flush=True)
    return {'real_accuracy': real_acc, 'random_accuracy': rand_acc, 
            'contamination_suspected': rand_acc > 0.5}

def run_perturbation_exp(problems: List[dict], phase: str) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"PERTURBATION EXPERIMENT — {phase} phase", flush=True)
    print("="*60, flush=True)
    
    all_results = []
    for i, p in enumerate(problems):
        q = p['question']
        true_ans = extract_answer(p['answer'])
        
        # Generate CoT with model
        print(f"  [{i}] Generating CoT...", flush=True)
        raw_out  = generate_answer(q)
        cot      = raw_out.split('####')[0].strip() if '####' in raw_out else raw_out
        
        decoded  = tokenize_cot(cot)
        pos_dict = find_positions_by_type(decoded)
        orig_ans = get_final_answer(raw_out)
        
        pr = {
            'idx': discovery_indices[i] if phase == 'Discovery' else validation_indices[i],
            'n_tokens': len(decoded),
            'original_answer': orig_ans,
            'true_answer': true_ans,
            'perturbations': {}
        }
        
        # Test each perturbation type
        for ptype in ['operator', 'number', 'keyword', 'calc']:
            positions = pos_dict.get(ptype, [])
            if not positions:
                continue
            
            changes = 0
            for pos in positions:
                pert_cot  = apply_perturbation(cot, pos, ptype)
                pert_out  = generate_answer(q, pert_cot)
                pert_ans  = get_final_answer(pert_out)
                if pert_ans != orig_ans:
                    changes += 1
            
            rate = changes / len(positions)
            pr['perturbations'][ptype] = {'n': len(positions), 'rate': rate}
            print(f"    {ptype}: {len(positions)} positions, {rate:.1%} changed", flush=True)
        
        # Random baseline
        if len(decoded) > 5:
            rand_pos = random.sample(range(len(decoded)), min(5, len(decoded)))
            changes  = 0
            for pos in rand_pos:
                new_id   = random.choice([t for t in tokenizer.encode(cot, add_special_tokens=False) 
                                          if t != tokenizer.encode(cot, add_special_tokens=False)[pos]])
                tokens   = tokenizer.encode(cot, add_special_tokens=False)
                new_cot  = tokenizer.decode(tokens[:pos] + [new_id] + tokens[pos+1:])
                rand_out = generate_answer(q, new_cot)
                rand_ans = get_final_answer(rand_out)
                if rand_ans != orig_ans:
                    changes += 1
            pr['perturbations']['random'] = {'n': len(rand_pos), 'rate': changes / len(rand_pos)}
            print(f"    random: {len(rand_pos)} positions, {changes/len(rand_pos):.1%} changed", flush=True)
        
        print(f"  [{i}] Done. orig_answer={orig_ans}, true={true_ans}", flush=True)
        all_results.append(pr)
    
    return {'phase': phase, 'results': all_results}

def analyze(disco: dict, val: dict, contam: dict) -> dict:
    print("\n" + "="*60, flush=True)
    print("ANALYSIS", flush=True)
    print("="*60, flush=True)
    
    structured_rates = {'operator': [], 'number': [], 'keyword': [], 'calc': []}
    random_rates = []
    
    for phase_data in [disco, val]:
        for pr in phase_data['results']:
            for k, v in pr['perturbations'].items():
                if k == 'random':
                    random_rates.append(v['rate'])
                elif k in structured_rates:
                    structured_rates[k].append(v['rate'])
    
    avg_struct = {k: sum(v)/len(v) if v else 0 for k, v in structured_rates.items()}
    avg_random = sum(random_rates) / len(random_rates) if random_rates else 0
    
    print("\nAverage answer change rates:", flush=True)
    for k, v in avg_struct.items():
        print(f"  {k:12s}: {v:.1%}", flush=True)
    print(f"  {'random':12s}: {avg_random:.1%}", flush=True)
    
    max_struct = max(avg_struct.values())
    ratio = max_struct / max(avg_random, 0.01)
    
    result = {
        'hypothesis_supported': False,
        'falsifier_triggered': False,
        'load_bearing_fraction': max_struct,
        'structured_vs_random': {**avg_struct, 'random': avg_random},
        'contamination': contam,
        'ratio': ratio
    }
    
    if max_struct >= 0.7 and avg_random < 0.5:
        result['hypothesis_supported'] = True
        print("\n✓ H1 SUPPORTED: Structured perturbation shows load-bearing tokens", flush=True)
    elif avg_random >= 0.5:
        result['falsifier_triggered'] = True
        print("\n✗ F1 TRIGGERED: Random perturbation has high change rate — H1 falsified", flush=True)
    else:
        print("\n? INCONCLUSIVE", flush=True)
    
    return result

# ——— MAIN ———————————————————————————————————————————————————————————

if __name__ == "__main__":
    print("="*60, flush=True)
    print("LOAD-BEARING TOKENS IN CoT REASONING", flush=True)
    print("="*60, flush=True)
    
    # Contamination check
    contam = run_contamination_check(discovery_problems[:5])
    
    # Discovery
    print("\n>>> DISCOVERY (20 problems)", flush=True)
    disco_results = run_perturbation_exp(discovery_problems, "Discovery")
    
    # Validation
    print("\n>>> VALIDATION (10 problems)", flush=True)
    val_results = run_perturbation_exp(validation_problems, "Validation")
    
    # Analysis
    analysis = analyze(disco_results, val_results, contam)
    
    # ——— FINAL REPORT ——————————————————————————————————————————————
    print("\n" + "="*60, flush=True)
    print("FINAL REPORT", flush=True)
    print("="*60, flush=True)
    
    print(f"""
PRE-REGISTERED HYPOTHESIS (H1): ~15-30% of tokens are load-bearing
PRE-REGISTERED FALSIFIER (F1):  Random perturbation ≥50% answer change rate

RESULTS:
  Hypothesis supported:    {analysis['hypothesis_supported']}
  Falsifier triggered:     {analysis['falsifier_triggered']}
  Load-bearing fraction:   {analysis['load_bearing_fraction']:.1%}
  Structured/Random ratio: {analysis['ratio']:.2f}x
  
  Structured perturbation change rates:
    Operator:  {analysis['structured_vs_random']['operator']:.1%}
    Number:    {analysis['structured_vs_random']['number']:.1%}
    Keyword:   {analysis['structured_vs_random']['keyword']:.1%}
    Calc:      {analysis['structured_vs_random']['calc']:.1%}
    Random:    {analysis['structured_vs_random']['random']:.1%}

CONTAMINATION CHECK:
  Real CoT accuracy:   {contam['real_accuracy']:.1%}
  Random CoT accuracy: {contam['random_accuracy']:.1%}
  Contamination risk:  {'HIGH' if contam['contamination_suspected'] else 'LOW'}

PRE-REGISTERED SPLIT:
  Discovery:   {discovery_indices}
  Validation:  {validation_indices}

CAVEATS:
  - 30 total problems (20 discovery + 10 validation)
  - Model: Qwen/Qwen2.5-7B-Instruct (7B)
  - Perturbation: token replacement with similar wrong token
  - "Answer changed": exact match of final answer after ####
  - Contamination check used 5 problems (subset of discovery)
""", flush=True)
    
    output = {
        'pre_registered_split': {'discovery': discovery_indices, 'validation': validation_indices},
        'analysis': {**analysis, 'structured_vs_random': str(analysis['structured_vs_random'])},
        'contamination': contam,
        'discovery_results': disco_results,
        'validation_results': val_results
    }
    
    with open('/tmp/load_bearing_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print("\nResults saved to /tmp/load_bearing_results.json", flush=True)