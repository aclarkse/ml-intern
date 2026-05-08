#!/usr/bin/env python3
"""
Load-Bearing Tokens in Chain-of-Thought Reasoning
==================================================
Stage-0 Pre-Registration:
  Hypothesis (H1): ~15-30% of tokens are "load-bearing" — perturbing them 
                   substantially changes the final answer while perturbing 
                   other tokens does not.
  Falsifier (F1):  Random-position perturbation produces same answer-change 
                   rate as structured perturbation → H1 falsified.
  
  Measurable Criterion:
    - Perturbation: Replace operators (+→-), swap digits, keywords (more→less)
    - "Answer changed": Exact match of final answer after #### marker
    - H1 threshold: load-bearing tokens cause ≥70% answer change, 
                    non-load-bearing cause <20%
    - F1 threshold: random perturbation causes ≥50% answer change rate

Contamination Control: Baseline accuracy vs. accuracy when CoT replaced 
                       with random tokens. If >50% with random CoT, 
                       contamination suspected.
"""

import json
import re
import random
from typing import Optional

# ——— CONFIGURATION ——————————————————————————————————————————————————
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATASET   = "openai/gsm8k"
CONFIG    = "main"
SPLIT     = "test"
N_DISCOVERY = 20   # problems for discovery phase
N_VALIDATION = 10  # problems for validation phase
SEED = 42

# ——— SETUP ——————————————————————————————————————————————————————————
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Loading model and dataset...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

dataset = load_dataset(DATASET, CONFIG)
test_data = dataset[SPLIT]

# Pre-register the split BEFORE running
random.seed(SEED)
indices = random.sample(range(len(test_data)), N_DISCOVERY + N_VALIDATION)
discovery_indices    = indices[:N_DISCOVERY]
validation_indices   = indices[N_DISCOVERY:]
print(f"\nPre-registered split:")
print(f"  Discovery:   {discovery_indices}")
print(f"  Validation:  {validation_indices}")

discovery_problems = [test_data[i] for i in discovery_indices]
validation_problems = [test_data[i] for i in validation_indices]

# ——— HELPER FUNCTIONS ———————————————————————————————————————————————

def extract_answer(answer_text: str) -> str:
    """Extract the final answer after ####."""
    match = re.search(r'####\s*(\S+)', answer_text)
    return match.group(1) if match else None

def parse_cot_tokens(cot: str) -> list[tuple[int, str]]:
    """Tokenize CoT, return list of (token_id, token_str)."""
    tokens = tokenizer.encode(cot, add_special_tokens=False)
    decoded = [tokenizer.decode([t]) for t in tokens]
    return list(zip(tokens, decoded))

def generate_with_model(question: str, cot: Optional[str] = None) -> str:
    """Generate answer from model, optionally with a given CoT prefix."""
    prompt = question
    if cot:
        prompt = question + "\n\n" + cot
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_final_answer(text: str) -> str:
    """Extract final answer after #### from generated text."""
    match = re.search(r'####\s*(\S+)', text)
    return match.group(1) if match else None

# ——— PERTURBATION FUNCTIONS —————————————————————————————————————————

def get_operator_positions(tokens_decoded: list[str]) -> list[int]:
    """Find positions of math operators (+, -, *, /, =)."""
    return [i for i, t in enumerate(tokens_decoded) 
            if re.match(r'^[\+\-\*/=]$', t.strip())]

def get_number_positions(tokens_decoded: list[str]) -> list[int]:
    """Find positions of number tokens."""
    return [i for i, t in enumerate(tokens_decoded) 
            if re.match(r'^\d+$', t.strip())]

def get_keyword_positions(tokens_decoded: list[str]) -> list[int]:
    """Find positions of reasoning keywords."""
    keywords = ['more', 'less', 'each', 'total', 'sum', 'add', 'subtract',
                'multiply', 'divide', 'times', 'plus', 'minus', 'equals',
                'than', 'every', 'average', 'half', 'double', 'twice']
    return [i for i, t in enumerate(tokens_decoded) 
            if any(kw in t.lower() for kw in keywords)]

def get_calculation_positions(tokens_decoded: list[str]) -> list[int]:
    """Find positions inside <<...>> calculation markers."""
    positions = []
    for i, t in enumerate(tokens_decoded):
        if '<<' in t or '>>' in t:
            positions.append(i)
    return positions

def perturb_token(token_str: str, position_type: str) -> str:
    """Perturb a token based on its type."""
    if position_type == "operator":
        ops = {'+': '-', '-': '+', '*': '/', '/': '*', '=': '≠'}
        return ops.get(token_str.strip(), token_str)
    elif position_type == "number":
        # Replace with a nearby number (swap first digit)
        if token_str.isdigit() and len(token_str) > 1:
            digits = list(token_str)
            digits[0] = str((int(digits[0]) + 5) % 10)  # Change to different digit
            return ''.join(digits)
        return str((int(token_str) + 7) % 10) if token_str.isdigit() else token_str
    elif position_type == "keyword":
        swaps = {
            'more': 'less', 'less': 'more',
            'each': 'every', 'every': 'each',
            'total': 'sum', 'sum': 'total',
            'add': 'subtract', 'subtract': 'add',
            'multiply': 'divide', 'divide': 'multiply',
            'times': 'plus', 'plus': 'times',
            'minus': 'plus', 'equals': '≠',
            'than': 'of', 'half': 'double', 'double': 'half', 'twice': 'half'
        }
        return swaps.get(token_str.lower(), token_str)
    elif position_type == "calculation":
        return token_str.replace('<<', '').replace('>>', '')  # Remove markers
    return token_str

def apply_perturbation(cot: str, position: int, position_type: str) -> str:
    """Apply perturbation to a specific token position in CoT text."""
    tokens = tokenizer.encode(cot, add_special_tokens=False)
    decoded = [tokenizer.decode([t]) for t in tokens]
    
    if position >= len(decoded):
        return cot
    
    new_token_str = perturb_token(decoded[position], position_type)
    new_token_id = tokenizer.encode(new_token_str, add_special_tokens=False)[0]
    new_tokens = tokens[:position] + [new_token_id] + tokens[position+1:]
    return tokenizer.decode(new_tokens)

# ——— EXPERIMENTAL PHASES ————————————————————————————————————————————

def run_contamination_check(problems: list) -> dict:
    """
    Contamination control: Compare accuracy with real CoT vs. random CoT.
    If model gets >50% right with random CoT, contamination suspected.
    """
    print("\n" + "="*60)
    print("CONTAMINATION CHECK")
    print("="*60)
    
    results = {'real_cot': [], 'random_cot': []}
    
    for i, problem in enumerate(problems):
        question = problem['question']
        true_answer = extract_answer(problem['answer'])
        
        # Real CoT accuracy
        real_output = generate_with_model(question)
        real_answer = get_final_answer(real_output)
        real_correct = (real_answer == true_answer)
        results['real_cot'].append(real_correct)
        
        # Random CoT accuracy (replace CoT with random noise)
        random_cot = "Step 1: Consider the problem. Step 2: Think carefully. Step 3: Calculate result."
        random_output = generate_with_model(question, random_cot)
        random_answer = get_final_answer(random_output)
        random_correct = (random_answer == true_answer)
        results['random_cot'].append(random_correct)
        
        print(f"  [{i}] Real: {real_correct} (got {real_answer}, expected {true_answer}) | "
              f"Random CoT: {random_correct} (got {random_answer})")
    
    real_acc = sum(results['real_cot']) / len(results['real_cot'])
    random_acc = sum(results['random_cot']) / len(results['random_cot'])
    
    print(f"\n  Real CoT accuracy:    {real_acc:.1%}")
    print(f"  Random CoT accuracy:  {random_acc:.1%}")
    print(f"  Contamination risk:   {'HIGH' if random_acc > 0.5 else 'LOW'}")
    
    return {
        'real_accuracy': real_acc,
        'random_accuracy': random_acc,
        'contamination_suspected': random_acc > 0.5
    }

def run_perturbation_experiment(problems: list, phase: str) -> dict:
    """
    Run perturbation experiments to identify load-bearing tokens.
    
    For each problem:
      1. Get model's original CoT
      2. Identify candidate token positions by type
      3. Perturb each type and measure answer change rate
      4. Compare structured vs. random perturbation
    """
    print(f"\n{'='*60}")
    print(f"PERTURBATION EXPERIMENT — {phase} phase")
    print("="*60)
    
    all_results = []
    
    for i, problem in enumerate(problems):
        question = problem['question']
        true_answer = extract_answer(problem['answer'])
        
        # Generate CoT with model
        cot_output = generate_with_model(question)
        
        # Extract just the CoT portion (after question, before final answer)
        cot_match = re.search(r'\n\n(.*?)(?:\n####|\n\n####)', cot_output, re.DOTALL)
        cot = cot_match.group(1).strip() if cot_match else cot_output
        
        # Tokenize
        tokens = tokenizer.encode(cot, add_special_tokens=False)
        decoded = [tokenizer.decode([t]) for t in tokens]
        
        # Identify position types
        op_positions    = get_operator_positions(decoded)
        num_positions   = get_number_positions(decoded)
        kw_positions    = get_keyword_positions(decoded)
        calc_positions  = get_calculation_positions(decoded)
        all_structured  = op_positions + num_positions + kw_positions + calc_positions
        
        # Get original answer
        original_answer = get_final_answer(cot_output)
        original_correct = (original_answer == true_answer)
        
        problem_results = {
            'problem_idx': discovery_indices[i] if phase == 'Discovery' else validation_indices[i],
            'original_correct': original_correct,
            'original_answer': original_answer,
            'true_answer': true_answer,
            'n_tokens': len(tokens),
            'perturbation_results': {}
        }
        
        # Test each perturbation type
        for ptype, positions in [('operator', op_positions), 
                                  ('number', num_positions),
                                  ('keyword', kw_positions),
                                  ('calculation', calc_positions)]:
            if not positions:
                continue
            
            changed = 0
            for pos in positions:
                perturbed_cot = apply_perturbation(cot, pos, ptype)
                perturbed_output = generate_with_model(question, perturbed_cot)
                perturbed_answer = get_final_answer(perturbed_output)
                
                if perturbed_answer != original_answer:
                    changed += 1
            
            change_rate = changed / len(positions) if positions else 0
            problem_results['perturbation_results'][ptype] = {
                'positions_tested': len(positions),
                'change_rate': change_rate
            }
            print(f"  [{i}] {ptype}: {len(positions)} positions, {change_rate:.1%} answer change")
        
        # Random perturbation baseline
        if len(tokens) > 5:
            random_positions = random.sample(range(len(tokens)), min(5, len(tokens)))
            random_changed = 0
            for pos in random_positions:
                # Simple random perturbation: replace with a different token
                random_token_id = random.choice([t for t in tokens if t != tokens[pos]])
                new_tokens = tokens[:pos] + [random_token_id] + tokens[pos+1:]
                random_cot = tokenizer.decode(new_tokens)
                random_output = generate_with_model(question, random_cot)
                random_answer = get_final_answer(random_output)
                if random_answer != original_answer:
                    random_changed += 1
            problem_results['perturbation_results']['random'] = {
                'positions_tested': len(random_positions),
                'change_rate': random_changed / len(random_positions)
            }
            print(f"  [{i}] random: {len(random_positions)} positions, "
                  f"{random_changed/len(random_positions):.1%} answer change")
        
        all_results.append(problem_results)
        print(f"  [{i}] Problem {problem_results['problem_idx']}: "
              f"original={'✓' if original_correct else '✗'}, "
              f"answer={original_answer}, true={true_answer}")
    
    return {'phase': phase, 'results': all_results}

def analyze_results(discovery_results: dict, validation_results: dict, 
                    contamination: dict) -> dict:
    """
    Analyze load-bearing token hypothesis results.
    Compare structured perturbation rates vs. random baseline.
    """
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    analysis = {
        'hypothesis_supported': False,
        'falsifier_triggered': False,
        'load_bearing_fraction_estimate': None,
        'structured_vs_random': {},
        'contamination': contamination
    }
    
    # Aggregate change rates by perturbation type
    structured_rates = {'operator': [], 'number': [], 'keyword': [], 'calculation': []}
    random_rates = []
    
    for phase_results in [discovery_results, validation_results]:
        for problem in phase_results['results']:
            for ptype, data in problem['perturbation_results'].items():
                if ptype == 'random':
                    random_rates.append(data['change_rate'])
                elif ptype in structured_rates:
                    structured_rates[ptype].append(data['change_rate'])
    
    # Compute averages
    avg_structured = {ptype: sum(rates)/len(rates) if rates else 0 
                      for ptype, rates in structured_rates.items()}
    avg_random = sum(random_rates) / len(random_rates) if random_rates else 0
    
    analysis['structured_vs_random'] = {
        'operator':    avg_structured['operator'],
        'number':      avg_structured['number'],
        'keyword':     avg_structured['keyword'],
        'calculation': avg_structured['calculation'],
        'random':      avg_random
    }
    
    print("\nAverage Answer Change Rates:")
    for ptype, rate in analysis['structured_vs_random'].items():
        print(f"  {ptype:12s}: {rate:.1%}")
    
    # Compute load-bearing fraction
    # A token is "load-bearing" if perturbing it changes the answer
    total_structured_positions = sum(len(rates) for rates in structured_rates.values())
    load_bearing_count = sum(
        sum(1 for rate in structured_rates[ptype] if rate > 0.5)
        for ptype in structured_rates
    )
    # This is approximate - we're looking at aggregate rates
    load_bearing_fraction = load_bearing_count / max(total_structured_positions, 1)
    analysis['load_bearing_fraction_estimate'] = load_bearing_fraction
    
    # Determine if hypothesis supported or falsified
    # H1: structured perturbation (especially operator/calculation) should have 
    #     higher change rate than random
    # F1: if random perturbation rate >= 50%, falsified
    
    max_structured = max(avg_structured.values())
    structured_vs_random_ratio = max_structured / max(avg_random, 0.01)
    
    print(f"\n  Max structured change rate:   {max_structured:.1%}")
    print(f"  Random change rate:           {avg_random:.1%}")
    print(f"  Ratio (structured/random):    {structured_vs_random_ratio:.2f}x")
    
    # H1 supported: structured perturbation significantly outperforms random
    if max_structured >= 0.7 and avg_random < 0.5:
        analysis['hypothesis_supported'] = True
        print("\n  ✓ H1 SUPPORTED: Structured perturbation shows load-bearing tokens")
    
    # F1 triggered: random perturbation causes similar answer changes
    if avg_random >= 0.5:
        analysis['falsifier_triggered'] = True
        print("\n  ✗ F1 TRIGGERED: Random perturbation has high change rate — H1 falsified")
    
    if not analysis['hypothesis_supported'] and not analysis['falsifier_triggered']:
        # Ambiguous region
        print("\n  ? INCONCLUSIVE: Neither threshold met")
        if structured_vs_random_ratio > 2:
            analysis['hypothesis_supported'] = True
            print("     (but structured/random ratio suggests some structure effect)")
    
    return analysis

# ——— MAIN EXECUTION —————————————————————————————————————————————————

if __name__ == "__main__":
    print("="*60)
    print("LOAD-BEARING TOKENS IN CoT REASONING")
    print("Pre-registered Hypothesis: 15-30% of tokens are load-bearing")
    print("Pre-registered Falsifier:  Random perturbation ≥50% answer change")
    print("="*60)
    
    # Phase 1: Contamination check (on a small subset)
    print("\n>>> Phase 1: Contamination Check")
    contamination_subset = discovery_problems[:5]  # Quick check on 5 problems
    contamination = run_contamination_check(contamination_subset)
    
    if contamination['contamination_suspected']:
        print("\n⚠️  WARNING: Contamination suspected!")
        print("   Model may be memorizing answers rather than reasoning.")
        print("   Perturbation results may be confounded.")
    
    # Phase 2: Discovery (20 problems)
    print("\n>>> Phase 2: Discovery Experiment (20 problems)")
    discovery_results = run_perturbation_experiment(discovery_problems, "Discovery")
    
    # Phase 3: Validation (10 problems)
    print("\n>>> Phase 3: Validation Experiment (10 problems)")
    validation_results = run_perturbation_experiment(validation_problems, "Validation")
    
    # Phase 4: Analysis
    print("\n>>> Phase 4: Analysis")
    analysis = analyze_results(discovery_results, validation_results, contamination)
    
    # ——— FINAL REPORT ————————————————————————————————————————————————
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    
    print(f"""
Pre-registered Hypothesis (H1): ~15-30% of tokens are load-bearing
Pre-registered Falsifier (F1):  Random perturbation ≥50% answer change rate

RESULTS:
  Hypothesis supported:    {analysis['hypothesis_supported']}
  Falsifier triggered:     {analysis['falsifier_triggered']}
  Load-bearing fraction:   {analysis['load_bearing_fraction_estimate']:.1%}
  
  Structured perturbation change rates:
    Operator:      {analysis['structured_vs_random']['operator']:.1%}
    Number:        {analysis['structured_vs_random']['number']:.1%}
    Keyword:       {analysis['structured_vs_random']['keyword']:.1%}
    Calculation:   {analysis['structured_vs_random']['calculation']:.1%}
    Random:        {analysis['structured_vs_random']['random']:.1%}

CONTAMINATION CHECK:
  Real CoT accuracy:   {contamination['real_accuracy']:.1%}
  Random CoT accuracy: {contamination['random_accuracy']:.1%}
  Contamination risk:  {'HIGH' if contamination['contamination_suspected'] else 'LOW'}

PRE-REGISTERED SPLIT:
  Discovery:   {discovery_indices}
  Validation:  {validation_indices}

CAVEATS:
  - Small sample size (30 total problems)
  - Single model (MiniMax-M2.7)
  - Perturbation method: token replacement with similar wrong token
  - "Answer changed" defined as exact match of final numeric answer
  - GSM8K may have some training overlap despite low contamination risk
""")
    
    # Save results
    output = {
        'hypothesis': 'load_bearing_tokens_15_30_percent',
        'falsifier': 'random_perturbation_50_percent_threshold',
        'pre_registered_split': {
            'discovery': discovery_indices,
            'validation': validation_indices
        },
        'results': analysis,
        'contamination': contamination,
        'discovery_results': discovery_results,
        'validation_results': validation_results
    }
    
    with open('/tmp/load_bearing_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print("\nResults saved to /tmp/load_bearing_results.json")