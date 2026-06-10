import json
import random

def categorize_logic(item):
    ans = str(item.get("answer", "")).strip().lower()
    exp = str(item.get("explanation", "")).strip().lower()
    if ans == "yes":
        return "yes"
    elif ans == "no":
        return "no"
    elif ans == "uncertain" or "uncertain" in ans:
        return "uncertain"
    elif ans in ["a", "b", "c", "d"] or "option a" in exp or "option b" in exp or "option c" in exp or "option d" in exp:
        return "choice"
    else:
        return "suy_luan"

def is_valid_unit(u):
    if not u:
        return False
    u = str(u).strip()
    # Reject empty, strings of only hyphens, or 'none'/'null'
    if not u or set(u) == {"-"} or u.lower() in ["none", "null"]:
        return False
    return True

def categorize_physics(item):
    if not is_valid_unit(item.get("unit")):
        return "INVALID_UNIT"
        
    qid = str(item.get("query_id", "")).strip().upper()
    if qid.startswith("CH"):
        return "CH"
    elif qid.startswith("DDT"):
        return "DDT"
    elif qid.startswith("DT") or qid.startswith("LD"):
        return "DT_LD"
    elif qid.startswith("NL"):
        return "NL"
    elif qid.startswith("TD"):
        return "TD"
    elif qid.startswith("THCB"):
        return "THCB"
    else:
        return "OTHER"

def main():
    with open("data/logic_output.json", "r", encoding="utf-8") as f:
        logic_data = json.load(f)
        
    with open("data/physics_output.json", "r", encoding="utf-8") as f:
        physics_data = json.load(f)

    logic_categorized = {"yes": [], "no": [], "uncertain": [], "choice": [], "suy_luan": []}
    for item in logic_data:
        cat = categorize_logic(item)
        if cat in logic_categorized:
            logic_categorized[cat].append(item)
            
    print("Logic categories count:")
    for k, v in logic_categorized.items():
        print(f"  {k}: {len(v)}")

    physics_categorized = {"CH": [], "DDT": [], "DT_LD": [], "NL": [], "TD": [], "THCB": [], "OTHER": []}
    for item in physics_data:
        cat = categorize_physics(item)
        if cat in physics_categorized:
            physics_categorized[cat].append(item)

    print("Physics categories count:")
    for k, v in physics_categorized.items():
        print(f"  {k}: {len(v)}")
        
    # Sampling Physics: DT_LD=7, CH=4, DDT=4, NL=4, TD=3, THCB=3
    phys_sample = []
    phys_sample.extend(random.sample(physics_categorized["DT_LD"], min(7, len(physics_categorized["DT_LD"]))))
    phys_sample.extend(random.sample(physics_categorized["CH"], min(4, len(physics_categorized["CH"]))))
    phys_sample.extend(random.sample(physics_categorized["DDT"], min(4, len(physics_categorized["DDT"]))))
    phys_sample.extend(random.sample(physics_categorized["NL"], min(4, len(physics_categorized["NL"]))))
    phys_sample.extend(random.sample(physics_categorized["TD"], min(3, len(physics_categorized["TD"]))))
    phys_sample.extend(random.sample(physics_categorized["THCB"], min(3, len(physics_categorized["THCB"]))))
    
    # Fill remaining if any category is short
    needed = 25 - len(phys_sample)
    if needed > 0:
        flat_rem = [item for k, v in physics_categorized.items() for item in v if item not in phys_sample]
        phys_sample.extend(random.sample(flat_rem, needed))
        
    # Sampling Logic: yes=5, no=5, uncertain=5, choice=5, suy_luan=5
    logic_sample = []
    for k in ["yes", "no", "uncertain", "choice", "suy_luan"]:
        avail = logic_categorized[k]
        take = min(5, len(avail))
        sampled = random.sample(avail, take)
        logic_sample.extend(sampled)
        
    needed_log = 25 - len(logic_sample)
    if needed_log > 0:
        flat_rem = [item for k, v in logic_categorized.items() for item in v if item not in logic_sample]
        logic_sample.extend(random.sample(flat_rem, needed_log))
        
    final_output = logic_sample + phys_sample
    random.shuffle(final_output)
    
    with open("data/test_50.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    print(f"Written {len(final_output)} items to data/test_50.json")

    # Generate the corresponding input file
    with open("data/logic_input.json", "r", encoding="utf-8") as f:
        logic_input_data = json.load(f)
    with open("data/physics_input.json", "r", encoding="utf-8") as f:
        physics_input_data = json.load(f)

    # Combine input data into a dictionary for fast lookup
    input_lookup = {}
    for item in logic_input_data + physics_input_data:
        qid = item.get("query_id")
        if qid:
            input_lookup[qid] = item

    final_input = []
    for out_item in final_output:
        qid = out_item.get("query_id")
        if qid in input_lookup:
            final_input.append(input_lookup[qid])
        else:
            print(f"Warning: Could not find input for query_id {qid}")

    with open("data/test_50_input.json", "w", encoding="utf-8") as f:
        json.dump(final_input, f, indent=2, ensure_ascii=False)

    print(f"Written {len(final_input)} items to data/test_50_input.json")

if __name__ == "__main__":
    main()
