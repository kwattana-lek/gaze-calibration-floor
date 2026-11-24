"""Review item 11: unified protocol inventory, read from the record files themselves (read-only).
For each record file: rows, subjects, seeds, contamination rates, severities, corruption models, levels / channels.
Writes results/11_protocol_inventory.csv. Nothing here is typed by hand except the file list.
"""
import csv, json, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); REV = os.path.dirname(HERE); LN = os.path.dirname(REV)
FILES = ["exp_a_records.jsonl", "exp_a_records_eyediap.jsonl", "exp_seed_extension_records.jsonl", "exp_seed_extension_records_eyediap.jsonl",
         "exp_b_records.csv", "sigma_sweep_records.jsonl", "sigma_sweep_records_K.jsonl", "sigma_clean_ehat_records.jsonl", "sigma_coherent_records.jsonl",
         "resnet_frozen_records.jsonl", "unigaze_arm_records.jsonl", "eyediap_unigaze_arm_records.jsonl", "rtgene_unigaze_arm_records.jsonl",
         "real_lapse_records.jsonl", "exp_purification_records.jsonl",
         os.path.join("revision_2026-11", "results", "gate_transfer_scores.jsonl"), os.path.join("revision_2026-11", "results", "gate_transfer_mae.jsonl")]
KEYS = {"subject": ["subject", "sid", "participant"], "seed": ["seed"], "pc": ["pc"], "delta": ["delta", "severity", "delta_deg"],
        "model": ["model", "corruption", "kind"], "level": ["level", "sigma_level", "arm", "channel", "config"], "K": ["K", "k"], "steps": ["steps", "n_steps"]}


def rows_of(path):
    if path.endswith(".csv"):
        yield from csv.DictReader(open(path))
    else:
        for line in open(path):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def fmt(vals, n=12):
    vals = sorted(vals, key=lambda v: (str(type(v)), v))
    s = ", ".join(str(v) for v in vals[:n])
    return s + (f", ... ({len(vals)} values)" if len(vals) > n else "")


out = []
for f in FILES:
    p = os.path.join(LN, f)
    if not os.path.exists(p):
        out.append(dict(file=f, rows="missing")); continue
    seen = defaultdict(set); n = 0
    for r in rows_of(p):
        n += 1
        for k, alts in KEYS.items():
            for a in alts:
                if a in r and not isinstance(r[a], (list, dict)) and r[a] not in ("", None):
                    v = r[a]
                    try:
                        v = float(v); v = int(v) if v == int(v) else round(v, 3)
                    except (TypeError, ValueError):
                        pass
                    seen[k].add(v); break
    rec = dict(file=f, rows=n, n_subjects=len(seen["subject"]))
    for k in ("seed", "pc", "delta", "model", "level", "K", "steps"):
        rec[k] = fmt(seen[k]) if seen[k] else ""
    out.append(rec)
fields = ["file", "rows", "n_subjects", "seed", "pc", "delta", "model", "level", "K", "steps"]
with open(os.path.join(REV, "results", "11_protocol_inventory.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(out)
for r in out:
    print(r)
