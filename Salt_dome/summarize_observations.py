import os, csv, math
from collections import defaultdict

IN = "pyro_summary.csv"
OUT = "observation_sweep_summary.csv"
if not os.path.exists(IN):
    raise SystemExit(f"Missing {IN}")

def fnum(s):
    try: return float(s)
    except Exception: return math.nan

# --- read CSV robustly (row-oriented OR key/value index style) ---
with open(IN, newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames or []
    data = list(reader)

def looks_like_index_csv(fields):
    if not fields: return False
    norm = [ (c or "").strip().lower() for c in fields ]
    return (len(norm) == 2) and (norm[1] in ("0","value")) and (norm[0] in ("","index","key"))

if looks_like_index_csv(fields):
    key_col, val_col = fields[0], fields[1]
    one = {}
    for r in data:
        k = (r.get(key_col) or "").strip()
        v = r.get(val_col)
        if k: one[k] = v
    rows = [one]
else:
    rows = data

if not rows:
    raise SystemExit("Empty pyro_summary.csv after parsing")

# numeric keys present
num_keys = [k for k in rows[0]
            if k not in ("obs_coords","obs_count")
            and any(r.get(k) not in ("",None) for r in rows)]
num_keys = [k for k in num_keys if all(not isinstance(r.get(k),(list,dict)) for r in rows)]

# group by obs_count + obs_coords
groups = defaultdict(list)
for r in rows:
    try: n = int(float(r.get("obs_count","-1")))
    except Exception: n = -1
    sig = r.get("obs_coords","NA")
    groups[(n,sig)].append(r)

def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs)/len(xs) if xs else math.nan

def std(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if len(xs) < 2: return 0.0 if xs else math.nan
    m = sum(xs)/len(xs)
    return (sum((x-m)**2 for x in xs)/(len(xs)-1))**0.5

out = []
for (n, sig), rs in groups.items():
    row = {"obs_count": n, "n_runs": len(rs), "obs_coords": sig}
    for k in num_keys:
        vals = [fnum(r.get(k,"")) for r in rs]
        row[f"{k}_mean"] = mean(vals)
        row[f"{k}_std"]  = std(vals)
    out.append(row)

out.sort(key=lambda d: (d.get("obs_count",-1), d.get("elbo_final_mean", float("inf"))))

base = ["obs_count","n_runs"]
rest = sorted([k for k in out[0] if k not in ("obs_count","n_runs","obs_coords")])
fields = base + rest + ["obs_coords"]

with open(OUT,"w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(out)

print(f"Wrote {OUT} with {len(out)} rows.")
