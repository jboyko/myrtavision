"""Check the model's pool calls by eye.

    python tools/spot_check.py build              # on Great Lakes: writes results/spot_check.html
    python3 tools/spot_check.py score ~/Downloads/spot_check_answers.csv

build samples sheets from results/pool_predictions.csv in two strata: hosts
that supplied under 1% of the scored sheets (new to the model) and the rest,
the same number from each. Each sheet is shrunk to --side px and embedded, so
the page is one file that opens offline. On the page you mark each organ
present or absent before the model's calls are shown, so they can't anchor
you; answers are kept in the browser and exported as a CSV.

score compares that CSV (standard library only) with the calls: agreement, and false calls either
way, per organ and stratum. Unsure marks are left out. The two strata are
reported apart because the sample over-weights new hosts.
"""
import argparse
import base64
import csv
import io
import json
import random
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORGANS = ("bud", "flower", "fruit")
NEW_HOST_SHARE = 0.01


def strata(predictions, scores):
    trained = Counter(row["url"].split("/")[2] for row in csv.DictReader(scores.open()))
    total = sum(trained.values())
    return {row["image_id"]: "new host" if trained[row["host"]] / total < NEW_HOST_SHARE else "seen host"
            for row in predictions}


def thumbnail(path, side):
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as source:
        source.draft("RGB", (side, side))
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((side, side), Image.Resampling.LANCZOS)
    encoded = io.BytesIO()
    image.save(encoded, "JPEG", quality=80)
    return base64.b64encode(encoded.getvalue()).decode()


def build(args):
    predictions = list(csv.DictReader(args.predictions.open()))
    stratum = strata(predictions, args.scores)
    rng = random.Random(args.seed)
    chosen = []
    for name in ("new host", "seen host"):
        rows = [row for row in predictions if stratum[row["image_id"]] == name]
        chosen += rng.sample(rows, min(args.per_stratum, len(rows)))
    rng.shuffle(chosen)

    sheets = []
    for row in chosen:
        path = args.images / f"{row['image_id']}.jpg"
        try:
            image = thumbnail(path, args.side)
        except Exception as error:
            print(f"skipped {path}: {error}")
            continue
        sheets.append({
            "id": row["image_id"], "stratum": stratum[row["image_id"]], "image": image,
            "label": f"{row['genus']} {row['species'].split(' ', 1)[-1] if row['species'] else ''} · "
                     f"{row['institution']} · {row['host']} · {row['country']} {row['year']}",
            "calls": {organ: int(row[organ]) for organ in ORGANS},
            "probs": {organ: float(row[f"{organ}_prob"]) for organ in ORGANS},
        })
    page = PAGE.replace("__SHEETS__", json.dumps(sheets).replace("</", "<\\/"))
    args.out.write_text(page)
    print(f"wrote {len(sheets)} sheets ({Counter(s['stratum'] for s in sheets)}) to {args.out}, "
          f"{args.out.stat().st_size / 1e6:.0f} MB")


def score(args):
    answers = list(csv.DictReader(args.answers.open()))  # the export carries each sheet's stratum and model calls
    print(f"{len(answers)} sheets checked")
    for name in ("new host", "seen host", "all"):
        rows = [row for row in answers if name == "all" or row["stratum"] == name]
        if not rows:
            continue
        print(f"\n{name} ({len(rows)} sheets)")
        for organ in ORGANS:
            pairs = [(row[organ], row[f"model_{organ}"]) for row in rows if row[organ] in ("0", "1")]
            if not pairs:
                continue
            agree = sum(you == model for you, model in pairs)
            missed = sum(you == "1" and model == "0" for you, model in pairs)
            false = sum(you == "0" and model == "1" for you, model in pairs)
            present = sum(you == "1" for you, _ in pairs)
            print(f"  {organ:7} agree {agree}/{len(pairs)} ({agree / len(pairs):.0%})  "
                  f"model missed {missed} of {present} present  false calls {false}  "
                  f"unsure {len(rows) - len(pairs)}")


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pool Spot Check</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--ink:#1d1d1b;--muted:#6b6a65;--line:#dcdad3;--yes:#2f7d4f;--no:#a33b32;--unsure:#8a6d1f;--accent:#2b5c8a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--panel:#21211f;--ink:#ecebe6;--muted:#9c9b94;--line:#3a3935;--yes:#5fb784;--no:#e0776d;--unsure:#d4b45a;--accent:#7fb0e0}}
:root[data-theme="dark"]{--bg:#161615;--panel:#21211f;--ink:#ecebe6;--muted:#9c9b94;--line:#3a3935;--yes:#5fb784;--no:#e0776d;--unsure:#d4b45a;--accent:#7fb0e0}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.4 system-ui,sans-serif}
main{display:grid;grid-template-columns:1fr 320px;height:100vh}
#view{overflow:auto;background:#0e0e0d;cursor:zoom-in}#view.zoom{cursor:zoom-out}
#view img{display:block;max-height:100vh;max-width:100%;margin:auto}#view.zoom img{max-height:none;max-width:none}
aside{padding:16px;border-left:1px solid var(--line);background:var(--panel);overflow:auto}
h1{font-size:17px;margin:0 0 4px}.muted{color:var(--muted);font-size:13px}
.organ{margin:14px 0;padding:10px;border:1px solid var(--line);border-radius:8px}
.organ b{display:block;margin-bottom:6px;text-transform:capitalize}
.choices{display:flex;gap:6px}button{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink);cursor:pointer}
button.on[data-v="1"]{background:var(--yes);color:#fff;border-color:var(--yes)}button.on[data-v="0"]{background:var(--no);color:#fff;border-color:var(--no)}
button.on[data-v="?"]{background:var(--unsure);color:#fff;border-color:var(--unsure)}
.model{margin-top:6px;font-size:13px;color:var(--muted)}.model.agree::before{content:"✓ ";color:var(--yes)}.model.differ::before{content:"✗ ";color:var(--no)}
nav{display:flex;gap:6px;margin-top:12px}nav button{flex:1}.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
kbd{font-size:12px;border:1px solid var(--line);border-radius:4px;padding:0 4px}
@media (max-width:760px){main{grid-template-columns:1fr;height:auto}#view img{max-height:70vh}aside{border-left:0}}
</style></head><body><main>
<div id="view"><img id="sheet" alt="herbarium sheet"></div>
<aside>
<h1 id="count"></h1><div class="muted" id="label"></div>
<div id="organs"></div>
<nav><button id="prev">← Prev</button><button id="next" class="primary">Next →</button></nav>
<p class="muted">Click the sheet to zoom; scroll to pan. Keys: <kbd>q</kbd>/<kbd>a</kbd>/<kbd>z</kbd> bud yes/no/unsure,
<kbd>w</kbd>/<kbd>s</kbd>/<kbd>x</kbd> flower, <kbd>e</kbd>/<kbd>d</kbd>/<kbd>c</kbd> fruit, arrows to move.
The model's calls appear once all three are marked. Answers stay in this browser.</p>
<p class="muted" id="progress"></p>
<button id="export" class="primary">Export answers CSV</button>
</aside></main>
<script>
const SHEETS=__SHEETS__, ORGANS=["bud","flower","fruit"], KEY="myrta-spot-check";
let answers={}; try{answers=JSON.parse(localStorage.getItem(KEY))||{}}catch(e){}
let at=0;
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(answers))}catch(e){}};
const done=s=>ORGANS.every(o=>(answers[s.id]||{})[o]!==undefined);
function render(){
  const s=SHEETS[at], a=answers[s.id]||{};
  document.getElementById("sheet").src="data:image/jpeg;base64,"+s.image;
  document.getElementById("count").textContent=`Sheet ${at+1} of ${SHEETS.length}`;
  document.getElementById("label").textContent=s.label;
  const show=done(s);
  document.getElementById("organs").innerHTML=ORGANS.map(o=>{
    const v=a[o], m=s.calls[o];
    const btn=(val,txt)=>`<button data-o="${o}" data-v="${val}" class="${v===val?"on":""}">${txt}</button>`;
    const model=show?`<div class="model ${v==="?"?"":(String(m)===v?"agree":"differ")}">model: ${m?"present":"absent"} (${s.probs[o].toFixed(2)})</div>`:"";
    return `<div class="organ"><b>${o}</b><div class="choices">${btn("1","Present")}${btn("0","Absent")}${btn("?","Unsure")}</div>${model}</div>`}).join("");
  document.getElementById("progress").textContent=`${SHEETS.filter(done).length} of ${SHEETS.length} sheets marked`;
}
function mark(o,v){const id=SHEETS[at].id;answers[id]={...(answers[id]||{}),[o]:v};save();render()}
function move(d){at=Math.max(0,Math.min(SHEETS.length-1,at+d));document.getElementById("view").classList.remove("zoom");render();document.getElementById("view").scrollTo(0,0)}
document.getElementById("organs").onclick=e=>{const b=e.target.closest("button");if(b)mark(b.dataset.o,b.dataset.v)};
document.getElementById("prev").onclick=()=>move(-1);document.getElementById("next").onclick=()=>move(1);
document.getElementById("view").onclick=e=>e.currentTarget.classList.toggle("zoom");
const KEYS={q:["bud","1"],a:["bud","0"],z:["bud","?"],w:["flower","1"],s:["flower","0"],x:["flower","?"],e:["fruit","1"],d:["fruit","0"],c:["fruit","?"]};
document.onkeydown=e=>{if(KEYS[e.key])mark(...KEYS[e.key]);else if(e.key==="ArrowRight")move(1);else if(e.key==="ArrowLeft")move(-1)};
document.getElementById("export").onclick=()=>{
  const rows=[["image_id","stratum",...ORGANS,...ORGANS.map(o=>"model_"+o)]];
  for(const s of SHEETS){const a=answers[s.id];if(!a)continue;rows.push([s.id,s.stratum,...ORGANS.map(o=>a[o]??""),...ORGANS.map(o=>s.calls[o])])}
  const link=document.createElement("a");link.href=URL.createObjectURL(new Blob([rows.map(r=>r.join(",")).join("\\n")],{type:"text/csv"}));
  link.download="spot_check_answers.csv";link.click()};
render();
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "results/pool_predictions.csv")
    parser.add_argument("--scores", type=Path, default=PROJECT_ROOT / "scores.csv")
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build")
    b.add_argument("--images", type=Path, default=PROJECT_ROOT / "data/pool")
    b.add_argument("--out", type=Path, default=PROJECT_ROOT / "results/spot_check.html")
    b.add_argument("--per-stratum", type=int, default=50)
    b.add_argument("--side", type=int, default=2000, help="long side of the embedded sheets, px")
    b.add_argument("--seed", type=int, default=0)
    s = commands.add_parser("score")
    s.add_argument("answers", type=Path)
    args = parser.parse_args()
    build(args) if args.command == "build" else score(args)


if __name__ == "__main__":
    main()
