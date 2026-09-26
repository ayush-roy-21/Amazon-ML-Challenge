"""Generate a small fake dataset with the competition's exact schema, purely so the pipeline in `src/` can
be run and sanity-checked end-to-end without the real competition data.

Every name/city/street word below is hand-written filler for building *fictitious* records - nothing here
is a real business, and nothing here is read by the matching pipeline itself (see `src/dev/__init__.py`).

Usage (run from the `business_entity_resolution/` directory, same as src.train / src.predict):
    python3 -m src.dev.make_synthetic_data --out-dir dataset --seed 42
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass, field
from typing import List, Tuple

# ----------------------------------------------------------------------------- fictitious vocabulary
NAME_STEM = {
    "us": ["Meridian", "Falcon", "Cascade", "Summit", "Beacon", "Liberty", "Harbor", "Maple", "Union",
           "Granite", "Silverline", "Crestview", "Ironwood", "Rivergate", "Pinecrest"],
    "india": ["Shree Ganesh", "Lakshmi", "Sundaram", "Krishna", "Annapurna", "Vishwakarma", "Shivam",
              "Sai Ram", "Bharat", "Navjyoti", "Mahalaxmi", "Om Sai", "Jai Bharat", "Ganga"],
    "france": ["Provence", "Chevalier", "Beaumont", "Lumiere", "Cascade", "Fontaine", "Bellevue",
               "Montagne", "Rivage", "Lefevre", "Girard", "Dupont"],
}
NAME_NOUN = ["Textiles", "Trading Co", "Motors", "Foods", "Logistics", "Solutions", "Enterprises",
             "Construction", "Industries", "Manufacturing", "Engineering", "Exports", "Technologies",
             "Apparel", "Chemicals", "Plastics", "Electricals", "Foundries", "Distributors", "Agro"]
LEGAL_SUFFIX = {"us": ["Inc", "LLC", "Corp", "Co"], "india": ["Pvt Ltd", "Ltd", "Enterprises"],
                "france": ["SARL", "SAS", "SA", "Etablissements"]}
LEGAL_ABBR_LONG = {"Inc": "Incorporated", "Corp": "Corporation", "Co": "Company", "Ltd": "Limited",
                    "Pvt Ltd": "Private Limited"}

STREET_TYPE_US = ["St", "Ave", "Blvd", "Dr", "Ln", "Ct", "Rd"]
STREET_TYPE_FR = ["Rue", "Avenue", "Boulevard", "Chemin", "Impasse"]
CITY = {"us": [("Springfield", "IL"), ("Franklin", "TN"), ("Georgetown", "TX"), ("Bristol", "CT"),
               ("Fairview", "OH"), ("Clinton", "IA"), ("Salem", "OR"), ("Madison", "WI")],
        "india": [("Surat", "Gujarat"), ("Nashik", "Maharashtra"), ("Bhopal", "Madhya Pradesh"),
                  ("Coimbatore", "Tamil Nadu"), ("Rajkot", "Gujarat"), ("Indore", "Madhya Pradesh"),
                  ("Nagpur", "Maharashtra"), ("Vadodara", "Gujarat")],
        "france": [("Lyon", "69000"), ("Nantes", "44000"), ("Rennes", "35000"), ("Reims", "51100"),
                   ("Dijon", "21000"), ("Angers", "49000"), ("Le Mans", "72000")]}
AREA_IN = ["Ring Road", "Gandhi Nagar", "Model Town", "Industrial Estate", "Civil Lines", "Station Road"]
LANDMARK = {"us": ["Near the old mill", "Behind city hall", "Opposite the mall"],
            "india": ["Near SBI ATM", "Opp. City Mall", "Behind Bus Stand", "Near Railway Station"],
            "france": ["Pres de la gare", "A cote de la mairie", "Face au marche"]}
FIRST_NAMES_IN = ["Shri", "Sri", "Sree"]  # transliteration variants applied to Indian names


@dataclass
class Rec:
    name: str
    address: str
    country: str


def _rng(seed):
    return random.Random(seed)


def _base_name_parts(r: random.Random, country: str) -> Tuple[str, str, str]:
    return r.choice(NAME_STEM[country]), r.choice(NAME_NOUN), r.choice(LEGAL_SUFFIX[country])


def _base_name(r: random.Random, country: str) -> str:
    return "{} {} {}".format(*_base_name_parts(r, country))


def _sibling_name(r: random.Random, stem: str, noun: str, country: str) -> str:
    """A different, unrelated business that happens to share the same generic stem+noun words (e.g. two
    unrelated "Meridian Logistics" companies) - a deliberate hard-negative: close enough in the name-only
    TF-IDF/fuzzy views to land in the same blocking candidate set, but a different real business that the
    address features (and, once retrieved, the model) must learn to reject."""
    return f"{stem} {noun} {r.choice(LEGAL_SUFFIX[country])}"


def _base_address(r: random.Random, country: str) -> str:
    house = r.randint(1, 999)
    if country == "us":
        street = f"{r.choice(NAME_STEM['us'])} {r.choice(STREET_TYPE_US)}"
        city, state = r.choice(CITY["us"])
        zip5 = r.randint(10000, 99999)
        return f"{house} {street}, {city}, {state} {zip5}"
    if country == "india":
        street = f"{r.choice(AREA_IN)}"
        area = r.choice(AREA_IN)
        city, state = r.choice(CITY["india"])
        pin = r.randint(100000, 999999)
        return f"Shop No. {house}, {street}, {area}, {city}, {state} {pin}"
    street = f"{r.choice(STREET_TYPE_FR)} {r.choice(NAME_STEM['france'])}"
    city, postal = r.choice(CITY["france"])
    return f"{house} {street}, {postal} {city}"


# ----------------------------------------------------------------------------- noise
def _typo(r: random.Random, tok: str) -> str:
    if len(tok) < 4:
        return tok
    i = r.randint(0, len(tok) - 2)
    op = r.choice(("swap", "drop", "dup"))
    if op == "swap":
        return tok[:i] + tok[i + 1] + tok[i] + tok[i + 2:]
    if op == "drop":
        return tok[:i] + tok[i + 1:]
    return tok[:i] + tok[i] + tok[i:]


def _maybe_typo(r: random.Random, text: str, p: float) -> str:
    toks = text.split()
    if not toks:
        return text
    if r.random() < p:
        i = r.randrange(len(toks))
        toks[i] = _typo(r, toks[i])
    return " ".join(toks)


def _noisy_name(r: random.Random, name: str, country: str) -> str:
    for full, abbr in LEGAL_ABBR_LONG.items():
        if r.random() < 0.35 and full in name:
            name = name.replace(full, abbr if r.random() < 0.5 else full.upper())
    if country == "india" and r.random() < 0.4:
        for a in ("Shree", "Shri", "Sri", "Sree"):
            if name.startswith(a + " "):
                name = r.choice(FIRST_NAMES_IN) + name[len(a):]
                break
    if r.random() < 0.25:
        name = name.replace("&", "and") if "&" in name else name.replace(" and ", " & ")
    toks = name.split()
    if len(toks) > 2 and r.random() < 0.15:
        i = r.randrange(len(toks) - 1)
        toks[i], toks[i + 1] = toks[i + 1], toks[i]
        name = " ".join(toks)
    name = _maybe_typo(r, name, 0.25)
    if r.random() < 0.15:
        name = name.replace(",", "").replace(".", "")
    return name


def _noisy_address(r: random.Random, addr: str, country: str) -> str:
    if r.random() < 0.3:
        addr = (addr.replace("Street", "St").replace("Road", "Rd").replace("Avenue", "Ave")
                    .replace("Boulevard", "Blvd"))
    if r.random() < 0.25:
        parts = [p.strip() for p in addr.split(",")]
        if len(parts) > 2:
            drop = r.randrange(1, len(parts) - 1)
            addr = ", ".join(p for i, p in enumerate(parts) if i != drop)
    if r.random() < 0.3:
        addr = f"{addr}, {r.choice(LANDMARK[country])}"
    if r.random() < 0.2:
        parts = [p.strip() for p in addr.split(",")]
        if len(parts) > 2:
            i, j = sorted(r.sample(range(len(parts) - 1), 2))
            parts[i], parts[j] = parts[j], parts[i]
            addr = ", ".join(parts)
    if country == "india" and r.random() < 0.2:
        addr = addr.replace("Shop No.", "Shop no").replace(",", "", 1)
    addr = _maybe_typo(r, addr, 0.2)
    return addr


# ----------------------------------------------------------------------------- assembly
def generate(n_entities: int, seed: int, countries: List[str], match_rate: float = 0.75,
             multi_rate: float = 0.25, distractor_rate: float = 0.4, sibling_rate: float = 0.3):
    """Returns (s1_rows, s2_rows, s3_rows, truth) where each *_rows is a list of (id, name, addr, country)
    and truth is {s1_id: [matched ids]}. `countries` controls which country pool entities are drawn from -
    pass ["us","india"] for the train split and ["us","india","france"] for test, per the spec (training
    covers US and India; the test set adds France). `sibling_rate` of entities also get a same-name,
    different-suffix, different-address "sibling" hard-negative planted in S2/S3 (see `_sibling_name`),
    so the candidate pools are not just true matches plus easily-dismissed unrelated distractors."""
    r = _rng(seed)
    s1, s2, s3, truth = [], [], [], {}
    s2_n = s3_n = 0
    for k in range(1, n_entities + 1):
        sid = f"S1-{k:05d}"
        country = r.choice(countries)
        stem, noun, suffix = _base_name_parts(r, country)
        name, addr = f"{stem} {noun} {suffix}", _base_address(r, country)
        s1.append((sid, name, addr, country))
        truth[sid] = []
        if r.random() < sibling_rate:
            sib_name = _sibling_name(r, stem, noun, country)
            sib_addr = _base_address(r, country)  # independently drawn: a genuinely different location
            if r.random() < 0.5:
                s2_n += 1
                s2.append((f"S2-{s2_n:05d}", sib_name, sib_addr, country))
            else:
                s3_n += 1
                s3.append((f"S3-{s3_n:05d}", sib_name, sib_addr, country))
        if r.random() >= match_rate:
            continue  # singleton: appears in S1 only
        n_matches = 2 if r.random() < multi_rate else 1
        for _ in range(n_matches):
            to_s3 = r.random() < 0.45
            nm = _noisy_name(r, name, country)
            ad = _noisy_address(r, addr, country)
            ctry_noisy = country if r.random() > 0.05 else ""  # occasionally missing
            if to_s3:
                s3_n += 1
                s3.append((f"S3-{s3_n:05d}", nm, ad, ctry_noisy))
                truth[sid].append(f"S3-{s3_n:05d}")
            else:
                s2_n += 1
                s2.append((f"S2-{s2_n:05d}", nm, ad, ctry_noisy))
                truth[sid].append(f"S2-{s2_n:05d}")
    # pure distractors: unrelated records with no S1 counterpart at all
    n_distract = int(n_entities * distractor_rate)
    for _ in range(n_distract):
        country = r.choice(countries)
        nm, ad = _base_name(r, country), _base_address(r, country)
        if r.random() < 0.5:
            s2_n += 1
            s2.append((f"S2-{s2_n:05d}", nm, ad, country))
        else:
            s3_n += 1
            s3.append((f"S3-{s3_n:05d}", nm, ad, country))
    r.shuffle(s2)
    r.shuffle(s3)
    return s1, s2, s3, truth


def _write_tsv(path: str, header: Tuple[str, ...], rows) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(str(x) for x in row) + "\n")


def _write_truth(path: str, truth) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for sid, ids in truth.items():
            fh.write(f"{sid}\t{','.join(ids)}\n")


def build(out_dir: str, n_train: int, n_test: int, seed: int) -> None:
    tr1, tr2, tr3, tr_truth = generate(n_train, seed, ["us", "india"])
    te1, te2, te3, te_truth = generate(n_test, seed + 1, ["us", "india", "france"])
    cols = ("entity_id", "business_name", "business_address", "country")
    _write_tsv(os.path.join(out_dir, "train", "train_source1.tsv"), cols, tr1)
    _write_tsv(os.path.join(out_dir, "train", "train_source2.tsv"), cols, tr2)
    _write_tsv(os.path.join(out_dir, "train", "train_source3.tsv"), cols, tr3)
    _write_truth(os.path.join(out_dir, "train", "train_ground_truth.tsv"), tr_truth)
    _write_tsv(os.path.join(out_dir, "test", "test_source1.tsv"), cols, te1)
    _write_tsv(os.path.join(out_dir, "test", "test_source2.tsv"), cols, te2)
    _write_tsv(os.path.join(out_dir, "test", "test_source3.tsv"), cols, te3)
    # dev-only oracle, kept OUTSIDE dataset/test/ and never read by predict.py - mirrors the fact that the
    # real competition's test labels are hidden on the leaderboard side.
    _write_truth(os.path.join(out_dir, "dev_only_test_ground_truth.tsv"), te_truth)
    print(f"train: S1={len(tr1)} S2={len(tr2)} S3={len(tr3)}  |  test: S1={len(te1)} S2={len(te2)} S3={len(te3)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="dataset")
    ap.add_argument("--n-train", type=int, default=2500, help="number of Source-1 entities in the train split")
    ap.add_argument("--n-test", type=int, default=1200, help="number of Source-1 entities in the test split")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args.out_dir, args.n_train, args.n_test, args.seed)


if __name__ == "__main__":
    main()
