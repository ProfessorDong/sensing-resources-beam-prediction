"""
Audit S2S-Diff_refs.bib against SparseDiffusion_reference_verification.txt.
Robust parser using brace-counting to handle nested braces in titles/abstracts.
"""
import re, os, sys
from difflib import SequenceMatcher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIB = os.path.join(ROOT, "S2S-Diff_refs.bib")
VER = os.path.join(ROOT, "SparseDiffusion_reference_verification.txt")
TEX = os.path.join(ROOT, "S2S-Diff.tex")

def normalize(s):
    s = s.lower()
    s = re.sub(r"[{}\\\"']+", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sim(a, b):
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()

def parse_bib(text):
    """Return list of {key, type, body, title, year} dicts using brace counting."""
    entries = []
    i = 0
    while i < len(text):
        # Find next @
        m = re.search(r"@(\w+)\s*\{\s*([^,\s}]+)\s*,", text[i:])
        if not m: break
        entry_start = i + m.start()
        body_start = i + m.end()
        # Find matching closing brace
        depth = 1
        j = body_start
        while j < len(text) and depth > 0:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        body = text[body_start:j-1]
        # Extract title (handles nested braces in title)
        tmatch = re.search(r"title\s*=\s*\{", body, re.IGNORECASE)
        title = ""
        if tmatch:
            ti = tmatch.end()
            d = 1
            tj = ti
            while tj < len(body) and d > 0:
                if body[tj] == '{':
                    d += 1
                elif body[tj] == '}':
                    d -= 1
                tj += 1
            title = body[ti:tj-1]
            title = re.sub(r"\{|\}", "", title)
            title = re.sub(r"\s+", " ", title).strip()
        # Extract year
        ymatch = re.search(r"year\s*=\s*\{?(\d{4})\}?", body, re.IGNORECASE)
        year = ymatch.group(1) if ymatch else ""
        # Extract authors (first author surname)
        amatch = re.search(r"author\s*=\s*\{([^}]*)\}", body, re.IGNORECASE)
        first_author = ""
        if amatch:
            first_author = amatch.group(1).split(",")[0].split(" and ")[0].strip()
            # If "Last, First" format, take Last
            if "," in first_author and " " not in first_author.split(",")[0]:
                first_author = first_author.split(",")[0]
            else:
                # "First Last" - take last token
                tokens = first_author.split()
                if tokens:
                    first_author = tokens[-1]
        entries.append({
            "key": m.group(2).strip(),
            "type": m.group(1),
            "title": title,
            "year": year,
            "first_author": first_author,
            "body": body,
        })
        i = j
    return entries

bib_entries = parse_bib(open(BIB).read())
ver_entries = parse_bib(open(VER).read())
tex_text = open(TEX).read()

print(f"Bib entries parsed: {len(bib_entries)}")
print(f"Verified entries parsed: {len(ver_entries)}")
print()

# Match each bib entry to best verified entry
matched = []
unmatched = []
for be in bib_entries:
    best = None
    best_score = 0.0
    for ve in ver_entries:
        s = sim(be["title"], ve["title"])
        if s > best_score:
            best_score = s
            best = ve
    if best_score >= 0.78:
        matched.append((be, best, best_score))
    else:
        unmatched.append((be, best, best_score))

print("=" * 78)
print(f"MATCHED ({len(matched)}): bib entries that align with the verification file")
print("=" * 78)
for be, ve, s in matched:
    print(f"  [{s:.2f}]  {be['key']:30s}")
    print(f"            bib: {be['title'][:80]}")
    print(f"            ver: {ve['title'][:80]}")
print()

print("=" * 78)
print(f"UNMATCHED ({len(unmatched)}): bib entries with NO corresponding verified entry")
print("=" * 78)
for be, best, s in unmatched:
    print(f"  [{s:.2f}]  {be['key']:30s}  ({be['first_author']}, {be['year']})")
    print(f"            bib title: {be['title'][:100]}")
    if best and s > 0.4:
        print(f"            closest ver: {best['title'][:80]} [{best['key']}]")
    print()

# Verified entries not matched (potentially missing from .bib)
print("=" * 78)
print("VERIFIED ENTRIES with NO match in .bib (potentially uncited additions to consider)")
print("=" * 78)
matched_ver_keys = set(ve["key"] for _, ve, _ in matched)
for ve in ver_entries:
    if ve["key"] not in matched_ver_keys:
        print(f"  {ve['key']:32s}  ({ve['first_author']}, {ve['year']})")
        print(f"            {ve['title'][:100]}")
        print()

# Citation usage
print("=" * 78)
print("CITATION USAGE in S2S-Diff.tex")
print("=" * 78)
cited = set()
for m in re.finditer(r"\\cite\{([^}]+)\}", tex_text):
    for k in m.group(1).split(","):
        cited.add(k.strip())
bib_keys = set(be["key"] for be in bib_entries)
missing_bib = cited - bib_keys
unused_bib = bib_keys - cited
print(f"  Cited in .tex: {len(cited)}")
print(f"  Defined in .bib: {len(bib_keys)}")
if missing_bib:
    print(f"  CITED BUT NOT IN .bib (broken!): {sorted(missing_bib)}")
if unused_bib:
    print(f"  IN .bib BUT NOT CITED:           {sorted(unused_bib)}")
