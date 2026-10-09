"""
How many clean 3D models may we train on (Phase 9)? TexVerse's metadata (858,669 Sketchfab models with high-resolution
textures: per-object licence, PBR, categories, faces, texture size) joined on the Sketchfab uid with Objaverse++'s
quality annotations (789,195 Objaverse objects: a 0-3 quality score, style, and scene / multi-object / transparent /
single-colour flags). Only metadata is downloaded, no models.

    python3 ops/data_audit.py ops-out/private/data

Writes <out>/summary.json (counts) and <out>/private/ uid lists of the subsets (the models themselves stay on
Sketchfab and Hugging Face; a training run downloads them by uid). Runs anywhere with network access to Hugging Face
(the relay's GitHub runner), CPU only.
"""

from __future__ import annotations

import collections
import io
import json
import pathlib
import sys
import time
import zipfile

from huggingface_hub import HfApi, hf_hub_download

TEXVERSE = "YiboZhang2001/TexVerse"
OBJAVERSE_PP = "cindyxl/ObjaversePlusPlus"
# Licences a commercial model may be trained on: attribution (credited in the model's notice) or public domain.
# ShareAlike and NoDerivatives are kept apart for the attorney review; NonCommercial never.
ALLOWED = {"cc attribution", "cc0 public domain", "cc by", "cc0", "cc-by", "cc-by-4.0", "cc0-1.0", "by"}
REVIEW = {"cc attribution-sharealike", "cc attribution-noderivs", "cc by-sa", "cc by-nd", "by-sa", "by-nd"}
# What creators ask for (workers/test-sets/benchmark.txt), as Sketchfab's category names
PRODUCT_CATEGORIES = {
    "cars & vehicles", "electronics & gadgets", "fashion & style", "food & drink", "furniture & home",
    "sports & fitness", "science & technology",
}
CHARACTER_CATEGORIES = {"characters & creatures", "animals & pets"}
STEP = time.time()


def log(message: str) -> None:
    print(f"[data {time.time() - STEP:6.0f}s] {message}", flush=True)


def top_level(repo: str) -> list[tuple[str, int | None]]:
    api = HfApi()
    entries = api.list_repo_tree(repo, repo_type="dataset", recursive=False)
    return [(entry.path, getattr(entry, "size", None)) for entry in entries]


def find(files: list[tuple[str, int | None]], *names: str) -> str | None:
    lowered = {path.lower(): path for path, _ in files}
    for name in names:
        for path_lower, path in lowered.items():
            if path_lower.endswith(name):
                return path
    return None


def objects(path: str):
    """(uid, entry) pairs from a JSON object keyed by uid or a JSON list of entries, streamed if it is large."""
    import ijson

    with open(path, "rb") as handle:
        head = handle.read(1 << 16).lstrip()
    first = head[:1]
    with open(path, "rb") as handle:
        if first == b"{":
            for uid, entry in ijson.kvitems(handle, "", use_float=True):
                yield uid, entry
        elif first == b"[":
            for entry in ijson.items(handle, "item", use_float=True):
                uid = entry.get("uid") or entry.get("UID") or entry.get("id") if isinstance(entry, dict) else None
                yield uid, entry
        else:
            raise SystemExit(f"{path}: neither a JSON object nor a list")


def licence_of(entry: dict) -> str:
    value = entry.get("license") or entry.get("licence") or ""
    if isinstance(value, dict):
        value = value.get("label") or value.get("name") or value.get("slug") or ""
    return str(value).strip().lower()


def categories_of(entry: dict) -> list[str]:
    value = entry.get("categories") or []
    names = []
    for item in value if isinstance(value, list) else [value]:
        if isinstance(item, dict):
            item = item.get("name") or item.get("slug") or ""
        if item:
            names.append(str(item).strip().lower())
    return names


def truthy(value) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def main(out_dir: str) -> None:
    out = pathlib.Path(out_dir)
    (out / "private").mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    # --- Objaverse++: uid -> annotations --------------------------------------------------------------
    files = top_level(OBJAVERSE_PP)
    log(f"{OBJAVERSE_PP}: {files}")
    summary["objaverse_pp_files"] = files
    quality: dict[str, dict] = {}
    name = find(files, "annotated_800k.json", ".json", ".zip")
    if name:
        local = hf_hub_download(OBJAVERSE_PP, name, repo_type="dataset")
        if local.endswith(".zip"):
            with zipfile.ZipFile(local) as archive:
                member = next(m for m in archive.namelist() if m.endswith(".json"))
                extracted = out / "private" / "objaverse_pp.json"
                extracted.write_bytes(archive.read(member))
                local = str(extracted)
        for uid, entry in objects(local):
            if uid and isinstance(entry, dict):
                quality[str(uid)] = entry
    log(f"Objaverse++ annotations: {len(quality)}")
    if quality:
        sample = next(iter(quality.values()))
        summary["objaverse_pp_fields"] = sorted(sample)
        scores = collections.Counter(str(e.get("score")) for e in quality.values())
        summary["objaverse_pp_scores"] = dict(sorted(scores.items()))
        summary["objaverse_pp_styles"] = dict(collections.Counter(str(e.get("style")) for e in quality.values()).most_common())

    # --- TexVerse metadata: uid -> licence, PBR, categories, faces, texture size -----------------------
    files = top_level(TEXVERSE)
    log(f"{TEXVERSE}: {[(p, s) for p, s in files if not p.startswith(('glbs', 'thumbnails'))][:60]}")
    summary["texverse_files"] = [(p, s) for p, s in files][:200]
    name = find(files, "metadata.json")
    if not name:
        raise SystemExit("TexVerse's metadata.json wasn't found at the top level; see the file list above")
    local = hf_hub_download(TEXVERSE, name, repo_type="dataset")
    log(f"TexVerse metadata: {pathlib.Path(local).stat().st_size / 1e9:.2f} GB")

    licences = collections.Counter()
    tallies = collections.Counter()
    by_category = collections.Counter()
    by_category_clean = collections.Counter()
    faces = collections.Counter()
    textures = collections.Counter()
    first_entry = None
    lists = {name: [] for name in ("allowed", "allowed_pbr", "clean", "clean_products", "clean_characters", "review")}
    for uid, entry in objects(local):
        if not isinstance(entry, dict):
            continue
        if first_entry is None:
            first_entry = {k: (v if not isinstance(v, (list, dict)) else str(v)[:120]) for k, v in entry.items()}
        uid = str(uid or entry.get("uid") or "")
        tallies["models"] += 1
        licence = licence_of(entry)
        licences[licence] += 1
        if licence in REVIEW:
            lists["review"].append(uid)
        if licence not in ALLOWED:
            continue
        tallies["allowed"] += 1
        lists["allowed"].append(uid)
        pbr = entry.get("pbrType")
        has_pbr = bool(pbr) and str(pbr).lower() not in {"false", "none", "null", ""}
        if has_pbr:
            tallies["allowed_pbr"] += 1
            lists["allowed_pbr"].append(uid)
        cats = categories_of(entry)
        for cat in cats or ["(none)"]:
            by_category[cat] += 1
        count = entry.get("faceCount")
        try:
            count = int(count)
            bucket = ("<5k" if count < 5_000 else "5k-50k" if count < 50_000 else "50k-500k" if count < 500_000
                      else "500k-2M" if count < 2_000_000 else ">2M")
        except (TypeError, ValueError):
            bucket = "?"
        faces[bucket] += 1
        textures[str(entry.get("max_texture"))] += 1

        annotation = quality.get(uid)
        if annotation is None:
            continue
        tallies["allowed_annotated"] += 1
        try:
            score = int(annotation.get("score"))
        except (TypeError, ValueError):
            continue
        flags = {key: truthy(annotation.get(key)) for key in
                 ("is_scene", "is_multi_object", "is_single_color", "is_transparent", "is_figure")}
        if score >= 2 and not flags["is_scene"] and not flags["is_multi_object"] and not flags["is_single_color"]:
            tallies["clean"] += 1
            lists["clean"].append(uid)
            if has_pbr:
                tallies["clean_pbr"] += 1
            style = str(annotation.get("style"))
            tallies[f"clean_style_{style}"] += 1
            for cat in cats or ["(none)"]:
                by_category_clean[cat] += 1
            if set(cats) & PRODUCT_CATEGORIES:
                lists["clean_products"].append(uid)
            if set(cats) & CHARACTER_CATEGORIES:
                lists["clean_characters"].append(uid)
        if tallies["models"] % 100_000 == 0:
            log(f"{tallies['models']} models read")

    summary["texverse_first_entry"] = first_entry
    summary["tallies"] = dict(tallies)
    summary["licences"] = dict(licences.most_common())
    summary["allowed_by_category"] = dict(by_category.most_common(30))
    summary["clean_by_category"] = dict(by_category_clean.most_common(30))
    summary["allowed_faces"] = dict(faces)
    summary["allowed_max_texture"] = dict(textures.most_common(12))
    summary["lists"] = {name: len(uids) for name, uids in lists.items()}
    for name, uids in lists.items():
        (out / "private" / f"{name}.txt").write_text("\n".join(uids) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log(json.dumps({k: summary[k] for k in ("tallies", "lists", "licences")}, default=str))
    log(f"by category (allowed): {json.dumps(summary['allowed_by_category'])}")
    log(f"by category (clean): {json.dumps(summary['clean_by_category'])}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ops-out/private/data")
