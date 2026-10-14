"""DreamSim pole score for images (paper Appendix A.6): a judge-free complement to the concept rate.

Each generated image I is compared with two references of the same prompt, R+ and R- (the model's own unsteered
image edited toward each pole). With d the DreamSim distance (1 - cosine of the DreamSim embeddings),

    pole(I) = 100 * d(I, R-) / (d(I, R+) + d(I, R-))

so 100 means I matches the positive-pole reference, 0 the negative-pole one, 50 is equidistant.

    pole_scores(run, refs)   reads <run>/generations/sweep_image.csv and <refs>/<concept>/p<NN>_{pos,neg}.png
                             writes <run>/scores/dreamsim_scores.csv

Needs the `dreamsim` package (pip install dreamsim) and, for speed, a GPU.
"""

from __future__ import annotations

import csv
from pathlib import Path

COLUMNS = ["concept", "quadrant", "config", "alpha", "prompt_idx", "score", "baseline_score", "delta_score"]


def pole_score(e, e_pos, e_neg) -> float:
    """Pole score from unit-normalized embeddings (torch or numpy vectors)."""
    d_pos = 1.0 - float((e * e_pos).sum())
    d_neg = 1.0 - float((e * e_neg).sum())
    return 100.0 * d_neg / (d_pos + d_neg + 1e-8)


def embed(paths, device: str | None = None, batch_size: int = 128):
    """Unit-normalized DreamSim embeddings, one row per path."""
    import torch
    import torch.nn.functional as F
    from dreamsim import dreamsim
    from PIL import Image

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache = Path.home() / ".cache" / "dreamsim"                 # dreamsim's own default is ./models
    cache.mkdir(parents=True, exist_ok=True)
    model, preprocess = dreamsim(pretrained=True, device=device, cache_dir=str(cache))
    out = []
    with torch.no_grad():
        for k in range(0, len(paths), batch_size):
            x = torch.cat([preprocess(Image.open(p).convert("RGB")) for p in paths[k:k + batch_size]]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                e = model.embed(x)
            out.append(F.normalize(e.float(), dim=1).cpu())
    return torch.cat(out)


def pole_scores(run: str | Path, refs: str | Path, device: str | None = None, log=print) -> Path:
    run, refs = Path(run), Path(refs)
    gen = run / "generations"
    with open(gen / "sweep_image.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    ref_path = lambda c, pi, pole: refs / c / f"p{int(pi):02d}_{pole}.png"
    rows = [r for r in rows if ref_path(r["concept"], r["prompt_idx"], "pos").exists()
            and ref_path(r["concept"], r["prompt_idx"], "neg").exists()]
    if not rows:
        raise FileNotFoundError(f"no references under {refs} (expected <concept>/p<NN>_pos.png and _neg.png)")
    images = sorted({r["image_path"] for r in rows})
    keys = sorted({(r["concept"], int(r["prompt_idx"])) for r in rows})
    ref_files = [ref_path(c, pi, pole) for c, pi in keys for pole in ("pos", "neg")]
    log(f"[dreamsim] {len(images)} images, {len(ref_files)} references")
    emb = embed([gen / p for p in images] + ref_files, device)
    img = dict(zip(images, emb[:len(images)], strict=True))
    ref = {(c, pi, pole): e for (c, pi, pole), e in
           zip([(c, pi, pole) for c, pi in keys for pole in ("pos", "neg")], emb[len(images):], strict=True)}

    def score(r):
        c, pi = r["concept"], int(r["prompt_idx"])
        return pole_score(img[r["image_path"]], ref[(c, pi, "pos")], ref[(c, pi, "neg")])

    base = {(r["concept"], r["quadrant"], r["prompt_idx"]): score(r) for r in rows if float(r["alpha"]) == 0}
    out = run / "scores" / "dreamsim_scores.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in rows:
            if float(r["alpha"]) == 0:
                continue
            s, b = score(r), base.get((r["concept"], r["quadrant"], r["prompt_idx"]))
            w.writerow([r["concept"], r["quadrant"], r["config"], f"{float(r['alpha']):g}", r["prompt_idx"],
                        round(s, 3), "" if b is None else round(b, 3), "" if b is None else round(s - b, 3)])
    return out
