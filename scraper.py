"""Download Doppler / Gamma Doppler pattern renders from the csgoskins.gg CDN.

The site serves each pattern render from a predictable CDN URL, so there is no
page scraping, no Selenium — just direct image requests:

    <CDN_BASE>/<knife>-<finish>-<variant>/<seed>.png
    e.g. m9-bayonet-doppler-phase-3/42.png

For every knife × finish × variant in config, this fetches the same set of
random pattern seeds and lays them out as:

    data/<knife>/<finish>/<variant>/seed_XXXX.png

which is exactly the layout data.py expects. Running the classifier's task on
m9-bayonet therefore needs nothing more than running this first.

    python scraper.py                 # download everything in config
    python scraper.py --knife m9-bayonet   # just one knife

Already-downloaded files are skipped, so an interrupted run is safe to repeat.
Not every finish exists for every knife; missing ones return 404 and are
counted as failures without stopping the run.
"""

import argparse
import random
import time
from pathlib import Path

import requests

import config as cfg

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://csgoskins.gg/",
}


def pick_seeds(n=cfg.SCRAPE_N_SEEDS, hi=cfg.SCRAPE_SEED_MAX, seed=cfg.SEED):
    """The same seed list is shared across every variant, so all skins line up."""
    rng = random.Random(seed)
    return sorted(rng.sample(range(1, hi + 1), n))


def build_jobs(knives, variants, data_dir):
    """Every (output_folder, cdn_slug) to fetch.

    slug  = "<knife>-<finish>-<variant>"      -> the CDN path
    folder = data_dir/<knife>/<finish>/<variant>  -> where images land
    """
    jobs = []
    for knife in knives:
        for finish, variant_list in variants.items():
            for variant in variant_list:
                slug = f"{knife}-{finish}-{variant}"
                folder = Path(data_dir) / knife / finish / variant
                jobs.append((folder, slug))
    return jobs


def download_variant(folder, slug, seeds, delay=cfg.SCRAPE_DELAY):
    """Fetch all seeds for one variant. Returns (ok, skipped, failed)."""
    folder.mkdir(parents=True, exist_ok=True)
    ok = skipped = failed = 0

    for seed in seeds:
        dest = folder / f"seed_{seed:04d}.png"
        if dest.exists():
            skipped += 1
            continue

        url = f"{cfg.CDN_BASE}/{slug}/{seed}.png"
        try:
            r = requests.get(url, timeout=10, headers=HEADERS)
            ct = r.headers.get("content-type", "")
            # a real image is 200, an image content-type, and not a tiny error blob
            if r.status_code == 200 and "image" in ct and len(r.content) > 1000:
                dest.write_bytes(r.content)
                ok += 1
            else:
                failed += 1   # 404 = that finish doesn't exist for this knife
        except Exception:
            failed += 1

        time.sleep(delay)

    return ok, skipped, failed


def scrape(knives=None, variants=None, data_dir=None, n_seeds=cfg.SCRAPE_N_SEEDS):
    """Download every configured knife/finish/variant. Returns total saved."""
    knives = knives or cfg.SCRAPE_KNIVES
    variants = variants or cfg.SCRAPE_VARIANTS
    data_dir = data_dir or cfg.DATA_DIR

    seeds = pick_seeds(n_seeds)
    jobs = build_jobs(knives, variants, data_dir)

    est_min = len(jobs) * n_seeds * cfg.SCRAPE_DELAY / 60
    print("=" * 60)
    print(f"  Knives:   {len(knives)}")
    print(f"  Variants: {len(jobs)}")
    print(f"  Seeds:    {n_seeds} per variant")
    print(f"  Images:   ~{len(jobs) * n_seeds:,} max")
    print(f"  Est time: ~{est_min:.0f} min ({est_min / 60:.1f} h)")
    print(f"  Output:   {data_dir}/<knife>/<finish>/<variant>/")
    print("=" * 60)

    total_ok = 0
    for i, (folder, slug) in enumerate(jobs, 1):
        ok, skip, fail = download_variant(folder, slug, seeds)
        total_ok += ok
        print(f"  [{i:3d}/{len(jobs)}]  ok {ok:3d}  skip {skip:3d}  fail {fail:3d}  {slug}")

    print(f"\n  Total saved: {total_ok} images")
    return total_ok



ap = argparse.ArgumentParser(description="Download Doppler pattern renders.")
ap.add_argument("--knife", help="only this knife slug (e.g. m9-bayonet)")
ap.add_argument("--seeds", type=int, default=cfg.SCRAPE_N_SEEDS,
                    help="pattern seeds per variant")
args = ap.parse_args()

knife_list = [args.knife] if args.knife else None
scrape(knives=knife_list, n_seeds=args.seeds)
