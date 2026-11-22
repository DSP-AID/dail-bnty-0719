"""Zero-shot transfer evaluation of the trained HVAC policy (bnty_0719).

The SAME network checkpoint is deployed, with no retraining or fine-tuning,
to held-out buildings and unseen weather. A mid-range-setpoint constant
baseline runs on the same episodes for reference.

Usage:
    python evaluate.py --model artifacts/sac_transfer.zip \
        --out artifacts/transfer_results.json
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys

faulthandler.enable()

import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CONSTRAINT_MID, TransferInterface,  # noqa: E402
                    evaluate_episode)

# Held-out evaluation grid. "source" envs reuse a trained building under new
# weather; the rest are buildings never seen during training.
EVAL_ENVS = [
    # unseen weather on source buildings (weather-only transfer)
    ("Eplus-5zone-hot-continuous-v1", "unseen weather"),
    ("Eplus-5zone-cool-continuous-v1", "unseen weather"),
    ("Eplus-office-hot-continuous-v1", "unseen weather"),
    ("Eplus-office-cool-continuous-v1", "unseen weather"),
    # unseen buildings, mixed climate
    ("Eplus-shop-mixed-continuous-v1", "unseen building"),
    ("Eplus-warehouse-mixed-continuous-v1", "unseen building"),
    ("Eplus-datacenter_cw-mixed-continuous-v1", "unseen building"),
    # unseen buildings AND unseen weather
    ("Eplus-shop-hot-continuous-v1", "unseen building+weather"),
    ("Eplus-shop-cool-continuous-v1", "unseen building+weather"),
    ("Eplus-warehouse-hot-continuous-v1", "unseen building+weather"),
    ("Eplus-warehouse-cool-continuous-v1", "unseen building+weather"),
    ("Eplus-datacenter_cw-hot-continuous-v1", "unseen building+weather"),
    ("Eplus-datacenter_cw-cool-continuous-v1", "unseen building+weather"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--model", default=os.path.join(here, "artifacts",
                                                    "sac_transfer.zip"))
    ap.add_argument("--out", default=os.path.join(here, "artifacts",
                                                  "transfer_results.json"))
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--max-envs", type=int, default=len(EVAL_ENVS),
                    help="smoke-test helper: evaluate only the first N envs")
    ap.add_argument("--envs", default="",
                    help="comma-separated env ids (parallel workers); "
                         "default: the full grid")
    args = ap.parse_args()

    model = SAC.load(args.model)

    def policy(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return action

    grid = EVAL_ENVS[: args.max_envs]
    if args.envs:
        by_id = {e: k for e, k in EVAL_ENVS}
        wanted = [e.strip() for e in args.envs.split(",") if e.strip()]
        unknown = [e for e in wanted if e not in by_id]
        if unknown:
            sys.exit(f"unknown env ids: {unknown}")
        grid = [(e, by_id[e]) for e in wanted]
    # Crash-resilient: episodes already present in --out (same model/seed) are
    # skipped, and the JSON is rewritten after every episode.
    done, rows = {}, []
    if os.path.exists(args.out):
        try:
            prev = json.load(open(args.out))
        except (OSError, ValueError):
            prev = {}
        if (prev.get("model_file") == os.path.basename(args.model)
                and prev.get("seed") == args.seed):
            done = {r["env_id"]: r for r in prev.get("episodes", [])}

    def dump() -> None:
        summary = {}
        for kind in sorted({r["transfer_kind"] for r in rows}):
            sel = [r for r in rows if r["transfer_kind"] == kind]
            summary[kind] = {
                "n_envs": len(sel),
                "mean_hvac_kwh_vs_baseline_pct": round(float(np.mean(
                    [r["hvac_kwh_vs_baseline_pct"] for r in sel])), 1),
                "mean_comfort_violation_rate": round(float(np.mean(
                    [r["policy"]["comfort_violation_rate"] for r in sel])), 4),
                "baseline_violation_rate": round(float(np.mean(
                    [r["baseline_const_midpoint"]["comfort_violation_rate"]
                     for r in sel])), 4),
            }
        out = {"model_file": os.path.basename(args.model), "seed": args.seed,
               "episodes": rows, "summary_by_transfer_kind": summary}
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        return summary

    for env_id, kind in grid:
        if env_id in done:
            print(f"[eval] {env_id:45s} cached", flush=True)
            rows.append(done[env_id])
            continue
        rl = evaluate_episode(policy, env_id, seed=args.seed)
        base = evaluate_episode(lambda o: CONSTRAINT_MID.copy(), env_id,
                                seed=args.seed)
        row = {
            "env_id": env_id, "transfer_kind": kind,
            "policy": rl, "baseline_const_midpoint": base,
            "hvac_kwh_vs_baseline_pct": round(
                100.0 * (rl["hvac_kwh_per_year"] / max(base["hvac_kwh_per_year"], 1e-9) - 1.0), 1),
        }
        rows.append(row)
        dump()
        print(f"[eval] {env_id:45s} kwh={rl['hvac_kwh_per_year']:10.1f} "
              f"(vs baseline {row['hvac_kwh_vs_baseline_pct']:+6.1f}%) "
              f"viol={rl['comfort_violation_rate']:.3f} "
              f"excess={rl['mean_degC_excess']:.3f}", flush=True)

    if len(rows) == len(grid):
        summary = dump()
        print("EVAL_DONE", json.dumps(summary), flush=True)
    else:
        print(f"EVAL_PARTIAL {len(rows)}/{len(grid)}", flush=True)
    sys.stdout.flush()
    os._exit(0)  # pyenergyplus teardown is segfault-prone


if __name__ == "__main__":
    main()
