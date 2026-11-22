# Transferable RL control for HVAC systems (DAiL bounty bnty_0719)

One reinforcement-learning agent is trained on **two source buildings** and
then deployed **zero-shot** — same network, no retraining, no fine-tuning — to
**unseen buildings and unseen weather** across a grid of 13 held-out
Sinergym/EnergyPlus environments.

## The idea: a building-agnostic interface

A policy only transfers if its inputs and outputs mean the same thing in every
building. `code/common.py` defines that interface on top of stock Sinergym 3.x
continuous envs:

* **Observation (16-D)** — semantic slots assembled by *name-pattern
  aggregation*, so it works for single- and multi-zone models alike:
  calendar encodings, outdoor weather (temp/RH/wind/solar), indoor air
  temperature mean/min/max across all zones, current heating/cooling
  setpoints, instantaneous HVAC power. Physical units, fixed scales — no
  per-building normalisation at deploy time.
* **Action** — 2-D vector in `[-1, 1]²` mapped onto each building's *own*
  setpoint bounds (relative position inside the allowed band), so one network
  drives a 5-zone house, an office, a shop, a warehouse or a data centre.
* **Scoring** — per-episode metrics use only the raw trajectory plus each
  env's *registered* comfort bands: integrated HVAC kWh (meter-free
  `HVAC_electricity_demand_rate`), worst-zone comfort-violation rate and mean
  °C outside the band.

## Held-out transfer grid

Trained on: `Eplus-5zone-mixed`, `Eplus-office-mixed`.
Evaluated zero-shot on unseen weather (hot/cool variants of both source
buildings), unseen buildings (shop, warehouse, two-zone data centre) and
unseen building **and** weather simultaneously. A constant mid-range-setpoint
baseline runs on the same episodes for reference.

## Reproducing

Requirements: Linux with EnergyPlus ≥ 24.1 (`energyplus` binary), Python 3.12,
then `pip install -r requirements.txt`. Point `EPLUS_PATH` at the EnergyPlus
installation folder. All building models (`*.epJSON`) and TMY3 weather files
(`*.epw`) actually used by these envs are bundled in `data/` (SHA-256 manifest:
`data/MANIFEST.json`); they are also shipped inside the `sinergym` package, so
no downloads are needed at run time.

```bash
export EPLUS_PATH=/path/to/EnergyPlus-24.1...   # folder containing energyplus + Energy+.idd
./run_all.sh                     # train (SAC, default 400k steps; override with TIMESTEPS=...) then evaluate all 13 envs
```

Training is checkpoint-resumable (`artifacts/last.zip` every 20k steps): the
EnergyPlus in-process restart at each annual episode boundary segfaults
occasionally under Python 3.12, so `run_all.sh` simply relaunches `train.py`
until the target step count is reached.

Individual steps:

```bash
python code/train.py    --timesteps 400000      # -> code/artifacts/sac_transfer.zip
python code/evaluate.py --model code/artifacts/sac_transfer.zip
                                                # -> code/artifacts/transfer_results.json
python code/common.py                            # interface self-check on two families
```

## Results (real runs, seed 1, SAC 400k steps, one annual episode per env)

Full per-episode numbers are in `code/artifacts/transfer_results.json`.
Means over each held-out group; `viol` = worst-zone comfort-violation rate of a
full year (policy / constant-midpoint baseline):

| Transfer kind | n | HVAC kWh vs baseline | viol policy | viol baseline |
|---|---|---|---|---|
| unseen weather | 4 | +79.0 % | **0.636** | 0.844 |
| unseen building | 3 | +26.5 % | **0.541** | 0.605 |
| unseen building + weather | 6 | +41.7 % | **0.548** | 0.630 |

* **Comfort transfers.** The zero-shot policy beats the constant-setpoint
  baseline on comfort violation rate in all 10 held-out envs where the
  baseline is not already perfect (biggest wins: office-cool −38.6 pts,
  shop-hot −33.2 pts). The other three are `datacenter_cw*`, where both are
  essentially violation-free (policy ≤ 0.009 vs baseline 0.000).
* **Energy does not come free.** The published env rewards weight comfort
  ~100× energy, so the policy buys comfort with kWh — most visibly on
  office-cool (+253 %) and shop-hot (+120 %). Under zero-shot transfer with
  no retraining this is the honest trade.

**Reward-weight ablation.** Re-training the identical setup with the energy
weight raised (`lambda_energy` 1e-4 → 1e-3) flips the story: unseen-building
energy drops to **−13.4 %** vs baseline, but comfort collapses on unseen
buildings (viol 0.825 vs baseline 0.605; on `datacenter_cw` the policy reaches
~0.8 while the baseline sits at 0.0). Both runs are reported; the shipped
model (`artifacts/sac_transfer.zip`) is the comfort-transfer one.

**Known limitation.** Actions are *relative* positions inside each env's own
setpoint band, which transfers poorly between buildings with very different
bands (e.g. shop heats to 10–22.5 °C and cools to 22.5–35 °C); absolute
setpoint deltas or a small domain-adaptation step would likely cut the energy
gap.

## Files

| Path | Purpose |
|---|---|
| `code/common.py` | transfer interface wrapper + scoring (`TransferInterface`, `score_episode`) |
| `code/train.py` | SAC training on the two source buildings |
| `code/evaluate.py` | zero-shot evaluation on the 13 held-out envs |
| `run_all.sh` | end-to-end reproduction |
| `data/` | bundled EnergyPlus models + weather (real public data) |
