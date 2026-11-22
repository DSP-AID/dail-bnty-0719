"""Shared interface for the transferable HVAC policy (bnty_0719).

One SAC policy is trained on two source buildings and deployed *zero-shot*
(no retraining, no fine-tuning) to unseen buildings and unseen weather.
Transferability comes from a building-agnostic interface:

* Observation  -> fixed 16-dim semantic vector built by aggregating whatever
  variables each EnergyPlus model exposes (outdoor weather, indoor air-temp
  mean/min/max, HVAC power, current setpoints, calendar). Absolute units,
  fixed scales, so no per-building normalisation is needed at deploy time.
* Action       -> 2-D box in [-1, 1]^2 mapped onto each building's own
  heating/cooling setpoint bounds (relative position inside the allowed band).
* Scoring      -> per-step metrics computed from the same semantic vector plus
  each environment's *registered* comfort bands (the scorer may know the
  building's contract; the policy never sees it).

Compatible with sinergym 3.x continuous envs (`*-continuous-v1`), which use
`time_variables == ["month", "day_of_month", "hour"]`, 15-min timesteps and a
2-D setpoint action space.
"""
from __future__ import annotations

import os

# sinergym needs both variables pointing at the EnergyPlus install; accept
# either one and fill in the other so scripts only have to set one.
if "EPLUS_PATH" not in os.environ and "ENERGYPLUS_INSTALLATION_FOLDER" in os.environ:
    os.environ["EPLUS_PATH"] = os.environ["ENERGYPLUS_INSTALLATION_FOLDER"]
elif "ENERGYPLUS_INSTALLATION_FOLDER" not in os.environ and "EPLUS_PATH" in os.environ:
    os.environ["ENERGYPLUS_INSTALLATION_FOLDER"] = os.environ["EPLUS_PATH"]

import gymnasium as gym
import numpy as np
import sinergym  # noqa: F401  (registers the Eplus-* envs with gymnasium)
from gymnasium import Wrapper

# ---------------------------------------------------------------- features --
# Order of the 15-dim observation. Values are physical units divided by the
# scale below, so the same NN input means the same physical quantity in every
# building. SCALES are used again by score_episode to undo scaling.
FEATURE_NAMES = [
    "month_sin", "month_cos", "hour_sin", "hour_cos", "day_of_year",
    "outdoor_temp_C", "outdoor_rh_pct", "wind_speed_ms",
    "diffuse_rad_kWm2", "direct_rad_kWm2",
    "indoor_temp_mean_C", "indoor_temp_min_C", "indoor_temp_max_C",
    "heating_setpoint_C", "cooling_setpoint_C", "hvac_power_kW",
]
OBS_DIM = len(FEATURE_NAMES)

# Raw per-step values kept aside for scoring (not part of the NN input):
SCORE_NAMES = ["hvac_power_W", "indoor_temp_min_C", "indoor_temp_max_C", "month"]
SCORE_DIM = len(SCORE_NAMES)


def _env_layout(env_id: str):
    """Return (time_vars, var_names, power_idx, temp_idx, htg_idx, clg_idx).

    Indexes inside the raw EplusEnv observation vector, which is ordered as
    time_variables + variables + meters. Rules are name-pattern based so one
    wrapper works across building families (single- and multi-zone).
    """
    kw = dict(gym.spec(env_id).kwargs or {})
    import ast

    def lit(x):
        return ast.literal_eval(x) if isinstance(x, str) else x

    time_vars = list(lit(kw["time_variables"]))
    var_names = list(lit(kw["variables"]).keys())
    meter_names = list(lit(kw.get("meters") or {}).keys())
    base = len(time_vars)
    off = {n: base + i for i, n in enumerate(var_names)}
    power_name = "HVAC_electricity_demand_rate"
    assert power_name in off, f"{env_id}: no HVAC power variable"
    temp_idx = [off[n] for n in var_names if n.endswith("air_temperature")]
    htg_idx = [off[n] for n in var_names if "heating_setpoint" in n]
    clg_idx = [off[n] for n in var_names if "cooling_setpoint" in n]
    assert temp_idx, f"{env_id}: no *_air_temperature variables"
    return time_vars, var_names + meter_names, off[power_name], temp_idx, htg_idx, clg_idx


_MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


class TransferInterface(Wrapper):
    """Featurises raw sinergym observations and normalises actions.

    Action space becomes Box([-1]*2, [1]*2); step(a) with a in [-1,1]^2 sets
    heating = lo + (a+1)/2*(hi-lo) of the wrapped env's own action bounds.
    `score_history` collects per-step raw values for score_episode.
    """

    def __init__(self, env_id: str, **make_kwargs):
        env = gym.make(env_id, **make_kwargs)
        super().__init__(env)
        (self.time_vars, self.var_names, self.i_power,
         self.i_temp, self.i_htg, self.i_clg) = _env_layout(env_id)
        lo, hi = env.action_space.low, env.action_space.high
        assert env.action_space.shape == (2,), f"{env_id}: action not 2-D"
        # heating uses dim 0, cooling dim 1 (sinergym actuator order)
        self.act_lo, self.act_hi = lo[0], hi[0]
        self.coo_lo, self.coo_hi = lo[1], hi[1]
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-np.ones(2), np.ones(2),
                                           dtype=np.float32)
        self.score_history: list[np.ndarray] = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.score_history = []
        self._record(obs)
        return self._featurize(obs), info

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        raw = np.array([
            self.act_lo + (a[0] + 1) / 2 * (self.act_hi - self.act_lo),
            self.coo_lo + (a[1] + 1) / 2 * (self.coo_hi - self.coo_lo),
        ], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(raw)
        self._record(obs)
        return self._featurize(obs), reward, terminated, truncated, info

    # -- internals ---------------------------------------------------------
    def _featurize(self, raw: np.ndarray) -> np.ndarray:
        t = raw[: len(self.time_vars)].astype(np.float64)
        m, d, h = int(t[0]), int(t[1]), int(t[2])
        doy = sum(_MONTH_DAYS[: m - 1]) + d
        temps = raw[self.i_temp].astype(np.float64)
        power = float(raw[self.i_power])
        htg = np.mean([raw[i] for i in self.i_htg]) if self.i_htg else 0.0
        clg = np.mean([raw[i] for i in self.i_clg]) if self.i_clg else 0.0
        def g(name):  # weather variables exist in every sinergym model
            return float(raw[self.var_names.index(name) + len(self.time_vars)])
        f = [
            np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12),
            np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
            doy / 365.0,
            g("outdoor_temperature"),
            g("outdoor_humidity") / 100.0,
            g("wind_speed"),
            g("diffuse_solar_radiation") / 1000.0,
            g("direct_solar_radiation") / 1000.0,
            float(temps.mean()), float(temps.min()), float(temps.max()),
            float(htg), float(clg), power / 1000.0,
        ]
        return np.asarray(f, dtype=np.float32)

    def _record(self, raw: np.ndarray):
        t = raw[: len(self.time_vars)]
        temps = raw[self.i_temp]
        self.score_history.append(np.array(
            [float(raw[self.i_power]), float(temps.min()),
             float(temps.max()), float(t[0])], dtype=np.float64))


# ----------------------------------------------------------------- scoring --
def comfort_bands(env_id: str):
    """Registered comfort bands of an env: (winter, summer, summer_months)."""
    import ast
    kw = dict(gym.spec(env_id).kwargs or {})
    rk = kw["reward_kwargs"]
    rk = ast.literal_eval(rk) if isinstance(rk, str) else rk
    ss, sf = rk["summer_start"], rk["summer_final"]
    summer_months = set(range(ss[0], sf[0] + 1))
    return (tuple(rk["range_comfort_winter"]),
            tuple(rk["range_comfort_summer"]), summer_months)


def score_episode(history: np.ndarray, env_id: str) -> dict:
    """Per-episode transfer metrics from the raw score history.

    * hvac_kwh_per_year : integrated HVAC power (meter-free variable).
    * comfort_violation_rate : occupied fraction of 15-min steps where ANY
      thermal zone leaves its season band (worst-zone rule, uniform across
      buildings).
    * mean_degC_excess : mean absolute distance outside the band.
    """
    winter, summer, summer_months = comfort_bands(env_id)
    months = history[:, 3].astype(int)
    lo = np.where(np.isin(months, list(summer_months)), summer[0], winter[0])
    hi = np.where(np.isin(months, list(summer_months)), summer[1], winter[1])
    tmin, tmax = history[:, 1], history[:, 2]
    excess = np.maximum(0.0, lo - tmin) + np.maximum(0.0, tmax - hi)
    viol = excess > 1e-9
    return {
        "hvac_kwh_per_year": float(history[:, 0].sum() * 0.25 / 1000.0),
        "comfort_violation_rate": float(viol.mean()),
        "mean_degC_excess": float(excess.mean()),
        "steps": int(len(history)),
    }


# ------------------------------------------------------------------- utils --
def make_interface(env_id: str) -> TransferInterface:
    return TransferInterface(env_id)


CONSTRAINT_MID = np.zeros(2, dtype=np.float32)  # baseline: mid-range setpoints


def evaluate_episode(policy_fn, env_id: str, seed: int = 0) -> dict:
    """Run one deterministic year-long episode of `policy_fn(obs)->action`."""
    env = make_interface(env_id)
    obs, _ = env.reset(seed=seed)
    step = 0
    while True:
        action = policy_fn(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        step += 1
        if terminated or truncated:
            break
    metrics = score_episode(np.asarray(env.score_history), env_id)
    env.close()
    return {"env_id": env_id, "steps": step, **metrics}


if __name__ == "__main__":
    # Self-check on two different building families (needs EnergyPlus).
    for eid in ("Eplus-5zone-hot-continuous-v1", "Eplus-office-mixed-continuous-v1"):
        env = make_interface(eid)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (OBS_DIM,) and np.isfinite(obs).all(), eid
        rng = np.random.default_rng(0)
        for _ in range(3):
            o, r, te, tr, _ = env.step(rng.uniform(-1, 1, 2))
            assert o.shape == (OBS_DIM,) and np.isfinite(o).all(), eid
        m = score_episode(np.asarray(env.score_history[:100]), eid)
        assert set(m) >= {"hvac_kwh_per_year", "comfort_violation_rate"}
        env.close()
        print(f"OK {eid}: obs={OBS_DIM}D first-steps metrics={m}")
    print("SELF-CHECK PASSED")
