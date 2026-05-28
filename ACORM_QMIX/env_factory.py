"""Environment factory for RACORM.

This file adds SMACv2 support while keeping the original SMAC path unchanged.
SMACv2 is loaded from a local checkout by adding --smacv2_path to sys.path.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, Tuple


_RACE_UNIT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "terran": {
        "map_name": "10gen_terran",
        "unit_types": ["marine", "marauder", "medivac"],
        "weights": [0.45, 0.45, 0.10],
        "exception_unit_types": ["medivac"],
    },
    "protoss": {
        "map_name": "10gen_protoss",
        "unit_types": ["stalker", "zealot", "colossus"],
        "weights": [0.45, 0.45, 0.10],
        "exception_unit_types": ["colossus"],
    },
    "zerg": {
        "map_name": "10gen_zerg",
        "unit_types": ["zergling", "hydralisk", "baneling"],
        "weights": [0.45, 0.45, 0.10],
        "exception_unit_types": ["baneling"],
    },
}


def _str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _parse_smacv2_scenario(env_name: str, args) -> Tuple[str, int, int]:
    """Infer race, ally count, and enemy count from names like protoss_5_vs_5.

    If env_name does not match SMACv2 generated-scenario naming, fall back to
    --smacv2_race, --smacv2_n_units and --smacv2_n_enemies.
    """
    match = re.match(r"^(terran|protoss|zerg)_(\d+)_vs_(\d+)$", env_name)
    if match is not None:
        race = match.group(1)
        n_units = int(match.group(2))
        n_enemies = int(match.group(3))
        return race, n_units, n_enemies

    race = str(getattr(args, "smacv2_race", "terran")).lower()
    if race == "auto":
        race = "terran"
    if race not in _RACE_UNIT_CONFIGS:
        raise ValueError(
            f"Unknown SMACv2 race {race!r}. Expected one of "
            f"{sorted(_RACE_UNIT_CONFIGS.keys())}, or use env_name like protoss_5_vs_5."
        )
    return race, int(getattr(args, "smacv2_n_units", 5)), int(getattr(args, "smacv2_n_enemies", 5))


def _load_optional_capability_config(args) -> Dict[str, Any] | None:
    """Load user-provided SMACv2 capability config from JSON text/file if supplied."""
    config_json = getattr(args, "smacv2_capability_config_json", "")
    config_path = getattr(args, "smacv2_capability_config_path", "")

    if config_json:
        return json.loads(config_json)

    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            text = f.read()
        if config_path.endswith(('.yaml', '.yml')):
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "PyYAML is required for --smacv2_capability_config_path *.yaml. "
                    "Install pyyaml or use a JSON config file."
                ) from exc
            return yaml.safe_load(text)
        return json.loads(text)

    return None


def build_smacv2_capability_config(args) -> Tuple[Dict[str, Any], str]:
    """Build a SMACv2 capability_config and map_name.

    The default mirrors the SMACv2 examples: weighted team generation and
    surrounded_and_reflect start positions.
    """
    explicit_config = _load_optional_capability_config(args)
    if explicit_config is not None:
        race, _, _ = _parse_smacv2_scenario(getattr(args, "env_name", ""), args)
        map_name = getattr(args, "smacv2_map_name", "") or _RACE_UNIT_CONFIGS[race]["map_name"]
        return explicit_config, map_name

    race, n_units, n_enemies = _parse_smacv2_scenario(getattr(args, "env_name", ""), args)
    race_cfg = _RACE_UNIT_CONFIGS[race]

    capability_config: Dict[str, Any] = {
        "n_units": n_units,
        "n_enemies": n_enemies,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": list(race_cfg["unit_types"]),
            "exception_unit_types": list(race_cfg["exception_unit_types"]),
            "weights": list(race_cfg["weights"]),
            "observe": _str2bool(getattr(args, "smacv2_team_gen_observe", True)),
        },
        "start_positions": {
            "dist_type": str(getattr(args, "smacv2_start_position_dist", "surrounded_and_reflect")),
            "p": float(getattr(args, "smacv2_surround_p", 0.5)),
            "n_enemies": n_enemies,
            "map_x": int(getattr(args, "smacv2_map_x", 32)),
            "map_y": int(getattr(args, "smacv2_map_y", 32)),
        },
    }

    map_name = getattr(args, "smacv2_map_name", "") or race_cfg["map_name"]
    return capability_config, map_name


def make_env(args):
    """Create either a SMAC or SMACv2 environment.

    Required args:
        env_backend: "smac" or "smacv2".
        env_name: SMAC map name or SMACv2 generated scenario name.
    """
    backend = str(getattr(args, "env_backend", "smac")).lower()

    if backend == "smac":
        from smac.env import StarCraft2Env

        return StarCraft2Env(map_name=args.env_name, seed=getattr(args, "seed", None))

    if backend != "smacv2":
        raise ValueError(f"Unsupported env_backend={backend!r}. Use 'smac' or 'smacv2'.")

    smacv2_path = os.path.abspath(str(getattr(args, "smacv2_path", "/home/zheping/RL/smacv2-main")))
    if smacv2_path and smacv2_path not in sys.path:
        sys.path.insert(0, smacv2_path)

    try:
        from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
    except ImportError as exc:
        raise ImportError(
            "Cannot import SMACv2. Check that --smacv2_path points to the SMACv2 repo "
            f"or install it with `pip install -e {smacv2_path}`. Current path: {smacv2_path}"
        ) from exc

    capability_config, map_name = build_smacv2_capability_config(args)
    env_kwargs: Dict[str, Any] = {
        "capability_config": capability_config,
        "map_name": map_name,
        "debug": _str2bool(getattr(args, "smacv2_debug", False)),
        "conic_fov": _str2bool(getattr(args, "smacv2_conic_fov", False)),
        "obs_own_pos": _str2bool(getattr(args, "smacv2_obs_own_pos", True)),
        "use_unit_ranges": _str2bool(getattr(args, "smacv2_use_unit_ranges", True)),
        "min_attack_range": float(getattr(args, "smacv2_min_attack_range", 2.0)),
    }

    # Some SMACv2 versions accept seed and some inherit older SMAC signatures.
    # Try seed first for reproducibility, then fall back cleanly if unsupported.
    seed = getattr(args, "seed", None)
    if seed is not None:
        try:
            return StarCraftCapabilityEnvWrapper(seed=seed, **env_kwargs)
        except TypeError as exc:
            if "seed" not in str(exc):
                raise

    return StarCraftCapabilityEnvWrapper(**env_kwargs)
