"""Configuration loading.

Engineering rule 3 (spec §2.4): no hard-coded parameters in code. Everything
lives in ``configs/default.yaml``; presets are thin overlays that deep-merge on
top of it, and CLI ``--set key.path=value`` overrides merge on top of those.

The resulting :class:`Config` is a read-only mapping with attribute access, so
call sites read as ``cfg.condition.blur.absolute_floor`` rather than a chain of
dictionary lookups that silently return ``None`` on a typo — a missing key
raises instead, which is what you want when a tuning parameter disappears.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_CONFIG = "default.yaml"


class ConfigError(KeyError):
    """Raised when a requested configuration key does not exist."""


class Config(Mapping):
    """Immutable nested mapping with attribute access.

    ``cfg.recon.track_b.chunk_frames`` and ``cfg["recon"]["track_b"]`` are
    equivalent; ``cfg.get_path("recon.track_b.chunk_frames")`` is the form used
    when the key itself is data (budget lookups, CLI overrides).
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]):
        object.__setattr__(self, "_data", dict(data))

    # -- Mapping protocol ---------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        try:
            value = self._data[key]
        except KeyError:
            raise ConfigError(f"no config key {key!r} (have: {sorted(self._data)})") from None
        return Config(value) if isinstance(value, dict) else value

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        return self[key]

    def __setattr__(self, key: str, value: Any) -> None:
        raise TypeError("Config is immutable; use Config.merged() to derive a variant")

    def __repr__(self) -> str:
        return f"Config({json.dumps(self._data, default=str, sort_keys=True)[:200]}...)"

    # -- Helpers ------------------------------------------------------------
    def get_path(self, path: str, default: Any = ...) -> Any:
        """Look up a dotted key path. Raises unless ``default`` is supplied."""
        node: Any = self
        for part in path.split("."):
            if isinstance(node, Config) and part in node:
                node = node[part]
            elif isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                if default is ...:
                    raise ConfigError(f"no config key {path!r} (failed at {part!r})")
                return default
        return node

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def merged(self, overlay: Mapping[str, Any]) -> "Config":
        """Return a new Config with ``overlay`` deep-merged on top."""
        return Config(deep_merge(self._data, overlay))


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base``, returning a new dict.

    Dicts merge key-by-key; every other type (including lists) is replaced
    wholesale, so a preset can shorten ``export.formats`` without having to
    express a removal.
    """
    out = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_override(text: str) -> tuple[str, Any]:
    """Parse a ``key.path=value`` CLI override.

    The value is parsed as YAML so ``--set budget.total_s=600``,
    ``--set condition.enabled=false`` and ``--set export.formats=[obj,ply]``
    all produce the right Python type.
    """
    if "=" not in text:
        raise ValueError(f"override {text!r} is not of the form key.path=value")
    path, _, raw = text.partition("=")
    path = path.strip()
    if not path:
        raise ValueError(f"override {text!r} has an empty key path")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw
    return path, value


def nest_override(path: str, value: Any) -> dict[str, Any]:
    """Turn ``("a.b.c", 1)`` into ``{"a": {"b": {"c": 1}}}``."""
    out: dict[str, Any] = {}
    node = out
    parts = path.split(".")
    for part in parts[:-1]:
        node[part] = {}
        node = node[part]
    node[parts[-1]] = value
    return out


def load_config(
    preset: str | None = None,
    overrides: list[str] | None = None,
    config_dir: os.PathLike[str] | str | None = None,
    extra_files: list[str] | None = None,
) -> Config:
    """Load ``default.yaml``, then the preset, then any explicit files and overrides.

    Args:
        preset: name of a preset in ``configs/`` (``fast``, ``accurate``), with
            or without the ``.yaml`` suffix. ``None``/``default`` loads nothing extra.
        overrides: ``key.path=value`` strings from the CLI, applied last.
        config_dir: alternate configs directory (tests use this).
        extra_files: additional YAML paths merged between preset and overrides.
    """
    directory = Path(config_dir) if config_dir else CONFIG_DIR
    base_path = directory / DEFAULT_CONFIG
    if not base_path.is_file():
        raise FileNotFoundError(f"default config not found at {base_path}")
    data = _read_yaml(base_path)
    sources = [str(base_path)]

    if preset and preset not in ("default", DEFAULT_CONFIG):
        preset_name = preset if preset.endswith((".yaml", ".yml")) else f"{preset}.yaml"
        preset_path = Path(preset_name)
        if not preset_path.is_file():
            preset_path = directory / preset_name
        if not preset_path.is_file():
            available = sorted(p.stem for p in directory.glob("*.yaml"))
            raise FileNotFoundError(f"preset {preset!r} not found; available: {available}")
        data = deep_merge(data, _read_yaml(preset_path))
        sources.append(str(preset_path))

    for extra in extra_files or []:
        extra_path = Path(extra)
        if not extra_path.is_file():
            raise FileNotFoundError(f"config file not found: {extra}")
        data = deep_merge(data, _read_yaml(extra_path))
        sources.append(str(extra_path))

    applied: list[str] = []
    for item in overrides or []:
        path, value = parse_override(item)
        data = deep_merge(data, nest_override(path, value))
        applied.append(item)

    # Provenance travels with the config so the QA report can state exactly
    # which knobs produced a given number.
    data["_provenance"] = {"sources": sources, "overrides": applied}
    return Config(data)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must contain a YAML mapping at the top level")
    return loaded
