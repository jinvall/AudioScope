"""Configuration validation (docs/ARCHITECTURE.md section 10)."""

from __future__ import annotations

import json

import pytest

from app.config import (
    RING_BUFFER_MAX_SECONDS,
    RING_BUFFER_MIN_SECONDS,
    AppConfig,
    ConfigError,
)


def test_defaults_match_the_documented_values():
    c = AppConfig()
    assert c.sample_rate == 48000
    assert c.buffer.seconds == 30.0
    assert c.event.pre_roll_seconds == 5.0
    assert c.event.post_roll_seconds == 5.0
    assert c.record.chunk_seconds == 900.0
    assert c.record.enabled is True
    assert c.record.retention_days is None


def test_derived_frame_counts():
    c = AppConfig()
    assert c.capacity_frames == 30 * 48000
    assert c.pre_roll_frames == 5 * 48000
    assert c.post_roll_frames == 5 * 48000


def test_buffer_range_is_enforced():
    AppConfig(buffer={"seconds": RING_BUFFER_MIN_SECONDS})
    AppConfig(buffer={"seconds": RING_BUFFER_MAX_SECONDS})
    with pytest.raises(ConfigError):
        AppConfig(buffer={"seconds": 1.0})
    with pytest.raises(ConfigError):
        AppConfig(buffer={"seconds": 500.0})


def test_buffer_must_hold_the_configured_pre_roll():
    """Pre-roll that cannot be satisfied must be rejected, not silently cut."""
    with pytest.raises(ConfigError):
        AppConfig(buffer={"seconds": 10.0}, event={"pre_roll_seconds": 30.0})


def test_mono_internal_path_is_enforced():
    with pytest.raises(ConfigError):
        AppConfig(audio={"channels": 2})


def test_invalid_block_size():
    with pytest.raises(ConfigError):
        AppConfig(audio={"block_size": 0})


def test_volume_range():
    with pytest.raises(ConfigError):
        AppConfig(playback={"volume": 1.5})
    with pytest.raises(ConfigError):
        AppConfig(playback={"volume": -0.1})


def test_retention_must_be_positive_or_none():
    AppConfig(record={"retention_days": None})
    with pytest.raises(ConfigError):
        AppConfig(record={"retention_days": 0})


def test_json_round_trip(tmp_path):
    c = AppConfig(audio={"block_size": 512}, record={"chunk_seconds": 60.0})
    path = str(tmp_path / "config.json")
    c.save(path)
    loaded = AppConfig.load(path)
    assert loaded.audio.block_size == 512
    assert loaded.record.chunk_seconds == 60.0


def test_load_missing_file_returns_defaults(tmp_path):
    loaded = AppConfig.load(str(tmp_path / "absent.json"))
    assert loaded.buffer.seconds == 30.0


def test_unknown_keys_are_rejected(tmp_path):
    """A typo in a config key must fail loudly, not be silently ignored.

    Silently dropping an unrecognised key is how a user ends up running with a
    setting they believe they changed.
    """
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"audio": {"block_size": 256, "bogus": 1}}))
    with pytest.raises(ConfigError) as excinfo:
        AppConfig.load(str(path))
    assert "bogus" in str(excinfo.value)


def test_non_object_section_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"audio": 5}))
    with pytest.raises(ConfigError):
        AppConfig.load(str(path))
