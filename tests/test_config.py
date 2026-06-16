import pytest
from dft.config import TrainConfig


def test_from_yaml_loads_base(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model_name: foo/bar\nmax_samples: 123\n", encoding="utf-8")
    cfg = TrainConfig.from_yaml(p)
    assert cfg.model_name == "foo/bar"
    assert cfg.max_samples == 123
    assert cfg.max_seq_len == 1024


def test_overrides_skip_none(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("run_name: base\n", encoding="utf-8")
    cfg = TrainConfig.from_yaml(p, overrides={"run_name": "x", "max_steps": None})
    assert cfg.run_name == "x"
    assert cfg.max_steps == -1


def test_unknown_key_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("nope: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        TrainConfig.from_yaml(p)
