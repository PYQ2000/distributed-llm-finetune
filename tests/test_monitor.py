from dft.monitor import parse_nvidia_smi_line, GPU_FIELDS


def test_parse_line():
    line = "2026/06/16 10:00:00.000, 0, 87, 9000, 16384, 61, 70.5"
    row = parse_nvidia_smi_line(line)
    assert row["index"] == "0"
    assert row["utilization_gpu"] == "87"
    assert row["memory_used"] == "9000"
    assert set(row.keys()) == set(GPU_FIELDS)
