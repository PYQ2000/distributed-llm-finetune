from dft.callbacks import compute_throughput


def test_throughput_basic():
    sps, tps = compute_throughput(num_samples=100, seq_len=1024, elapsed_s=10.0)
    assert sps == 10.0
    assert tps == 10.0 * 1024


def test_throughput_zero_elapsed_safe():
    assert compute_throughput(10, 1024, 0.0) == (0.0, 0.0)
