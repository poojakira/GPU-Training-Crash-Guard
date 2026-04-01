import pytest
from unittest.mock import patch
from gpudefrag.cli import main

def test_cli_help(capsys):
    with patch("sys.argv", ["gpu-defragger", "--help"]):
        with pytest.raises(SystemExit):
            main()
    out, err = capsys.readouterr()
    assert "gpudefrag" in out
    assert "profile" in out
    assert "train" in out

def test_cli_simulate(capsys):
    with patch("sys.argv", ["gpu-defragger", "simulate", "--runs", "1", "--steps", "2"]):
        with patch("benchmarks.run_local_benchmark.main") as mock_benchmark:
            main()
            mock_benchmark.assert_called_once()
    out, err = capsys.readouterr()
    assert "Launching benchmark" in out

@patch("gpudefrag.profiler.collector.collect_from_model")
def test_cli_profile(mock_collect, capsys):
    mock_collect.return_value = 100
    with patch("sys.argv", ["gpu-defragger", "profile", "--model", "gpt2", "--iterations", "10"]):
        main()
        mock_collect.assert_called_once_with("gpt2", iterations=10)
    out, err = capsys.readouterr()
    assert "100 events collected" in out
