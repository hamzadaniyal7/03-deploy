"""The command line, exercised the way a user would."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from selfserve.cli import main
from selfserve.mock import MockServer


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_no_command_prints_help(workdir, capsys) -> None:
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out


# -------------------------------------------------------------------- fit
def test_fit_needs_no_server(workdir, capsys) -> None:
    assert main(["fit", "llama-3.1-8b", "--gpu", "a100-80"]) == 0
    out = capsys.readouterr().out
    assert "KV cache" in out and "concurrent sequences" in out


def test_fit_json(workdir, capsys) -> None:
    assert main(["fit", "llama-3.1-8b", "--gpu", "h100", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["fits"] is True and data["bytes"]["kv_per_token"] == 131072


def test_fit_exits_non_zero_when_it_does_not_fit(workdir, capsys) -> None:
    """So a deployment script can check before starting anything."""
    assert main(["fit", "llama-3.1-70b", "--gpu", "a10g"]) == 1
    assert "DOES NOT FIT" in capsys.readouterr().out


def test_fit_lists_what_it_knows(workdir, capsys) -> None:
    assert main(["fit", "--list"]) == 0
    out = capsys.readouterr().out
    assert "llama-3.1-8b" in out and "h100" in out


def test_fit_without_a_model_explains_itself(workdir, capsys) -> None:
    assert main(["fit"]) == 2
    assert "--hf-config" in capsys.readouterr().err


def test_fit_rejects_an_unknown_model(workdir, capsys) -> None:
    assert main(["fit", "gpt-5"]) == 2
    assert "unknown model" in capsys.readouterr().err


def test_fit_reads_a_huggingface_config(workdir, capsys) -> None:
    config = workdir / "config.json"
    config.write_text(
        json.dumps(
            {
                "num_hidden_layers": 32,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 14336,
                "vocab_size": 128256,
                "max_position_embeddings": 8192,
            }
        )
    )
    assert main(["fit", "--hf-config", str(config), "--params-b", "8.03", "--gpu", "a100-80"]) == 0
    assert "KV per token" in capsys.readouterr().out


def test_fit_context_flag_changes_the_answer(workdir, capsys) -> None:
    main(["fit", "llama-3.1-8b", "--gpu", "a100-80", "--context", "4096", "--json"])
    small = json.loads(capsys.readouterr().out)["max_sequences"]
    main(["fit", "llama-3.1-8b", "--gpu", "a100-80", "--context", "32768", "--json"])
    large = json.loads(capsys.readouterr().out)["max_sequences"]
    assert small > large


# ------------------------------------------------------------------ probe
def test_probe_a_live_endpoint(workdir, mock: MockServer, capsys) -> None:
    assert main(["probe", mock.url]) == 0
    assert "usable" in capsys.readouterr().out


def test_probe_a_dead_endpoint(workdir, capsys) -> None:
    assert main(["probe", "http://127.0.0.1:9"]) == 1
    assert "NOT usable" in capsys.readouterr().out


def test_probe_json(workdir, mock: MockServer, capsys) -> None:
    assert main(["probe", mock.url, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True


# ------------------------------------------------------------------ bench
def test_bench_against_the_mock(workdir, mock: MockServer, capsys) -> None:
    assert (
        main(
            [
                "bench",
                mock.url,
                "-c",
                "2",
                "-n",
                "4",
                "--max-tokens",
                "4",
                "--warmup",
                "0",
                "--no-save",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "output tokens/s" in out and "time to first token" in out


def test_bench_json(workdir, mock: MockServer, capsys) -> None:
    assert (
        main(
            [
                "bench",
                mock.url,
                "-c",
                "2",
                "-n",
                "4",
                "--max-tokens",
                "4",
                "--warmup",
                "0",
                "--json",
            ]
        )
        == 0
    )
    data = json.loads(capsys.readouterr().out)
    assert data["counts"]["completed"] == 4


def test_bench_sweep(workdir, mock: MockServer, capsys) -> None:
    assert (
        main(
            [
                "bench",
                mock.url,
                "--sweep",
                "1,2",
                "-n",
                "4",
                "--max-tokens",
                "3",
                "--warmup",
                "0",
                "--no-save",
            ]
        )
        == 0
    )
    assert "peak throughput" in capsys.readouterr().out


def test_bench_writes_a_report(workdir, mock: MockServer, capsys) -> None:
    main(["bench", mock.url, "-c", "2", "-n", "4", "--max-tokens", "3", "--warmup", "0"])
    assert (workdir / "bench" / "reports" / "c2.json").is_file()


def test_bench_against_a_dead_endpoint_exits_non_zero(workdir, capsys) -> None:
    assert (
        main(["bench", "http://127.0.0.1:9", "-c", "1", "-n", "2", "--warmup", "0", "--no-save"])
        == 1
    )


# ------------------------------------------------------------------- init
def test_init_writes_a_config(workdir, capsys) -> None:
    assert main(["init"]) == 0
    assert "[selfserve]" in (workdir / "selfserve.toml").read_text()


def test_init_refuses_to_clobber(workdir, capsys) -> None:
    main(["init"])
    assert main(["init"]) == 2
    assert main(["init", "--force"]) == 0


def test_config_file_is_picked_up(workdir, mock: MockServer, capsys) -> None:
    (workdir / "selfserve.toml").write_text(
        f'[selfserve]\nbackends = ["{mock.url}"]\nconcurrency = 2\nrequests = 2\n'
    )
    # No URL argument: it should come from the config file.
    assert main(["probe"]) == 0
    assert "usable" in capsys.readouterr().out
