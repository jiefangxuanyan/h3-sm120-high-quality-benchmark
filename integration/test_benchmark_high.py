"""CPU checks of the actual direct launch chain and routing failure guards."""

import copy
import json
import os
import subprocess

import pytest

from integration import benchmark_high as task


def test_report_matrix_and_alternating_order():
    cases = task.load_cases(task.DEFAULT_CASES)
    assert len(cases) == 12
    plan = task.make_plan(cases)
    assert len(plan) == 60
    assert [x["case"]["name"] for x in plan[:12]] == [
        x["case"]["name"] for x in plan[12:24]
    ][::-1]
    for c in cases:
        assert sum(x["case"]["name"] == c["name"] for x in plan) == (
            8 if len(c["gpus"]) == 8 else 4
        )
    assert len(task.make_plan(cases, profile=True)) == 12


def test_dry_run_never_accesses_hardware(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["benchmark_high"])
    monkeypatch.setattr(
        task.subprocess, "check_output", lambda *a, **k: pytest.fail("hardware access")
    )
    monkeypatch.setattr(
        task.subprocess, "run", lambda *a, **k: pytest.fail("execution")
    )
    task.main()
    assert json.loads(capsys.readouterr().out)["trials"] == 60


def test_nic_bindings_and_environment_isolation(tmp_path, monkeypatch):
    cases = task.load_cases(task.DEFAULT_CASES)
    monkeypatch.setenv("NCCL_NET_DISABLE_INTRA", "1")
    monkeypatch.setenv("NCCL_IB_DISABLE", "1")
    monkeypatch.setenv("FLASHINFER_ULYSSES_PCIE_ROUTE", "hybrid")
    monkeypatch.setenv("H3_FAKE_POLICY", "inherited")
    monkeypatch.delenv("NCCL_IB_HCA", raising=False)
    task.bind_nics(cases, ["mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3"])
    shared = next(c for c in cases if c["name"] == "4-rdma-0123-shared")
    assert shared["nics"] == ["mlx5_0", "mlx5_0", "mlx5_1", "mlx5_1"]
    for c in cases:
        env = task.environment(c, tmp_path, task.DEFAULT_CONFIG)
        assert (
            "H3_FAKE_POLICY" not in env and "FLASHINFER_ULYSSES_PCIE_ROUTE" not in env
        )
        assert "NCCL_IB_DISABLE" not in env
        assert env.get("NCCL_NET_DISABLE_INTRA") == (
            "1" if c["mode"] == "pcie" else None
        )
        if c["mode"] == "rdma":
            assert "NCCL_IB_HCA" not in env
            assert env["FLASHINFER_ULYSSES_PCIE_NICS"].split(",") == c["nics"]
    with pytest.raises(ValueError, match="requires --nics"):
        task.bind_nics([copy.deepcopy(shared)], [], execute=True)
    with pytest.raises(ValueError, match="referenced slots"):
        task.bind_nics([copy.deepcopy(shared)], ["mlx5_0"])


@pytest.mark.parametrize("profile", [False, True])
@pytest.mark.parametrize(
    "case",
    task.load_cases(task.DEFAULT_CASES) + task.load_cases(task.EIGHT_ENDPOINT_CASES),
    ids=lambda c: c["name"],
)
def test_real_shell_chain_preserves_gpu_selection_and_capture(tmp_path, case, profile):
    c = copy.deepcopy(case)
    task.bind_nics([c], [f"mlx5_{i}" for i in range(8)])
    bin = tmp_path / "bin"
    bin.mkdir()
    (bin / "nvtx/include/nvtx3").mkdir(parents=True)
    (bin / "nvtx/include/nvtx3/nvToolsExt.h").touch()
    fake_python = bin / "python"
    fake_python.write_text(
        '#!/usr/bin/python3\nimport json,os,sys\nopen(os.environ["CAPTURE"],"w").write(json.dumps(dict(argv=sys.argv[1:],env=dict(os.environ))))\n'
    )
    fake_python.chmod(0o755)
    fake_nsys = bin / "nsys"
    fake_nsys.write_text("""#!/usr/bin/python3
from pathlib import Path
import subprocess,sys
args=sys.argv[1:]
assert '--capture-range=nvtx' in args and '--nvtx-capture=H3_MEASURED_REQUEST' in args
assert '--trace=cuda,nvtx,osrt,nccl' in args and '--capture-range-end=stop' in args
subprocess.run(args[args.index('bash'):],check=True)
Path(next(x.split('=',1)[1] for x in args if x.startswith('--output='))+'.nsys-rep').write_bytes(b'report')
""")
    fake_nsys.chmod(0o755)
    target = tmp_path / "result"
    target.mkdir()
    capture = tmp_path / "capture.json"
    env = task.environment(c, target, task.DEFAULT_CONFIG, profile)
    env.update(
        PATH=str(bin) + ":" + os.environ["PATH"],
        MODEL_PATH=str(tmp_path),
        NSYS_BIN=str(fake_nsys),
        CAPTURE=str(capture),
    )
    subprocess.run(
        task.command(c, profile), env=env, check=True, capture_output=True, text=True
    )
    data = json.loads(capture.read_text())
    argv = data["argv"]

    def value(key):
        # Caller-supplied options override shell defaults, as in the real CLI.
        return argv[max(i for i, arg in enumerate(argv) if arg == key) + 1]

    assert value("--num-gpus") == value("--ulysses-degree") == str(len(c["gpus"]))
    assert value("--gpu-ids") == ",".join(map(str, c["gpus"]))
    assert value("--minimax-h3-adaln-host-cache-gb") == "2"
    assert value("--num-inference-steps") == "50" and value("--quality") == "high"
    assert data["env"]["H3_FULL_REQUEST_WARMUP"] == "1"
    expected_residency = {"video_vae": "resident", **c.get("component_residency", {})}
    assert dict(p.split("=", 1) for p in value("--component-residency").split(",")) == expected_residency
    if c["nccl_nics"]:
        assert data["env"]["NCCL_IB_HCA"] == "=" + ",".join(c["nccl_nics"])
    if c["mode"] == "rdma":
        assert data["env"]["FLASHINFER_ULYSSES_PCIE_NICS"].split(",") == c["nics"]
    if profile:
        assert value("--enable-layerwise-nvtx-marker") == "true"
        assert (target / (c["name"] + ".nsys-rep")).read_bytes() == b"report"


def test_pcie_auxiliary_net_connection_is_rejected(tmp_path):
    c = dict(mode="pcie", gpus=[0, 1])
    lines = [
        "NCCL_NET_DISABLE_INTRA set by environment to 1",
        "rank 0 nranks 2",
        "rank 1 nranks 2",
        "Channel 00 : 0 -> 1 via P2P/CUMEM",
    ]
    p = tmp_path / "nccl.host.1.log"
    p.write_text("\n".join(lines))
    task.validate_nccl(c, tmp_path)
    p.write_text("\n".join(lines).replace("via P2P/CUMEM", "via NET/IB/0/GDRDMA"))
    with pytest.raises(ValueError, match="auxiliary NCCL used NET"):
        task.validate_nccl(c, tmp_path)


def test_gpu_preflight_checks_only_selected_devices_and_refuses_active_users(
    tmp_path, monkeypatch
):
    calls = []

    def query(args, **kw):
        calls.append(args)
        return (
            "123, GPU-uuid, 2000 MiB"
            if "compute-apps" in " ".join(args)
            else "2, 00000000:81:00.0, 2000\n"
        )

    monkeypatch.setattr(task.subprocess, "check_output", query)
    with pytest.raises(RuntimeError, match="active compute processes"):
        task.gpu_snapshot(dict(gpus=[2]), tmp_path, "before")
    assert all(args[args.index("-i") + 1] == "2" for args in calls)


def test_missing_pin_counters_are_explicitly_unavailable():
    assert task.memory_check(
        dict(pins=None, boot="a"), dict(pins=None, boot="a")
    ) == dict(same_boot=True, pin_growth_bytes=None)


def test_distinct_nic_slots_cannot_silently_become_shared():
    cases = task.load_cases(task.DEFAULT_CASES)
    with pytest.raises(ValueError, match="distinct device names"):
        task.bind_nics(cases, ["mlx5_0", "mlx5_0", "mlx5_2", "mlx5_3"])


def test_failed_trial_stops_and_preserves_failure_artifacts(tmp_path, monkeypatch):
    cases = task.load_cases(task.DEFAULT_CASES)[:1]
    plan = task.make_plan(cases, repeats=2)
    output = tmp_path / "output"
    monkeypatch.setattr(task, "source_revisions", lambda: {})
    monkeypatch.setattr(task, "gpu_snapshot", lambda *a: None)
    monkeypatch.setattr(task, "memory", lambda: dict(pins=0, boot="a"))
    calls = []

    def fail(*args, **kwargs):
        calls.append(args)
        raise subprocess.CalledProcessError(1, "failed model")

    monkeypatch.setattr(task.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        task.run(plan, output, task.DEFAULT_CONFIG, False, 10)
    assert len(calls) == 1
    assert (output / "repeat-01/8-nccl/failure.txt").is_file()
    assert (output / "repeat-01/8-nccl/memory.json").is_file()
    assert not (output / "repeat-02").exists()
    assert json.loads((output / "summary.json").read_text())["completed"] == 0


@pytest.mark.parametrize(
    "mask, allowed",
    [
        ("0,1,2,3", True),
        ("0,1", True),
        ("1,0", False),
        ("2,3", False),
        ("0", False),
        ("", False),
    ],
)
def test_execution_accepts_only_visibility_masks_preserving_physical_ids(
    tmp_path, monkeypatch, mask, allowed
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_high",
            "--case",
            "2-nccl-01",
            "--execute",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    calls = []
    monkeypatch.setattr(task, "run", lambda *args: calls.append(args))
    if allowed:
        task.main()
        assert len(calls) == 1
    else:
        with pytest.raises(SystemExit):
            task.main()
        assert not calls


def test_eight_endpoint_matrix_covers_shared_and_dedicated_nccl_sets(
    tmp_path, monkeypatch
):
    cases = task.load_cases(task.EIGHT_ENDPOINT_CASES)
    assert len(cases) == 16 and len(task.make_plan(cases)) == 84
    assert len(task.make_plan(cases, profile=True)) == 16
    devices = [f"mlx5_{i}" for i in range(8)]
    monkeypatch.setenv("NCCL_IB_HCA", "=unexpected_inherited_device")
    task.bind_nics(cases, devices)
    by_name = {c["name"]: c for c in cases}
    full = by_name["8-rdma-8endpoints"]
    shared = by_name["8-rdma-4endpoints"]
    assert full["nics"] == devices
    assert shared["nics"] == [devices[i] for i in [0, 0, 2, 2, 4, 4, 6, 6]]
    for c in cases:
        env = task.environment(c, tmp_path, task.DEFAULT_CONFIG)
        if c["mode"] == "pcie":
            assert env["NCCL_NET_DISABLE_INTRA"] == "1"
        elif len(c["gpus"]) > 1:
            assert env["NCCL_IB_HCA"] == "=" + ",".join(c["nccl_nics"])
            if c["mode"] == "rdma":
                assert set(c["nics"]) == set(c["nccl_nics"])
    assert by_name["4-rdma-adjacent-dedicated"]["nic_slots"] == [0, 1, 2, 3]
    assert by_name["4-rdma-spread-dedicated"]["nic_slots"] == [0, 2, 4, 6]
    assert by_name["2-rdma-shared"]["nic_slots"] == [0, 0]
    assert by_name["2-rdma-adjacent-dedicated"]["nic_slots"] == [0, 1]
    assert by_name["2-rdma-spread-dedicated"]["nic_slots"] == [0, 2]
    assert by_name["4-nccl"]["nccl_nic_slots"] == [0, 2]
    assert by_name["2-nccl"]["nccl_nic_slots"] == [0]
    nccl_four = by_name["8-nccl-4endpoints"]
    assert (
        task.environment(nccl_four, tmp_path, task.DEFAULT_CONFIG)["NCCL_IB_HCA"]
        == "=mlx5_0,mlx5_2,mlx5_4,mlx5_6"
    )


def test_single_gpu_offload_samples_four_devices_but_profiles_only_one():
    cases = [c for c in task.load_cases(task.EIGHT_ENDPOINT_CASES) if len(c["gpus"]) == 1]
    assert [t["case"]["gpus"] for t in task.make_plan(cases)] == [[0], [1], [2], [3]]
    profile = task.make_plan(cases, profile=True)
    assert len(profile) == 1 and profile[0]["case"]["gpus"] == [0]
    assert cases[0]["gpus"] == [0]


@pytest.mark.parametrize("mask, allowed", [("0", False), ("0,1,2,3", True)])
def test_visibility_checks_every_single_gpu_sample(tmp_path, monkeypatch, mask, allowed):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    monkeypatch.setattr("sys.argv", [
        "benchmark_high", "--cases", str(task.EIGHT_ENDPOINT_CASES),
        "--case", "1-local-layerwise-offload", "--execute", "--output", str(tmp_path / "out"),
    ])
    calls = []
    monkeypatch.setattr(task, "run", lambda *args: calls.append(args))
    if allowed:
        task.main()
        assert len(calls) == 1
    else:
        with pytest.raises(SystemExit):
            task.main()
        assert not calls


def test_single_gpu_offload_validates_the_actual_residency(tmp_path):
    case = next(c for c in task.load_cases(task.EIGHT_ENDPOINT_CASES) if len(c["gpus"]) == 1)
    args = dict(num_gpus=1, ulysses_degree=1, gpu_ids=[0], tp_size=1, ring_degree=1,
                component_attention_backends=dict(transformer="subblock_sparse_attn_sm120_sage"),
                minimax_h3_adaln_host_cache_gb=2, component_residency=dict(video_vae="resident"))
    (tmp_path / "nccl.log").write_text("server_args: " + json.dumps(args) + "\n")
    with pytest.raises(ValueError, match="residency"):
        task.validate(case, tmp_path)


def test_shared_host_pin_growth_is_recorded_without_discarding_the_sample(tmp_path, monkeypatch):
    plan = task.make_plan(task.load_cases(task.DEFAULT_CASES)[:1], repeats=1)
    monkeypatch.setattr(task, "source_revisions", lambda: {})
    monkeypatch.setattr(task, "gpu_snapshot", lambda *a: None)
    memory = iter([dict(pins=0, boot="a"), dict(pins=1024**2, boot="a")])
    monkeypatch.setattr(task, "memory", lambda: next(memory))
    monkeypatch.setattr(task.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(task, "validate", lambda *a: dict(
        **{k: 1 for k in task.METRICS}, cached_steps=[], video_sha256="unused"))
    output = tmp_path / "output"
    task.run(plan, output, task.DEFAULT_CONFIG, False, 10)
    assert json.loads((output / "summary.json").read_text())["complete"]
    recorded = json.loads((output / "repeat-01/8-nccl/memory.json").read_text())
    assert recorded["pin_growth_bytes"] > 64 * 1024**2


def test_inactive_unused_endpoints_do_not_block_a_subset_experiment(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    cases = [
        c
        for c in task.load_cases(task.EIGHT_ENDPOINT_CASES)
        if c["name"] == "8-nccl-4endpoints"
    ]
    checked = []

    def states(path, pattern):
        device = path.parent.name
        checked.append(device)
        return [
            SimpleNamespace(
                read_text=lambda: (
                    "4: ACTIVE" if int(device.split("_")[1]) % 2 == 0 else "1: DOWN"
                )
            )
        ]

    monkeypatch.setattr(Path, "glob", states)
    task.bind_nics(cases, [f"mlx5_{i}" for i in range(8)], execute=True)
    assert set(checked) == {"mlx5_0", "mlx5_2", "mlx5_4", "mlx5_6"}


@pytest.mark.parametrize("slots", [None, [], [0, 0], [-1], [True], ["0"]])
def test_invalid_nccl_endpoint_sets_are_rejected(tmp_path, slots):
    p = tmp_path / "cases.json"
    p.write_text(
        json.dumps([dict(name="bad", mode="nccl", gpus=[0, 1], nccl_nic_slots=slots)])
    )
    with pytest.raises(ValueError, match="nccl_nic_slots"):
        task.load_cases(p)
