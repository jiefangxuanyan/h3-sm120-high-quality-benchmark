"""Direct high-quality timing and profiling; planning is the default."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

from integration.summarize_high import summarize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "integration/examples/benchmark-cases.json"
DEFAULT_CONFIG = ROOT / "integration/examples/t2va.json"
EIGHT_ENDPOINT_CASES = ROOT / "integration/examples/benchmark-cases-8-endpoints.json"
METRICS = ("total_request_s", "denoise_s", "decode_s", "runtime_peak_mib")
TUNING = {
    "NCCL_P2P_LEVEL",
    "NCCL_P2P_DISABLE",
    "NCCL_IB_DISABLE",
    "NCCL_SHM_DISABLE",
    "NCCL_NET_DISABLE_INTRA",
    "NCCL_NETDEVS_POLICY",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_NET_GDR_READ",
    "NCCL_ALGO",
    "NCCL_PROTO",
    "ATTENTION_CONFIG",
    "MAX_SEQ_LEN",
    "MAX_GEOMETRIES",
}


def load_cases(path):
    cases = json.loads(path.read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases must be a nonempty JSON list")
    names = set()
    for c in cases:
        name, mode, gpus = c["name"], c["mode"], c["gpus"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in names:
            raise ValueError("Case names must be unique path-safe identifiers")
        names.add(name)
        if mode not in ("nccl", "rdma", "pcie"):
            raise ValueError("Expected nccl, rdma or pcie mode")
        if (
            len(gpus) not in (1, 2, 4, 8)
            or len(set(gpus)) != len(gpus)
            or any(type(g) is not int or not 0 <= g <= 7 for g in gpus)
        ):
            raise ValueError("Expected 1, 2, 4 or 8 distinct GPU indices in 0..7")
        slots = c.get("nic_slots", [])
        if mode == "rdma":
            if (
                len(gpus) == 1
                or len(slots) != len(gpus)
                or any(type(n) is not int or n < 0 for n in slots)
            ):
                raise ValueError(
                    "RDMA needs one nonnegative NIC slot per rank and multiple GPUs"
                )
        elif slots or (len(gpus) == 1 and mode != "nccl"):
            raise ValueError(
                "Only RDMA accepts NIC slots; single GPU uses nccl-labelled local attention"
            )
        if "gpu_selections" in c:
            selections = c["gpu_selections"]
            if (
                len(gpus) != 1
                or not isinstance(selections, list)
                or not selections
                or any(
                    not isinstance(s, list)
                    or len(s) != 1
                    or type(s[0]) is not int
                    or not 0 <= s[0] <= 7
                    for s in selections
                )
            ):
                raise ValueError(
                    "gpu_selections requires a nonempty list of single-GPU placements"
                )
        if "component_residency" in c:
            residency = c["component_residency"]
            if (
                not isinstance(residency, dict)
                or not residency
                or any(
                    not isinstance(k, str) or not isinstance(v, str)
                    for k, v in residency.items()
                )
            ):
                raise ValueError(
                    "component_residency must map component names to policies"
                )
        nccl_slots = c.get("nccl_nic_slots")
        if "nccl_nic_slots" in c and (
            not isinstance(nccl_slots, list)
            or not nccl_slots
            or any(type(n) is not int or n < 0 for n in nccl_slots)
            or len(set(nccl_slots)) != len(nccl_slots)
            or mode == "pcie"
            or len(gpus) == 1
        ):
            raise ValueError(
                "nccl_nic_slots must be a nonempty unique NIC set for multi-GPU NCCL/RDMA"
            )
    return cases


def make_plan(cases, repeats=None, profile=False):
    counts = [
        1
        if profile
        else repeats
        if repeats is not None
        else 8
        if len(c["gpus"]) == 8
        else 4
        for c in cases
    ]
    plan = []
    for n in range(1, max(counts) + 1):
        active = [c for c, count in zip(cases, counts) if n <= count]
        if n % 2 == 0:
            active.reverse()
        for c in active:
            selected = c
            if "gpu_selections" in c:
                selected = dict(
                    c, gpus=c["gpu_selections"][(n - 1) % len(c["gpu_selections"])]
                )
            plan.append(dict(case=selected, repeat=n))
    return plan


def bind_nics(cases, devices, execute=False):
    if len(set(devices)) != len(devices):
        raise ValueError(
            "NIC slots must refer to distinct device names; share a NIC by repeating its slot"
        )
    for c in cases:
        main_slots = c.get("nic_slots", [])
        nccl_slots = c.get("nccl_nic_slots", [])
        required = main_slots + nccl_slots
        if devices and required and max(required) >= len(devices):
            raise ValueError(
                f"{c['name']}: --nics does not supply all referenced slots"
            )
        if required and not devices and execute:
            raise ValueError("RDMA or restricted NCCL execution requires --nics")

        def names(slots):
            return [devices[i] if devices else f"<nic{i}>" for i in slots]

        c["nics"] = names(main_slots)
        c["nccl_nics"] = names(nccl_slots)
    if execute:
        used = {n for c in cases for n in c["nics"] + c["nccl_nics"]}
        for device in sorted(used):
            ports = Path("/sys/class/infiniband") / device / "ports"
            if not any("ACTIVE" in p.read_text() for p in ports.glob("*/state")):
                raise ValueError(f"No ACTIVE RDMA port for {device}")


def environment(case, directory, config, profile=False):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in TUNING
        and not k.startswith(("H3_", "FLASHINFER_ULYSSES_", "SGLANG_H3_NSYS_", "NSYS_"))
    }
    env.update(
        RUN_DIR=str(directory),
        CONFIG=str(config),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        SGLANG_SRC=str(ROOT / "sglang-all"),
        FLASHINFER_SRC=str(ROOT / "flashinfer"),
        H3_FULL_REQUEST_WARMUP="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        SGLANG_USE_MODELSCOPE="false",
        FLASHINFER_NO_DOWNLOAD="1",
        NCCL_DEBUG="INFO",
        NCCL_DEBUG_SUBSYS="INIT,GRAPH,P2P,SHM,NET,ENV",
        NCCL_DEBUG_FILE=str(directory / "nccl.%h.%p.log"),
    )
    if case["mode"] == "rdma":
        env.update(
            FLASHINFER_ULYSSES_PCIE_ALLOW_SHARED_NICS="1",
            FLASHINFER_ULYSSES_PCIE_NICS=",".join(case["nics"]),
        )
    if case.get("nccl_nics"):
        env["NCCL_IB_HCA"] = "=" + ",".join(case["nccl_nics"])
    if case["mode"] == "pcie":
        env["NCCL_NET_DISABLE_INTRA"] = "1"
    if profile:
        env.update(NSYS_REPORT_DIR=str(directory), NSYS_REPORT_NAME=case["name"])
    return env


def command(case, profile=False):
    script = "profile_high_sglang.sh" if profile else "test_high_sglang.sh"
    args = [
        "bash",
        str(ROOT / "integration" / script),
        case["mode"],
        "--gpus",
        ",".join(map(str, case["gpus"])),
    ]
    if case.get("component_residency"):
        args += [
            "--component-residency",
            ",".join(
                f"{component}={policy}"
                for component, policy in case["component_residency"].items()
            ),
        ]
    return args


def gpu_snapshot(case, directory, label):
    selected = ",".join(map(str, case["gpus"]))
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            selected,
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    )
    inventory = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            selected,
            "--query-gpu=index,pci.bus_id,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    (directory / f"gpu-{label}.txt").write_text(inventory)
    if apps.strip():
        raise RuntimeError(
            "Selected GPUs have active compute processes; no process was stopped"
        )
    rows = list(csv.reader(inventory.splitlines()))
    if {int(r[0]) for r in rows} != set(case["gpus"]) or any(
        float(r[2]) > 1024 for r in rows
    ):
        raise RuntimeError(
            "Selected GPUs are missing or have more than 1024 MiB allocated"
        )


def memory():
    p = dict(l.split()[:2] for l in Path("/proc/vmstat").read_text().splitlines())
    pins = None
    if "nr_foll_pin_acquired" in p and "nr_foll_pin_released" in p:
        pins = int(p["nr_foll_pin_acquired"]) - int(p["nr_foll_pin_released"])
    return dict(
        pins=pins, boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    )


def memory_check(before, after):
    growth = (
        None
        if before["pins"] is None or after["pins"] is None
        else (after["pins"] - before["pins"]) * os.sysconf("SC_PAGE_SIZE")
    )
    return dict(same_boot=before["boot"] == after["boot"], pin_growth_bytes=growth)


def validate_nccl(case, directory):
    if len(case["gpus"]) == 1:
        return
    logs = list(directory.glob("nccl.*.log"))
    text = "\n".join(p.read_text() for p in logs)
    n = len(case["gpus"])
    ranks = {int(s) for s in re.findall(r"\brank (\d+) nranks " + str(n) + r"\b", text)}
    if ranks != set(range(n)):
        raise ValueError("Missing NCCL communicator diagnostics for all ranks")
    if case["mode"] == "pcie":
        if not re.search(r"NCCL_NET_DISABLE_INTRA.*\b1\b", text):
            raise ValueError("NCCL did not acknowledge NET_DISABLE_INTRA=1")
        connections = [s for s in text.splitlines() if "Channel" in s and " via " in s]
        if not connections or any(
            not re.search(r" via (?:P2P|SHM)/", s) for s in connections
        ):
            raise ValueError(
                "PCIe auxiliary NCCL used NET or another unexpected transport"
            )


def pci_key(text):
    return tuple(int(s, 16) for s in re.split("[:.]", text.strip().lower()))


def validate(case, directory):
    n, mode = len(case["gpus"]), case["mode"]
    text = (directory / f"{mode}.log").read_text()
    args = json.loads(re.search(r"server_args: (\{[^\n\r]+\})", text).group(1))
    backend = "subblock_sparse_attn_sm120_sage" + ("_a2a" if n > 1 else "")
    if (
        args["num_gpus"] != n
        or args["ulysses_degree"] != n
        or args["gpu_ids"] != case["gpus"]
        or args["tp_size"] != 1
        or args["ring_degree"] != 1
        or args["component_attention_backends"]["transformer"] != backend
        or args["minimax_h3_adaln_host_cache_gb"] != 2
        or args.get("component_residency") != {
            "video_vae": "resident",
            **case.get("component_residency", {}),
        }
    ):
        raise ValueError(
            "Unexpected GPU placement, parallelism, backend, cache capacity or residency"
        )
    if n > 1 and (
        not args["minimax_h3_ulysses_strict"] or "fell back before exchange" in text
    ):
        raise ValueError("Expected strict quantized Sage A2A without fallback")
    if n > 1 and mode == "nccl" and "input a2a=nccl" not in text:
        raise ValueError("Missing quantized NCCL A2A marker")
    if mode in ("rdma", "pcie"):
        inventory = {
            int(row[0]): pci_key(row[1])
            for row in csv.reader(
                (directory / "gpu-before.txt").read_text().splitlines()
            )
        }
        records = [
            json.loads(s) for s in re.findall(r"H3_ULYSSES_ROUTE (\{[^\n\r]+\})", text)
        ]
        if {r["rank"] for r in records} != set(range(n)):
            raise ValueError("Missing per-rank transport diagnostics")
        route = "rdma" if mode == "rdma" else "cuda_p2p"
        for r in records:
            rank = r["rank"]
            if (
                pci_key(r["gpu_pci"]) != inventory[case["gpus"][rank]]
                or r["transport"] != ("rdma" if mode == "rdma" else "p2p")
                or set(r["peer_routes"]) != {str(i) for i in range(n) if i != rank}
                or set(r["peer_routes"].values()) != {route}
                or (mode == "rdma" and r["nic"] != case["nics"][rank])
            ):
                raise ValueError(
                    "Actual GPU, NIC or peer route differs from the selected case"
                )
    validate_nccl(case, directory)
    metrics = summarize(directory, mode, num_gpus=n)
    return dict(
        **{k: metrics[k] for k in METRICS},
        cached_steps=metrics["cached_step_indices"],
        video_sha256=metrics["video_sha256"],
    )


def stats(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    return dict(
        n=len(values),
        mean=mean,
        stdev=sd,
        cv_percent=100 * sd / mean if sd is not None and mean else None,
    )


def source_revisions():
    result = {}
    for name in (".", "sglang-all", "flashinfer"):
        try:
            revision = subprocess.check_output(
                ["git", "-C", str(ROOT / name), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            dirty = subprocess.run(
                [
                    "git",
                    "-C",
                    str(ROOT / name),
                    "diff",
                    "--quiet",
                    "--ignore-submodules=untracked",
                    "HEAD",
                    "--",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            result[name] = dict(commit=revision, tracked_changes=bool(dirty))
        except (OSError, subprocess.CalledProcessError):
            result[name] = dict(commit=None, tracked_changes=None)
    return result


def run(plan, output, config, profile, timeout):
    output.mkdir(parents=True, exist_ok=False)
    frozen = output / "request.json"
    frozen.write_bytes(config.read_bytes())
    (output / "manifest.json").write_text(
        json.dumps(
            dict(
                plan=plan,
                profiling=profile,
                request_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                repositories=source_revisions(),
                sources={
                    p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                    for p in (
                        "integration/benchmark_high.py",
                        "integration/test_high_sglang.sh",
                        "integration/profile_high_sglang.sh",
                        "integration/warmup_generate.py",
                        "integration/summarize_high.py",
                        "pixi.toml",
                        "pixi.lock",
                    )
                },
            ),
            indent=2,
        )
        + "\n"
    )
    rows = []

    def save():
        groups = {t["case"]["name"] for t in plan}
        summary = dict(
            complete=len(rows) == len(plan),
            completed=len(rows),
            expected=len(plan),
            profiling=profile,
            cases={
                name: {
                    k: stats([r[k] for r in rows if r["name"] == name]) for k in METRICS
                }
                for name in sorted(groups)
                if any(r["name"] == name for r in rows)
            },
        )
        (output / "trials.json").write_text(json.dumps(rows, indent=2) + "\n")
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    save()
    for t in plan:
        case, number = t["case"], t["repeat"]
        target = (
            output / case["name"]
            if profile
            else output / f"repeat-{number:02d}" / case["name"]
        )
        target.mkdir(parents=True)
        print(f"case={case['name']} repeat={number} profiling={profile}", flush=True)
        try:
            gpu_snapshot(case, target, "before")
            before = memory()
            start = time.monotonic()
            env = environment(case, target, frozen, profile)
            recorded = (
                "MODEL_PATH",
                "MODEL_REVISION",
                "NCCL_IB_HCA",
                "NCCL_SOCKET_IFNAME",
                "NCCL_DEBUG",
                "NCCL_DEBUG_SUBSYS",
                "NCCL_NET_DISABLE_INTRA",
                "FLASHINFER_ULYSSES_PCIE_NICS",
                "FLASHINFER_ULYSSES_PCIE_ALLOW_SHARED_NICS",
            )
            (target / "launch.json").write_text(
                json.dumps(
                    dict(
                        command=command(case, profile),
                        environment={k: env[k] for k in recorded if k in env},
                    ),
                    indent=2,
                )
                + "\n"
            )
            try:
                with (target / "process.log").open("w") as log:
                    subprocess.run(
                        [
                            "timeout",
                            "--signal=TERM",
                            "--kill-after=60s",
                            str(timeout),
                            *command(case, profile),
                        ],
                        cwd=ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            finally:
                balance = memory_check(before, memory())
                (target / "memory.json").write_text(
                    json.dumps(balance, indent=2) + "\n"
                )
            gpu_snapshot(case, target, "after")
            # These counters cover every user on the host, not just this trial.
            if not balance["same_boot"]:
                raise RuntimeError("Host reboot during trial; stop")
            row = dict(
                **case,
                repeat=number,
                **validate(case, target),
                trial_wall_s=time.monotonic() - start,
                artifact_directory=str(target),
            )
            previous = [r for r in rows if r["name"] == case["name"]]
            if previous and row["cached_steps"] != previous[0]["cached_steps"]:
                raise ValueError("Cache-hit pattern changed across repetitions")
            if profile:
                report = target / (case["name"] + ".nsys-rep")
                if not report.is_file() or not report.stat().st_size:
                    raise ValueError("Missing final nsys report")
                row.update(report=str(report), report_bytes=report.stat().st_size)
            rows.append(row)
            save()
        except Exception as exc:
            (target / "failure.txt").write_text(f"{type(exc).__name__}: {exc}\n")
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    p.add_argument(
        "--case", action="append", help="Select a case; repeat for multiple cases"
    )
    p.add_argument("--nics", help="Comma-separated device names for NIC slots 0,1,...")
    p.add_argument(
        "--repeats",
        type=int,
        help="Override defaults: eight for 8 GPUs, four otherwise",
    )
    p.add_argument(
        "--profile", action="store_true", help="One nsys capture per selected case"
    )
    p.add_argument("--execute", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--trial-timeout", type=int, default=1800)
    a = p.parse_args()
    try:
        if a.repeats is not None and a.repeats < 1 or a.trial_timeout < 1:
            raise ValueError("Repeats and timeout must be positive")
        cases = load_cases(a.cases)
        if a.case:
            unknown = set(a.case) - {c["name"] for c in cases}
            if unknown:
                raise ValueError("Unknown cases: " + ",".join(sorted(unknown)))
            cases = [c for c in cases if c["name"] in a.case]
        devices = a.nics.split(",") if a.nics else []
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", x) for x in devices):
            raise ValueError("Invalid NIC device name")
        bind_nics(cases, devices, a.execute)
        plan = make_plan(cases, a.repeats, a.profile)
        if not a.execute:
            print(
                json.dumps(
                    dict(trials=len(plan), profiling=a.profile, plan=plan), indent=2
                )
            )
            return
        if not a.output:
            raise ValueError("--execute requires --output")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            visible_ids = visible.split(",")
            if visible_ids != [str(i) for i in range(len(visible_ids))] or any(
                g >= len(visible_ids) for t in plan for g in t["case"]["gpus"]
            ):
                raise ValueError(
                    "CUDA_VISIBLE_DEVICES remaps or hides requested physical GPU indices"
                )
        config = a.config.resolve()
        request = json.loads(config.read_text())
        if type(request.get("seed")) is not int:
            raise ValueError("Request must specify a fixed integer seed")
        run(plan, a.output.resolve(), config, a.profile, a.trial_timeout)
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
