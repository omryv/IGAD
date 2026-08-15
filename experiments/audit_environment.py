"""
experiments/audit_environment.py

Part 0 -- the hard gate, executed rather than asserted.

The brief lists five preconditions for any real 3D benchmark:

    1. a runnable 3D generator with actual MoE routing
    2. a real checkpoint
    3. access to the router logits or probabilities
    4. a way to generate multiple 3D outputs
    5. at least one quantitative 3D-quality measurement

and one instruction if they are not all met: STOP, and write down precisely
what is missing instead of substituting synthetic routing vectors.

This script decides that question from measurements, not from memory. It
probes the interpreter, the filesystem, the accelerator and the network, then
maps each observation onto the five conditions and prints a verdict. Re-run it
in any candidate environment; a machine that passes is a machine where
`experiments/capture_router_traces.py` becomes writable.

    python -m experiments.audit_environment [--skip-network] [--scan-root /]

Raw output: experiments/results/environment_audit.json
"""

import argparse
import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from experiments._router_common import print_table, save_results

# ─────────────────────────────────────────────────────────────────────────────
# What each condition needs
# ─────────────────────────────────────────────────────────────────────────────

PACKAGES = {
    "generation runtime": ["torch", "jax", "tensorflow"],
    "model plumbing": ["transformers", "diffusers", "safetensors",
                       "huggingface_hub", "accelerate", "einops", "timm"],
    "3D geometry": ["trimesh", "open3d", "pytorch3d", "kaolin", "pymeshlab",
                    "point_cloud_utils", "pyrender", "nvdiffrast", "libigl"],
    "numerics": ["numpy", "scipy", "sklearn", "matplotlib", "mpmath"],
    "image/conditioning": ["PIL", "cv2", "clip", "open_clip"],
}

CHECKPOINT_SUFFIXES = (".safetensors", ".ckpt", ".pth", ".gguf", ".onnx", ".msgpack")
# .pt and .bin are deliberately excluded from the primary scan: Go and Vim ship
# files with those names. They are counted separately and reported.
AMBIGUOUS_SUFFIXES = (".pt", ".bin")
MESH_SUFFIXES = (".obj", ".ply", ".glb", ".gltf", ".stl", ".off", ".usd", ".usdz",
                 ".fbx", ".dae", ".3mf")
SKIP_DIRS = {"/proc", "/sys", "/dev", "/run", "/snap", "/usr/local/go1.25.1",
             "/usr/share/vim", "/usr/lib/go", "/var/lib/docker"}

HOSTS = [
    ("huggingface.co", "https://huggingface.co/api/models?limit=1", "model weights"),
    ("pypi.org", "https://pypi.org/simple/torch/", "python packages"),
    ("files.pythonhosted.org", "https://files.pythonhosted.org/", "python wheels"),
    ("github.com", "https://github.com/", "source checkouts"),
    ("raw.githubusercontent.com", "https://raw.githubusercontent.com/", "source files"),
    ("objaverse (allenai)", "https://huggingface.co/datasets/allenai/objaverse",
     "3D evaluation assets"),
    ("archive.ubuntu.com", "http://archive.ubuntu.com/ubuntu/", "system packages"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Probes
# ─────────────────────────────────────────────────────────────────────────────

def probe_packages():
    found = {}
    for group, names in PACKAGES.items():
        found[group] = {}
        for name in names:
            try:
                spec = importlib.util.find_spec(name)
            except (ImportError, ValueError):
                spec = None
            found[group][name] = None
            if spec is not None:
                version = "present"
                try:
                    mod = importlib.import_module(name)
                    version = getattr(mod, "__version__", "present")
                except Exception as exc:                     # noqa: BLE001
                    version = "import failed: %s" % type(exc).__name__
                found[group][name] = version
    return found


def probe_accelerator():
    out = {"nvidia_smi": shutil.which("nvidia-smi") is not None,
           "rocm_smi": shutil.which("rocm-smi") is not None,
           "nvidia_devices": sorted(p for p in os.listdir("/dev")
                                    if p.startswith("nvidia")) if os.path.isdir("/dev") else [],
           "torch_cuda": None, "torch_mps": None}
    if shutil.which("nvidia-smi"):
        try:
            out["nvidia_smi_output"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as exc:                             # noqa: BLE001
            out["nvidia_smi_output"] = "failed: %s" % exc
    try:
        import torch                                          # noqa: PLC0415
        out["torch_cuda"] = bool(torch.cuda.is_available())
        out["torch_mps"] = bool(getattr(torch.backends, "mps", None)
                                and torch.backends.mps.is_available())
    except Exception:                                         # noqa: BLE001
        pass
    return out


# A suffix is a hint, not a finding. Both of the checks below exist because
# the first version of this audit reported 80 "checkpoints" (uv cache
# .msgpack files, none over 1.7 KB) and 20 "meshes" (Go's ELF linker test
# fixtures, which are named .obj but start with \x7fELF).
MIN_CHECKPOINT_BYTES = 10 * 1024 * 1024
ELF_MAGIC = b"\x7fELF"
MESH_MAGIC = {".ply": (b"ply",), ".stl": (b"solid",), ".glb": (b"glTF",),
              ".gltf": (b"{",), ".3mf": (b"PK",), ".fbx": (b"Kaydara",)}


def probe_filesystem(root, limit=200):
    """Walk `root` for model checkpoints and 3D assets.

    Each hit is classified as `plausible` or `implausible`; only plausible
    hits count toward the gate.
    """
    hits = {"checkpoints": [], "meshes": [], "ambiguous": [],
            "n_checkpoints": 0, "n_meshes": 0, "n_ambiguous": 0,
            "n_checkpoints_plausible": 0, "n_meshes_plausible": 0,
            "scanned_dirs": 0, "truncated": False,
            "min_checkpoint_bytes": MIN_CHECKPOINT_BYTES}
    t0 = time.time()
    for dirpath, dirnames, filenames in os.walk(root, topdown=True,
                                                onerror=lambda e: None):
        if any(dirpath == s or dirpath.startswith(s + os.sep) for s in SKIP_DIRS):
            dirnames[:] = []
            continue
        hits["scanned_dirs"] += 1
        if time.time() - t0 > 120:
            hits["truncated"] = True
            break
        for fn in filenames:
            low = fn.lower()
            path = os.path.join(dirpath, fn)
            if low.endswith(CHECKPOINT_SUFFIXES) or low.endswith(AMBIGUOUS_SUFFIXES):
                ambiguous = low.endswith(AMBIGUOUS_SUFFIXES)
                rec = _stat(path)
                rec["plausible"] = _plausible_checkpoint(rec)
                key = "ambiguous" if ambiguous else "checkpoints"
                hits["n_" + key] += 1
                if rec["plausible"]:
                    hits["n_checkpoints_plausible"] += 1
                if len(hits[key]) < limit:
                    hits[key].append(rec)
            elif low.endswith(MESH_SUFFIXES):
                rec = _stat(path)
                rec["plausible"] = _plausible_mesh(path)
                hits["n_meshes"] += 1
                if rec["plausible"]:
                    hits["n_meshes_plausible"] += 1
                if len(hits["meshes"]) < limit:
                    hits["meshes"].append(rec)
    return hits


def _stat(path):
    try:
        return {"path": path, "bytes": os.path.getsize(path)}
    except OSError:
        return {"path": path, "bytes": None}


def _plausible_checkpoint(rec):
    """Model weights are large. A 1 KB .msgpack is a package-manager cache."""
    return bool(rec["bytes"] and rec["bytes"] >= MIN_CHECKPOINT_BYTES)


def _plausible_mesh(path):
    """Reject files that carry a mesh suffix but are not meshes.

    `.obj` is the worst offender: it is both Wavefront geometry and the
    conventional suffix for a compiler object file.
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096)
    except OSError:
        return False
    if not head or head.startswith(ELF_MAGIC):
        return False
    if ext in MESH_MAGIC:
        return any(head.lstrip().startswith(m) for m in MESH_MAGIC[ext])
    if ext == ".obj":
        # Wavefront: ASCII, and vertex/face/mtllib directives appear early
        try:
            text = head.decode("ascii")
        except UnicodeDecodeError:
            return False
        return any(line.startswith(("v ", "vn ", "vt ", "f ", "mtllib", "o ", "g "))
                   for line in text.splitlines())
    return True


def probe_network(timeout=15):
    results = []
    for name, url, purpose in HOSTS:
        rec = {"host": name, "url": url, "purpose": purpose,
               "reachable": False, "detail": ""}
        try:
            req = urllib.request.Request(url, method="HEAD",
                                         headers={"User-Agent": "igad-audit"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                rec["reachable"] = 200 <= resp.status < 400
                rec["detail"] = "HTTP %d" % resp.status
        except urllib.error.HTTPError as exc:
            rec["detail"] = "HTTP %d" % exc.code
            rec["reachable"] = exc.code < 400
        except (urllib.error.URLError, socket.timeout, OSError) as exc:
            rec["detail"] = "%s: %s" % (type(exc).__name__, exc)
        results.append(rec)
    return results


def probe_resources():
    total, used, free = shutil.disk_usage("/")
    mem_total = mem_avail = None
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    mem_avail = int(line.split()[1]) * 1024
    except OSError:
        pass
    return {"disk_total_bytes": total, "disk_free_bytes": free,
            "mem_total_bytes": mem_total, "mem_available_bytes": mem_avail,
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0], "platform": platform.platform()}


# ─────────────────────────────────────────────────────────────────────────────
# Mapping observations onto the five gate conditions
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_gate(pkgs, accel, fs, net, res):
    have = {n: v for grp in pkgs.values() for n, v in grp.items() if v}
    reachable = {r["host"] for r in net if r["reachable"]}

    conditions = []

    runtime = "torch" in have or "jax" in have or "tensorflow" in have
    conditions.append({
        "id": 1,
        "condition": "a runnable 3D generator with actual MoE routing",
        "met": False,
        "evidence": (
            "no tensor runtime installed (torch/jax/tensorflow all absent)"
            if not runtime else
            "a tensor runtime is present, but no 3D generator package "
            "(diffusers/trellis/hunyuan3d/shap-e) was found"),
        "blocking": ["tensor runtime"] * (not runtime) + ["3D generator code"],
    })

    ckpt = fs["n_checkpoints_plausible"] > 0
    conditions.append({
        "id": 2,
        "condition": "a real checkpoint",
        "met": ckpt,
        "evidence": "%d files carry a weight suffix but only %d are larger than "
                    "%d MiB; the rest are package-manager caches and toolchain "
                    "fixtures, not model weights"
                    % (fs["n_checkpoints"] + fs["n_ambiguous"],
                       fs["n_checkpoints_plausible"],
                       fs["min_checkpoint_bytes"] // (1024 * 1024)),
        "blocking": [] if ckpt else
                    ["no weights on disk",
                     "huggingface.co unreachable" if "huggingface.co" not in reachable
                     else "weights must be downloaded"],
    })

    conditions.append({
        "id": 3,
        "condition": "access to the router logits or probabilities",
        "met": False,
        "evidence": "moot without a model; capturing pre-top-k softmax needs a "
                    "forward hook on an MoE gate module that does not exist here",
        "blocking": ["no MoE module to hook"],
    })

    conditions.append({
        "id": 4,
        "condition": "a way to generate multiple 3D outputs",
        "met": False,
        "evidence": "no generator, and no accelerator: %s"
                    % ("CUDA visible" if accel.get("torch_cuda") else
                       "no nvidia-smi, no /dev/nvidia*, no CUDA-capable torch"),
        "blocking": ["no generator"] + ([] if accel.get("torch_cuda") else ["no GPU"]),
    })

    mesh_libs = [n for n in ("trimesh", "open3d", "pytorch3d", "kaolin",
                             "pymeshlab", "point_cloud_utils") if n in have]
    conditions.append({
        "id": 5,
        "condition": "at least one quantitative 3D-quality measurement",
        "met": bool(mesh_libs),
        "evidence": ("mesh libraries available: %s" % ", ".join(mesh_libs))
                    if mesh_libs else
                    "no mesh library installed; of %d mesh-suffixed files on "
                    "disk, %d survive a header check (the rest are ELF object "
                    "files named .obj)" % (fs["n_meshes"],
                                           fs["n_meshes_plausible"]),
        "blocking": [] if mesh_libs else ["no mesh/geometry library",
                                          "no reference meshes"],
    })

    passed = all(c["met"] for c in conditions)
    return {
        "gate_passed": passed,
        "conditions": conditions,
        "installable": bool({"pypi.org", "files.pythonhosted.org"} & reachable),
        "weights_downloadable": "huggingface.co" in reachable,
        "reachable_hosts": sorted(reachable),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan-root", default="/")
    p.add_argument("--skip-network", action="store_true")
    p.add_argument("--network-timeout", type=float, default=15.0)
    a = p.parse_args()

    print("scanning environment ...")
    pkgs = probe_packages()
    accel = probe_accelerator()
    res = probe_resources()
    fs = probe_filesystem(a.scan_root)
    net = [] if a.skip_network else probe_network(a.network_timeout)
    gate = evaluate_gate(pkgs, accel, fs, net, res)

    print_table("PACKAGES",
                ["group", "package", "status"],
                [[g, n, v or "ABSENT"] for g, d in pkgs.items() for n, v in d.items()])

    print_table("ACCELERATOR / RESOURCES", ["item", "value"], [
        ["nvidia-smi on PATH", accel["nvidia_smi"]],
        ["/dev/nvidia* devices", accel["nvidia_devices"] or "none"],
        ["torch.cuda.is_available()", accel["torch_cuda"]],
        ["cpu count", res["cpu_count"]],
        ["RAM available", "%.1f GiB" % (res["mem_available_bytes"] / 2 ** 30)
         if res["mem_available_bytes"] else "unknown"],
        ["disk free", "%.1f GiB" % (res["disk_free_bytes"] / 2 ** 30)],
    ])

    print_table("FILESYSTEM SCAN (root=%s)" % a.scan_root,
                ["item", "suffix hits", "survive verification"], [
        ["model checkpoints (%s)" % ",".join(CHECKPOINT_SUFFIXES + AMBIGUOUS_SUFFIXES),
         fs["n_checkpoints"] + fs["n_ambiguous"], fs["n_checkpoints_plausible"]],
        ["3D meshes (%s ...)" % ",".join(MESH_SUFFIXES[:4]),
         fs["n_meshes"], fs["n_meshes_plausible"]],
        ["directories scanned", fs["scanned_dirs"], "-"],
        ["scan truncated by time budget", fs["truncated"], "-"],
    ])
    print("  verification: a checkpoint must exceed %d MiB; a mesh must pass a "
          "header check.\n  Suffix hits alone are not evidence -- .msgpack is a "
          "uv cache format and .obj is\n  also a compiler object file.\n"
          % (fs["min_checkpoint_bytes"] // (1024 * 1024)))

    if net:
        print_table("NETWORK", ["host", "purpose", "result", "usable"],
                    [[r["host"], r["purpose"], r["detail"],
                      "yes" if r["reachable"] else "no"] for r in net])

    print_table("PART 0 HARD GATE",
                ["#", "condition", "met", "evidence"],
                [[c["id"], c["condition"], "YES" if c["met"] else "NO",
                  c["evidence"]] for c in gate["conditions"]])

    print("=" * 78)
    if gate["gate_passed"]:
        print("GATE PASSED -- proceed to experiments/capture_router_traces.py")
    else:
        missing = [c["id"] for c in gate["conditions"] if not c["met"]]
        print("GATE FAILED -- conditions %s unmet." % ", ".join(map(str, missing)))
        print("STOP. Do not substitute synthetic routing vectors.")
        print("Acquisition requirements: docs/acquisition_checklist.md")
    print("=" * 78)
    print()

    path = save_results("environment_audit",
                        {"gate": gate, "packages": pkgs, "accelerator": accel,
                         "filesystem": fs, "network": net, "resources": res,
                         "scan_root": a.scan_root})
    print("raw audit written to %s" % path)
    return 0 if gate["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
