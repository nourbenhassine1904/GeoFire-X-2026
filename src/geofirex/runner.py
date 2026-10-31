"""Common B0/B1/B2/B3 runner. No network acquisition; test is a separate phase.

Checkpoints are trusted local Python/PyTorch files, not a format for untrusted
uploads. Resume forks a new run from the last completed epoch (never overwrites).
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from geofirex.benchmark_metrics import rank_architectures, rank_validation_candidates
from geofirex.benchmark_models import make_model
from geofirex.corpus import sha256_file
from geofirex.runner_data import FoldData, TemporalStore, TrainTiles, digest, fit_fold_normalizer
from geofirex.runner_evaluation import evaluate
from geofirex.unet import SegmentationLoss
from geofirex.unet_data import Normalizer

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "protocol_version": "annual-v2-runner", "dataset_version": "WildfireSpreadTS v1.0 / zenodo.8006177",
    "feature_policy": "conservative_fire_terrain", "qualification": "diagnostic",
    "model": "B1", "fold": 0, "context": 3, "seed": 42, "device": "cpu", "amp": False,
    "cpu_threads": 2, "workers": 0, "reader_cache_mb": 64, "patch_size": 64, "batch_size": 8, "eval_batch_size": 8,
    "epochs": 1, "samples_per_epoch": 32, "learning_rate": .001, "weight_decay": .0001,
    "dice_weight": .5, "pos_weight": "auto", "max_pos_weight": 100., "gradient_clip": 1.,
    "normalization_clip": 8., "model_kwargs": {}, "thresholds": [.01, .05, .1, .2, .3, .5, .7, .9],
    "preflight_steps": 30, "preflight_lr": .01, "data_root": str(ROOT),
    "manifest": "data/manifests/real/manifest_wildfirespreadts.csv",
    "split": "splits/benchmark/annual_v1/fold_00.csv",
    "index": "data/manifests/real/corpus_zip_index_v1.json", "output_root": "artifacts/benchmark",
    "mlflow_enabled": True, "experiment": "geofirex-benchmark-diagnostic-v2",
    "tracking_uri": "", "artifact_root": "", "storage_policy": "ephemeral_export",
    "data_roots": [], "positive_tile_fraction": 0., "early_stopping_patience": 0,
    "early_stopping_min_delta": 0., "checkpoint_dataset": "", "checkpoint_every_epochs": 1,
    "checkpoint_upload_budget_bytes": 0, "checkpoint_private_free_bytes": 0,
    "feature_availability": "",
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def load_config(path, overrides=None):
    supplied = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if set(supplied)-DEFAULTS.keys():
        raise ValueError(f"Unknown configuration keys: {set(supplied)-DEFAULTS.keys()}")
    c = {**DEFAULTS, **supplied, **(overrides or {})}
    from geofirex.feature_profiles import PROFILES
    if c["feature_policy"] not in PROFILES or c["protocol_version"] != "annual-v2-runner":
        raise ValueError("Unsupported feature policy/protocol")
    if c["model"] not in {"B0", "B1", "B2", "B3"} or c["context"] not in {1, 3, 5}:
        raise ValueError("Models B0..B3 and contexts 1/3/5 only")
    if c["qualification"] not in {"diagnostic", "pilot", "scientific"}:
        raise ValueError("Invalid qualification")
    for key in ("epochs", "samples_per_epoch", "batch_size", "eval_batch_size", "patch_size",
                "cpu_threads", "preflight_steps"):
        if not isinstance(c[key], int) or isinstance(c[key], bool) or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["workers"] < 0 or c["seed"] < 0 or not 0 <= c["fold"] < 12:
        raise ValueError("Invalid workers/seed/fold")
    if not np.isfinite(c["reader_cache_mb"]) or c["reader_cache_mb"] < 0:
        raise ValueError("reader_cache_mb must be nonnegative")
    if c["model"] == "B2" and c["patch_size"] < 2**c["model_kwargs"].get("depth", 3):
        raise ValueError("Patch smaller than U-Net depth")
    if not c["thresholds"] or any(not 0 < t < 1 for t in c["thresholds"]):
        raise ValueError("Threshold grid must lie inside (0,1)")
    for key in ("learning_rate", "max_pos_weight", "gradient_clip", "normalization_clip", "preflight_lr"):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not 0 <= c["dice_weight"] <= 1 or c["weight_decay"] < 0:
        raise ValueError("Invalid loss/weight decay")
    if c["pos_weight"] != "auto" and (not np.isfinite(c["pos_weight"]) or c["pos_weight"] <= 0):
        raise ValueError("pos_weight must be auto or positive")
    if c["storage_policy"] not in {"persistent", "ephemeral_export"}:
        raise ValueError("Declare persistent or ephemeral_export storage")
    if not 0 <= c["positive_tile_fraction"] <= 1:
        raise ValueError("positive_tile_fraction must be in [0,1]")
    for key in ("early_stopping_patience", "checkpoint_upload_budget_bytes", "checkpoint_private_free_bytes"):
        if not isinstance(c[key], int) or isinstance(c[key], bool) or c[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if (not np.isfinite(c["early_stopping_min_delta"]) or c["early_stopping_min_delta"] < 0
            or not isinstance(c["checkpoint_every_epochs"], int) or c["checkpoint_every_epochs"] < 1):
        raise ValueError("Invalid early stopping/checkpoint interval")
    if c["checkpoint_dataset"] and min(c["checkpoint_upload_budget_bytes"], c["checkpoint_private_free_bytes"]) < 1:
        raise ValueError("Explicit checkpoint upload and private storage quotas required")
    if not isinstance(c["data_roots"], list) or any(not isinstance(p, str) for p in c["data_roots"]):
        raise ValueError("data_roots must be a list of mounted directories")
    if c["qualification"] == "scientific" and "diagnostic" in c["experiment"]:
        raise ValueError("Use a separate scientific MLflow experiment")
    for key in ("data_root", "manifest", "split", "index", "output_root"):
        c[key] = str((ROOT / c[key]).resolve())
    c["data_roots"] = [str((ROOT/p).resolve()) for p in c["data_roots"]]
    if c["feature_availability"]:
        c["feature_availability"] = str((ROOT/c["feature_availability"]).resolve())
    from geofirex.feature_profiles import feature_profile
    feature_profile(c)
    return c


def choose_device(requested):
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto/cpu/cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; PyTorch must retain its CUDA build")
    return "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"


def seed_all(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def hardware(device):
    try:
        import psutil
        ram_bytes = psutil.virtual_memory().total
    except ImportError:
        ram_bytes = None
    return {"python": sys.version, "platform": platform.platform(), "cpu": platform.processor(), "ram_bytes": ram_bytes,
            "logical_cpus": os.cpu_count(), "torch": str(torch.__version__),
            "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if device == "cuda" else None,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}


def snapshot(out):
    files = [ROOT/"pyproject.toml"]
    for name in ("src", "scripts", "configs", "tests"):
        files += [p for p in (ROOT/name).rglob("*") if p.suffix in {".py", ".yaml"} and "__pycache__" not in p.parts]
    hashes = {}
    for p in sorted(files):
        relative = p.relative_to(ROOT)
        dest = out/"source"/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        hashes[relative.as_posix()] = sha256_file(p)
    write_json(out/"source_hashes.json", hashes)
    def git(*args):
        try:
            return subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
                                           cwd=ROOT, text=True, encoding="utf-8", stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    return {"source_hash": digest(hashes), "git_commit": git("rev-parse", "HEAD"),
            "git_branch": git("branch", "--show-current"), "git_status": git("status", "--short")}


def semantic_config(c):
    runtime = {"device", "amp", "cpu_threads", "workers", "output_root", "tracking_uri", "artifact_root",
               "mlflow_enabled", "experiment", "storage_policy", "data_root", "data_roots", "manifest", "split", "index", "reader_cache_mb",
               "checkpoint_dataset", "checkpoint_every_epochs", "checkpoint_upload_budget_bytes", "checkpoint_private_free_bytes", "feature_availability"}
    return {k: v for k, v in c.items() if k not in runtime}


def portable_normalization(norm, data):
    value = norm.to_dict()
    value["fit_paths"] = sorted(data.path_tokens[p] for p in value["fit_paths"])
    return value


def runner_model(c, norm):
    kwargs = dict(c["model_kwargs"])
    if c["model"] == "B0":
        kwargs["fire_channel"] = norm.channel_names.index("fire_J")
    return make_model(c["model"], channels=len(norm.channel_names), context=c["context"], **kwargs)


class EpochBackup:
    """Opt-in official Kaggle upload, bounded across the entire training session."""
    def __init__(self, c, out):
        self.c, self.out = c, out
        self.remaining = c["checkpoint_upload_budget_bytes"]
        self.private_remaining = c["checkpoint_private_free_bytes"]

    def save(self, epoch, *, final=False):
        if not self.c["checkpoint_dataset"] or (not final and epoch % self.c["checkpoint_every_epochs"]):
            return
        from geofirex.kaggle_persistence import publish_package, session_id, snapshot_training
        if not Path("/kaggle/input").is_dir():
            raise RuntimeError("Checkpoint publication must run inside Kaggle")
        package = self.out.parent/(self.out.name+f"-backup-{epoch:04d}"+("-final" if final else ""))
        db_uri = tracking_uri(self.c)
        db = db_uri[len("sqlite:///"):] if db_uri.startswith("sqlite:///") else None
        snapshot_training(self.out, package, tracking_db=db)
        receipt_path = self.out/(f"backup_{epoch:04d}"+("_final" if final else "")+".json")
        receipt = publish_package(package, self.c["checkpoint_dataset"], receipt_path,
                                  upload_bytes=self.remaining, private_free_bytes=self.private_remaining,
                                  session=session_id(), allow_version=True)
        # Conservatively charge the full version, even if backend deduplicates.
        self.remaining -= receipt["bytes"]
        self.private_remaining -= receipt["bytes"]


def tracking_uri(c):
    return c["tracking_uri"] or "sqlite:///" + (Path(c["output_root"])/"tracking.db").as_posix()


class Tracking:
    def __init__(self, c, out, tags, phase):
        self.enabled, self.out = c["mlflow_enabled"], out
        self.run_id = None
        if not self.enabled:
            return
        import mlflow
        self.m = mlflow
        mlflow.set_tracking_uri(tracking_uri(c))
        location = c["artifact_root"] or (Path(c["output_root"])/"mlflow_artifacts").resolve().as_uri()
        exp = mlflow.get_experiment_by_name(c["experiment"])
        if exp is None:
            exp_id = mlflow.create_experiment(c["experiment"], artifact_location=location)
        else:
            exp_id = exp.experiment_id
            if exp.lifecycle_stage != "active" or exp.artifact_location.rstrip("/") != location.rstrip("/"):
                raise ValueError("Existing MLflow experiment has a different/deleted artifact store; choose a new experiment")
        run = mlflow.start_run(experiment_id=exp_id, run_name=f"{c['model']}-fold{c['fold']:02d}-T{c['context']}-seed{c['seed']}-{out.name}")
        self.run_id = run.info.run_id
        mlflow.set_tags({**tags, "phase": phase, "qualification": c["qualification"],
                         "protocol_version": c["protocol_version"], "model": c["model"],
                         "fold_id": c["fold"], "context_days": c["context"], "seed": c["seed"]})
        mlflow.log_params({k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in c.items()})

    def metrics(self, metrics, step):
        if self.enabled:
            self.m.log_metrics({k: float(v) for k, v in metrics.items() if v is not None and np.isfinite(v)}, step=step)

    def sync(self):
        if self.enabled:
            for p in self.out.rglob("*"):
                if p.is_file() and p.suffix not in {".bin", ".tmp"} and p.name != "tracking.db":
                    parent = p.relative_to(self.out).parent.as_posix()
                    self.m.log_artifact(str(p), artifact_path=None if parent == "." else parent)

    def close(self, failed=False):
        if self.enabled:
            try:
                self.sync()
            finally:
                self.m.end_run(status="FAILED" if failed else "FINISHED")


def preflight(model, tiles, criterion, config, device):
    # One training tile with positives, chosen only for this discarded diagnostic.
    choice = next((t for t in tiles.tiles if (tiles.store.tile(*t, tiles.size)[1]
                                            *tiles.store.tile(*t, tiles.size)[2]).sum()), None)
    if choice is None:
        raise ValueError("No positive train tile for mini-overfit")
    x, y, valid, _ = tiles.store.tile(*choice, tiles.size)
    x, y, valid = [torch.from_numpy(a.copy())[None].to(device) for a in (x, y, valid)]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["preflight_lr"])
    history = []
    model.train()
    for _ in range(config["preflight_steps"]):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y, valid)
        if logits.shape != y.shape or not torch.isfinite(loss):
            raise RuntimeError("Invalid preflight shape/loss")
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Invalid preflight gradients")
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    with torch.inference_mode():
        final = float(criterion(model(x), y, valid).cpu())
    result = {"loss_history": history, "final_loss": final, "coordinate": choice,
              "input_shape": list(x.shape), "passed": final < history[0]}
    return result


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        for i, s in enumerate(state["cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(s.cpu(), device=i)


def new_output(c, phase):
    name = f"{c['model']}-fold{c['fold']:02d}-T{c['context']}-seed{c['seed']}-{phase}-"
    name += datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
    out = Path(c["output_root"])/name
    out.mkdir(parents=True, exist_ok=False)
    return out


def figures(out, history, evaluation_dir, threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if history:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        for key in ("train_loss", "validation_loss"):
            axes[0].plot([r["epoch"] for r in history], [r.get(key) for r in history], marker="o", label=key)
        axes[0].legend()
        axes[1].plot([r["epoch"] for r in history], [r["validation_average_precision"] for r in history], marker="o")
        axes[1].set_ylabel("Validation full-scene micro AP")
        fig.tight_layout()
        fig.savefig(out/"curves.png")
        plt.close(fig)
    example = np.load(evaluation_dir/"example_tile.npz")
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for ax, key in zip(axes[:3], ("observed", "target", "probability")):
        ax.imshow(np.where(example["valid"][0], example[key][0], np.nan), vmin=0, vmax=1)
        ax.set_title(key)
    axes[3].imshow(np.where(example["valid"][0], example["probability"][0] >= threshold, np.nan), vmin=0, vmax=1)
    axes[3].set_title(f"threshold={threshold:g}")
    fig.suptitle("First deterministic tile: diagnostic illustration")
    fig.tight_layout()
    fig.savefig(out/"example.png")
    plt.close(fig)


def train(c, resume=None):
    start = time.perf_counter()
    device = choose_device(c["device"])
    torch.set_num_threads(c["cpu_threads"])
    seed_all(c["seed"])
    data = FoldData(c)
    out = new_output(c, "train")
    print(f"Output: {out}\nDevice: {device}", flush=True)
    write_json(out/"config.json", c)
    provenance = {**snapshot(out), **hardware(device)}
    write_json(out/"provenance.json", provenance)
    audit = data.audit({"train", "val"})
    write_json(out/"data_audit.json", audit)
    for key in ("manifest", "split", "index"):
        shutil.copy2(c[key], out/(key+Path(c[key]).suffix))
    from geofirex.feature_profiles import availability_hash, feature_profile
    profile = feature_profile(c)
    norm = fit_fold_normalizer(data.groups["train"][5], clip=c["normalization_clip"],
                               continuous=profile["continuous"], landcover=profile["landcover"])
    write_json(out/"features.json", {"policy": c["feature_policy"], "channels": norm.channel_names,
                                    "availability_sha256": availability_hash(c)})
    write_json(out/"normalization.json", norm.to_dict())
    stores = {role: TemporalStore(data.windows(role), norm, data.groups[role][5], int(c["reader_cache_mb"]*1024**2))
              for role in ("train", "val")}
    tiles = TrainTiles(stores["train"], c["patch_size"], c["samples_per_epoch"], c["seed"], c["positive_tile_fraction"])
    prior = tiles.prior()
    weight = min(prior["negative_positive_ratio"], c["max_pos_weight"]) if c["pos_weight"] == "auto" else c["pos_weight"]
    prior["effective_pos_weight"] = weight
    write_json(out/"train_prior.json", prior)
    norm_identity = portable_normalization(norm, data)
    identity = digest({"config": semantic_config(c), "data": audit, "normalization": norm_identity,
                       "availability_sha256": availability_hash(c),
                       "source_hash": provenance["source_hash"]})
    model = runner_model(c, norm).to(device)
    criterion = SegmentationLoss(c["dice_weight"], weight).to(device)
    tracking = Tracking(c, out, {"identity": identity, "git_commit": provenance["git_commit"],
                                "dataset_hash": data.hashes["manifest"], "source_hash": provenance["source_hash"]}, "train")
    write_json(out/"tracking.json", {"run_id": tracking.run_id})
    history, best_ap, best_epoch, start_epoch = [], -1., 0, 0
    backup = EpochBackup(c, out)
    failed = True
    try:
        if c["model"] == "B0":
            if resume:
                raise ValueError("B0 has no training to resume")
            metrics = evaluate(model, stores["val"], c, device, out/"validation_0000")
            write_json(out/"validation_0000.json", metrics)
            best_ap, best_epoch = metrics["micro"]["average_precision"], 0
            checkpoint = out/"epoch_0000.pt"
            torch.save({"model": model.state_dict(), "epoch": 0, "identity": identity}, checkpoint)
        else:
            if resume is None:
                result = preflight(model, tiles, criterion, c, device)
                write_json(out/"preflight.json", result)
                if not result["passed"]:
                    raise RuntimeError("Mini-overfit did not reduce train-tile loss; inspect preflight.json")
            seed_all(c["seed"])
            model = runner_model(c, norm).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
            amp = bool(c["amp"] and device == "cuda")
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
            if resume:
                state = torch.load(resume, map_location="cpu", weights_only=False)
                if state["identity"] != identity:
                    raise ValueError("Resume source/config/data/normalization identity differs")
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                if amp and state["scaler"]:
                    scaler.load_state_dict(state["scaler"])
                history, best_ap, best_epoch = state["history"], state["best_ap"], state["best_epoch"]
                start_epoch = state["epoch"]
                restore_rng(state["rng"])
                parent = Path(resume).parent
                best = parent/f"epoch_{best_epoch:04d}.pt"
                shutil.copy2(best, out/best.name)
                # The selected validation outputs remain reviewable in the fork.
                shutil.copy2(parent/f"validation_{best_epoch:04d}.json", out/f"validation_{best_epoch:04d}.json")
                shutil.copytree(parent/f"validation_{best_epoch:04d}", out/f"validation_{best_epoch:04d}")
                write_json(out/"resume.json", {"parent_checkpoint": str(Path(resume).resolve()),
                                              "sha256": sha256_file(resume), "restart_epoch": start_epoch+1})
            generator = torch.Generator().manual_seed(c["seed"])
            loader = DataLoader(tiles, batch_size=c["batch_size"], shuffle=False,
                                num_workers=c["workers"], generator=generator, pin_memory=device == "cuda")
            for epoch in range(start_epoch+1, c["epochs"]+1):
                epoch_start = time.perf_counter()
                model.train()
                tiles.epoch = epoch
                write_json(out/f"sampling_{epoch:04d}.json", [tiles.coordinate(i) for i in range(len(tiles))])
                total = pixels = seen_positive = 0.
                optimizer_updates = 0
                train_io_seconds = train_compute_seconds = 0.
                covered = set()
                batch_ready = time.perf_counter()
                for x, y, valid in loader:
                    train_io_seconds += time.perf_counter()-batch_ready
                    compute_start = time.perf_counter()
                    x, y, valid = [a.to(device, non_blocking=True) for a in (x, y, valid)]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device, enabled=amp, dtype=torch.float16):
                        # Keep masked reductions in FP32, including large patches.
                        loss = criterion(model(x).float(), y, valid)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), c["gradient_clip"], error_if_nonfinite=True)
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_updates += int(scaler.get_scale() >= previous_scale)
                    n = float(valid.sum().detach().cpu())
                    total += float(loss.detach().cpu())*n
                    pixels += n
                    seen_positive += float((y*valid).sum().detach().cpu())
                    train_compute_seconds += time.perf_counter()-compute_start
                    batch_ready = time.perf_counter()
                if not pixels:
                    raise ValueError("Training sampled no valid pixels")
                for i in range(len(tiles)):
                    covered.add(stores["train"].windows[tiles.coordinate(i)[0]].event_id)
                training_seconds = time.perf_counter()-epoch_start
                metrics = evaluate(model, stores["val"], c, device, out/f"validation_{epoch:04d}", criterion=criterion)
                write_json(out/f"validation_{epoch:04d}.json", metrics)
                ap = metrics["micro"]["average_precision"]
                if ap is None:
                    raise ValueError("Undefined validation AP")
                stale_epochs = (history[-1].get("stale_epochs", 0) if history else 0)
                early_reference = history[-1].get("early_stopping_reference_ap", -1.) if history else -1.
                if ap > early_reference+c["early_stopping_min_delta"]:
                    early_reference, stale_epochs = ap, 0
                else:
                    stale_epochs += 1
                if ap > best_ap:
                    best_ap, best_epoch = ap, epoch
                row = {"epoch": epoch, "train_loss": total/pixels, "validation_loss": metrics["validation_loss"],
                       "validation_average_precision": ap, "train_seconds": training_seconds,
                       "epoch_seconds": time.perf_counter()-epoch_start, "seen_positive_pixels": seen_positive,
                       "seen_valid_pixels": pixels, "seen_prevalence": seen_positive/pixels,
                       "events_covered": len(covered), "stale_epochs": stale_epochs,
                       "early_stopping_reference_ap": early_reference,
                       "optimizer_updates": optimizer_updates, **metrics["timings"]}
                sampled = {tiles.coordinate(i)[0] for i in range(len(tiles))}
                row["observations_covered"] = len({p for i in sampled for p in
                    (*stores["train"].domain_windows[i].context_paths, stores["train"].domain_windows[i].target_path)})
                row["model_input_observations_covered"] = len({p for i in sampled
                    for p in stores["train"].windows[i].context_paths})
                row.update({"train_io_seconds": train_io_seconds, "train_compute_seconds": train_compute_seconds})
                history.append(row)
                state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                         "rng": rng_state(), "epoch": epoch, "best_ap": best_ap, "best_epoch": best_epoch,
                         "history": history, "identity": identity}
                # Atomic publication: partial checkpoints never become resume candidates.
                temporary = out/f"epoch_{epoch:04d}.pt.tmp"
                torch.save(state, temporary)
                temporary.rename(out/f"epoch_{epoch:04d}.pt")
                tracking.metrics(row, epoch)
                tracking.metrics({"validation_dice": metrics["micro"]["dice"],
                                  "validation_new_ignition_ap": metrics["new_ignitions"]["average_precision"]}, epoch)
                tracking.sync()
                backup.save(epoch)
                print(f"Epoch {epoch}: train_loss={row['train_loss']:.6f}; val_micro_AP={ap:.6f}", flush=True)
                if c["early_stopping_patience"] and stale_epochs >= c["early_stopping_patience"]:
                    break
            checkpoint = out/f"epoch_{best_epoch:04d}.pt"
            metrics = json.loads((out/f"validation_{best_epoch:04d}.json").read_text(encoding="utf-8"))
        summary = {"status": "FINISHED", "run_id": tracking.run_id or out.name, "model": c["model"],
                   "context_days": c["context"], "seed": c["seed"], "fold_id": c["fold"],
                   "protocol_version": c["protocol_version"], "dataset_hash": data.hashes["manifest"],
                   "evaluation_target_hash": data.hashes["targets"], "feature_policy": c["feature_policy"],
                   "qualification": c["qualification"], "identity": identity,
                   "comparison_hash": digest({"config": {k: v for k, v in semantic_config(c).items()
                                                         if k not in {"model", "model_kwargs", "context", "seed"}},
                                              "data": data.hashes, "normalization": norm_identity,
                                              "source_hash": provenance["source_hash"]}),
                   "best_epoch": best_epoch, "checkpoint": checkpoint.name,
                   "checkpoint_sha256": sha256_file(checkpoint), "threshold": metrics["threshold"],
                   "validation_average_precision": best_ap, "history": history,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "optimizer_updates": sum(r["optimizer_updates"] for r in history),
                   "corpus_counts": audit["corpus_counts"], "complete_corpus": audit["complete_corpus"],
                   "early_stopped": bool(history and history[-1]["epoch"] < c["epochs"]),
                   "seconds": time.perf_counter()-start}
        write_json(out/"summary.json", summary)
        figures(out, history, out/f"validation_{best_epoch:04d}", metrics["threshold"])
        tracking.metrics({"validation_average_precision": best_ap, "total_seconds": summary["seconds"],
                          "parameters": summary["parameters"]}, best_epoch)
        failed = False
    finally:
        tracking.close(failed)
    backup.save(history[-1]["epoch"] if history else 0, final=True)
    return out


LOCK_FILES = ("summary.json", "config.json", "normalization.json", "data_audit.json", "source_hashes.json", "features.json")


def lock_selection(run_dirs, destination, expected_seeds=(42, 43, 44)):
    dirs = [Path(p).resolve() for p in run_dirs]
    summaries = [json.loads((p/"summary.json").read_text(encoding="utf-8")) for p in dirs]
    rank_validation_candidates(summaries)
    if len(dirs) != len(set(dirs)) or len({s["run_id"] for s in summaries}) != len(summaries):
        raise ValueError("Duplicate selection candidate")
    for p, s in zip(dirs, summaries):
        if s["status"] != "FINISHED" or sha256_file(p/s["checkpoint"]) != s["checkpoint_sha256"]:
            raise ValueError("Candidate incomplete or checkpoint modified")
        validation = json.loads((p/f"validation_{s['best_epoch']:04d}.json").read_text(encoding="utf-8"))
        if (validation["micro"]["average_precision"] != s["validation_average_precision"]
                or validation["threshold"] != s["threshold"]):
            raise ValueError("Summary differs from selected validation metrics")
        config = json.loads((p/"config.json").read_text(encoding="utf-8"))
        if config["mlflow_enabled"]:
            from mlflow.tracking import MlflowClient
            if MlflowClient(tracking_uri=tracking_uri(config)).get_run(s["run_id"]).info.status != "FINISHED":
                raise ValueError("Only FINISHED MLflow runs may be selected")
    if len({s["comparison_hash"] for s in summaries}) != 1:
        raise ValueError("Candidates have different budgets, data or normalization")
    scientific = summaries[0]["qualification"] == "scientific"
    # Persistence has no optimization/seed variability. Evaluate it once and
    # duplicate its deterministic validation score for the seed aggregation only.
    ranking_candidates = []
    for s in summaries:
        ranking_candidates.extend([{**s, "seed": seed} for seed in expected_seeds] if s["model"] == "B0" else [s])
    ranking = rank_architectures(ranking_candidates, expected_seeds) if scientific or len(summaries) > 1 else None
    selected = ranking[0]["run_ids"] if ranking else [summaries[0]["run_id"]]
    entries = []
    for p, s in zip(dirs, summaries):
        if s["run_id"] not in selected:
            continue
        files = {name: sha256_file(p/name) for name in (*LOCK_FILES, s["checkpoint"],
                                                       f"validation_{s['best_epoch']:04d}.json")}
        entries.append({"run_dir": str(p), "files": files})
    payload = {"schema_version": 1, "qualification": summaries[0]["qualification"],
               "selection": "validation_only", "ranking": ranking, "expected_seeds": list(expected_seeds),
               "b0_seed_policy": "one deterministic run, repeated score only for seed aggregation",
               "candidates": [{"run_id": s["run_id"], "validation_average_precision": s["validation_average_precision"],
                               "model": s["model"], "context_days": s["context_days"], "seed": s["seed"],
                               "seconds": s["seconds"]} for s in summaries], "selected": entries}
    payload["lock_hash"] = digest(payload)
    write_json(destination, payload)
    return payload


def final_test(run_dir, lock_path):
    run_dir = Path(run_dir).resolve()
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    lock_hash = lock.pop("lock_hash")
    if digest(lock) != lock_hash or lock["selection"] != "validation_only":
        raise ValueError("Invalid selection lock")
    entry = next((e for e in lock["selected"] if Path(e["run_dir"]).resolve() == run_dir), None)
    if entry is None or any(sha256_file(run_dir/name) != h for name, h in entry["files"].items()):
        raise ValueError("Run is not selected or a locked artifact was modified")
    c = json.loads((run_dir/"config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir/"summary.json").read_text(encoding="utf-8"))
    device = choose_device(c["device"])
    torch.set_num_threads(c["cpu_threads"])
    seed_all(c["seed"])
    data = FoldData(c)
    old_audit = json.loads((run_dir/"data_audit.json").read_text(encoding="utf-8"))
    if data.hashes != old_audit["hashes"]:
        raise ValueError("Manifest/split/index/targets changed since selection")
    if not data.windows("test"):
        raise ValueError("No aligned test windows; diagnostic subset cannot support this fold's final test")
    # Include prior fingerprints to catch identical train/test files, without re-reading train.
    data.fingerprints = old_audit["raster_fingerprints"]
    audit = data.audit({"test"})
    norm = Normalizer(**json.loads((run_dir/"normalization.json").read_text(encoding="utf-8")))
    store = TemporalStore(data.windows("test"), norm, data.groups["test"][5], int(c["reader_cache_mb"]*1024**2))
    model = runner_model(c, norm).to(device)
    state = torch.load(run_dir/summary["checkpoint"], map_location="cpu", weights_only=False)
    if state["identity"] != summary["identity"]:
        raise ValueError("Checkpoint identity mismatch")
    # Check executable source against the source that produced the selection.
    source_hashes = json.loads((run_dir/"source_hashes.json").read_text(encoding="utf-8"))
    if any(sha256_file(ROOT/name) != h for name, h in source_hashes.items()):
        raise ValueError("Source changed after training; use the saved source snapshot for final test")
    model.load_state_dict(state["model"])
    out = new_output(c, "final_test")
    write_json(out/"config.json", c)
    write_json(out/"test_audit.json", audit)
    write_json(out/"selection_lock.json", {**lock, "lock_hash": lock_hash})
    write_json(out/"provenance.json", hardware(device))
    tracking = Tracking(c, out, {"selection_lock_hash": lock_hash, "parent_run_id": summary["run_id"]}, "final_test")
    failed = True
    try:
        metrics = evaluate(model, store, c, device, out/"test", threshold=summary["threshold"])
        write_json(out/"test_metrics.json", metrics)
        figures(out, [], out/"test", summary["threshold"])
        tracking.metrics({"test_average_precision": metrics["micro"]["average_precision"],
                          **{f"test_{k}": metrics["micro"][k] for k in ("dice", "iou", "precision", "recall", "fp", "fn")}}, 0)
        failed = False
    finally:
        tracking.close(failed)
    return out


def export_run(run_dir, destination):
    """Portable run bundle; omit regenerable AP sorting scratch (.bin)."""
    run_dir = Path(run_dir).resolve()
    destination = Path(destination).resolve()
    if destination == run_dir or run_dir in destination.parents:
        raise ValueError("Export archive must be outside the run directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as f, tarfile.open(fileobj=f, mode="w:gz") as archive:
        for p in sorted(run_dir.rglob("*")):
            if p.is_file() and p.suffix not in {".bin", ".tmp"}:
                archive.add(p, arcname=str(Path(run_dir.name)/p.relative_to(run_dir)), recursive=False)
    write_json(destination.with_suffix(destination.suffix+".sha256.json"), {"sha256": sha256_file(destination)})
    return destination
