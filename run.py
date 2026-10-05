#!/usr/bin/env python
"""Every experiment and figure in this project, behind one command.

    python run.py smoke                      # end-to-end check, run this first
    python run.py hitrate --n-samples 20000
    python run.py designability
    python run.py budget --seconds 600       # the decisive experiment
    python run.py sweep --list               # how many array tasks
    python run.py sweep --task-id 7
    python run.py amortized --family junction
    python run.py eterna --shard 0 --n-shards 10
    python run.py figures
    python run.py summary

Every subcommand writes one self-describing JSON to --out (default results/).
Nothing reads another subcommand's output at runtime except to pick targets,
and that always falls back -- which is what lets them run as independent Slurm
jobs, in any order, sharded across an array.

torch is imported only by the subcommands that need it, so hitrate /
designability / eterna run on a CPU partition with no GPU build installed.
"""
from __future__ import annotations

import argparse
import getpass
import itertools
import json
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np

import rnagfn as R

HARD_FALLBACK = ["junction_d4_s3", "junction_d5_s3", "junction_d3_s3",
                 "chain_m4_s4"]


# --------------------------------------------------------------------------
# Result files
# --------------------------------------------------------------------------
def metadata(args) -> dict:
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": socket.gethostname(), "user": getpass.getuser(),
        "python": platform.python_version(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "args": {k: v for k, v in vars(args).items() if k != "func"},
    }
    try:
        meta["git"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
            text=True, check=True, cwd=os.path.dirname(os.path.abspath(__file__))
        ).stdout.strip()
    except Exception:
        meta["git"] = "unknown"
    try:
        import RNA
        meta["viennarna"] = RNA.__version__
    except ImportError:
        pass
    if "torch" in sys.modules:
        import torch
        meta["torch"] = torch.__version__
        meta["gpu"] = (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else None)
    return meta


def save(payload: dict, args, name: str, tag: str = "") -> None:
    os.makedirs(args.out, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    path = os.path.join(args.out, f"{name}{suffix}.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f"wrote {path}")


def load(out_dir: str, prefix: str) -> list[dict]:
    """Every JSON whose name starts with `prefix`; missing shards are fine."""
    if not os.path.isdir(out_dir):
        return []
    found = []
    for filename in sorted(os.listdir(out_dir)):
        if filename.startswith(prefix) and filename.endswith(".json"):
            with open(os.path.join(out_dir, filename)) as handle:
                found.append(json.load(handle))
    return found


def hit_rate_rows(out_dir: str) -> list[dict]:
    seen, rows = set(), []
    for payload in load(out_dir, "hitrate"):
        for row in payload.get("rows", []):
            if row["target"] not in seen:
                seen.add(row["target"])
                rows.append(row)
    return rows


def rank_hardest(out_dir: str, n: int) -> list[str] | None:
    """Rank by the BEST baseline, not uniform alone: many targets tie at
    exactly zero on uniform, and an arbitrary tiebreak can land on one where
    the Boltzmann sampler still works perfectly well."""
    rows = hit_rate_rows(out_dir)
    if not rows:
        return None
    rows.sort(key=lambda r: r.get("best_baseline", 1.0))
    return [r["target"] for r in rows[:n]]


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def cmd_hitrate(args) -> None:
    """How often is a merely-compatible sequence actually a solution?"""
    targets = R.load_targets(include_taneda=not args.no_taneda,
                             max_length=args.max_length)
    names = sorted(targets)[args.shard::args.n_shards]
    print(f"shard {args.shard}/{args.n_shards}: {len(names)} targets")

    rng = np.random.default_rng(args.seed + args.shard)
    rows, started = [], time.time()

    for name in names:
        structure = targets[name]
        oracle = R.FoldingOracle(structure, workers=args.workers)

        worst = R.soundness_gate(oracle, R.sample_uniform, args.gate_samples, rng)
        if worst < -R.HIT_TOLERANCE:
            print(f"{name}: DROPPED (folder missed the target, {worst:+.3f})")
            continue

        row = {"target": name, "structure": structure,
               "length": len(structure), "pairs": R.n_pairs(structure),
               "source": "Rfam" if name.startswith("taneda") else "synthetic"}

        for key, sampler in R.SAMPLERS.items():
            solutions, drawn, gc_total = [], 0, 0.0
            while drawn < args.n_samples:
                size = min(args.chunk, args.n_samples - drawn)
                seqs = sampler(oracle.table, size, rng)
                hit = oracle.hits(seqs)
                if hit.any():
                    solutions.append(seqs[hit])
                gc_total += float(np.isin(seqs, [R.C, R.G]).mean()) * size
                drawn += size

            found = (np.concatenate(solutions) if solutions
                     else np.zeros((0, oracle.length), dtype=np.int64))
            row[f"{key}_hit"] = len(found) / args.n_samples
            row[f"{key}_distinct"] = int(len(np.unique(found, axis=0))) \
                if len(found) else 0
            row[f"{key}_gc"] = gc_total / args.n_samples

        row["best_baseline"] = max(row[f"{k}_hit"] for k in R.SAMPLERS)
        rows.append(row)
        print(f"{name:<22} " + "  ".join(
            f"{k} {row[f'{k}_hit'] * 100:6.2f}%" for k in R.SAMPLERS))

    R.FoldingOracle.shutdown_pools()
    save({"experiment": "hitrate", "meta": metadata(args),
          "n_samples": args.n_samples, "seconds": time.time() - started,
          "rows": rows}, args, "hitrate",
         args.tag or (f"shard{args.shard:02d}" if args.n_shards > 1 else ""))


def cmd_designability(args) -> None:
    """Are the zero-hit targets actually solvable?

    A 0% hit rate only means something if the target can be solved at all --
    some structures are provably undesignable, and for those 0% is correct.
    Every solution is independently re-verified; the search's own bookkeeping
    is never trusted.
    """
    catalogue = R.load_targets(include_taneda=not args.no_taneda,
                               max_length=args.max_length)
    names = args.targets or rank_hardest(args.out, args.n_targets)
    if names is None:
        print("no hitrate results found; using the built-in hard list")
        names = HARD_FALLBACK
    names = [n for n in names if n in catalogue]

    rng = np.random.default_rng(args.seed)
    rows = []

    for name in names:
        oracle = R.FoldingOracle(catalogue[name], workers=args.workers)
        started = time.time()
        solutions, best_gap = R.adaptive_walk(oracle, rng,
                                              n_walkers=args.n_walkers,
                                              n_steps=args.n_steps)
        reverified = int(oracle.hits(solutions).sum()) if len(solutions) else 0
        elapsed = time.time() - started

        rows.append({"target": name, "structure": catalogue[name],
                     "designable": len(solutions) > 0,
                     "n_distinct": int(len(solutions)),
                     "reverified": reverified, "best_gap": best_gap,
                     "seconds": elapsed,
                     "examples": [R.array_to_sequence(s)
                                  for s in solutions[:5]]})
        verdict = (f"DESIGNABLE -- {len(solutions)} distinct "
                   f"(re-verified {reverified}/{len(solutions)})"
                   if len(solutions)
                   else f"none found; best gap {best_gap:.2f} kcal/mol")
        print(f"{name:<22} {verdict}   [{elapsed:.0f}s]")

    R.FoldingOracle.shutdown_pools()
    save({"experiment": "designability", "meta": metadata(args), "rows": rows},
         args, "designability", args.tag)


def cmd_budget(args) -> None:
    """Compute-matched comparison. The experiment that decides the paper.

    Every method gets the same wall-clock budget, every candidate is verified,
    and solutions are deduplicated -- raw hit counts can be inflated by
    resampling what you already found. Two phases: training inside the budget
    (the number for a one-off design job) and training paid up front (the
    number if the model is reused).
    """
    catalogue = R.load_targets(include_taneda=not args.no_taneda,
                               max_length=args.max_length)
    name = args.target or (rank_hardest(args.out, 1) or [None])[0]
    if name is None:
        raise SystemExit("no --target given and no hitrate results to rank by")
    structure = catalogue[name]

    oracle = R.FoldingOracle(structure, workers=args.workers)
    print(f"target {name} ({len(structure)} nt), {args.seconds:.0f}s per method")
    print(f"methods: {args.methods}\n")

    results, train_seconds = [], None

    for key in args.methods:
        if key not in R.CANDIDATE_GENERATORS:
            continue
        result = R.run_under_budget(
            R.GENERATOR_LABELS[key], R.CANDIDATE_GENERATORS[key], oracle,
            args.seconds, np.random.default_rng(args.seed))
        results.append(result)
        print(f"  {result['method']:<32} "
              f"{result['distinct_solutions']:>7,} distinct")

    if any(m.startswith("gflownet") for m in args.methods):
        import torch
        from model import TrajectoryBalanceTrainer, build_policy, gflownet_generator

        cfg = R.gflownet_from_args(args).replace(log_every=0)

        if "gflownet_training" in args.methods:
            result = R.run_under_budget(
                "GFlowNet (training included)",
                lambda o, r, batch_size=512: gflownet_generator(
                    o, r, cfg, seed=args.seed, workers=args.workers,
                    batch_size=batch_size),
                oracle, args.seconds, np.random.default_rng(args.seed))
            results.append(result)
            print(f"  {result['method']:<32} "
                  f"{result['distinct_solutions']:>7,} distinct")

        if "gflownet_inference" in args.methods:
            torch.manual_seed(args.seed)
            trainer = TrajectoryBalanceTrainer(build_policy(structure, cfg),
                                               cfg, workers=args.workers)
            trainer.oracles[structure] = oracle
            started = time.time()
            trainer.fit(structure)
            train_seconds = time.time() - started
            print(f"  (training took {train_seconds:.0f}s)")

            def trained(o, r, batch_size=512):
                while True:
                    yield trainer.sample(structure, batch_size,
                                         chunk_size=batch_size)

            result = R.run_under_budget("GFlowNet (inference only)", trained,
                                        oracle, args.seconds,
                                        np.random.default_rng(args.seed))
            results.append(result)
            print(f"  {result['method']:<32} "
                  f"{result['distinct_solutions']:>7,} distinct")

    analysis = None
    inference = [r for r in results if r["method"] == "GFlowNet (inference only)"]
    baselines = [r for r in results if not r["method"].startswith("GFlowNet")]
    if inference and baselines and train_seconds is not None:
        analysis = R.break_even(train_seconds, inference[0]["per_second"],
                                max(r["per_second"] for r in baselines))
        if analysis["reachable"]:
            print(f"\nbreak-even at ~{analysis['solutions']:.0f} solutions: "
                  f"beyond that, {train_seconds:.0f}s of training pays off.")
        else:
            print("\nThe trained policy is SLOWER per second than the best "
                  "baseline, so there\nis no break-even on a single target. "
                  "The per-target claim is dead and\namortisation across "
                  "targets is the only remaining story.")

    R.FoldingOracle.shutdown_pools()
    save({"experiment": "budget", "meta": metadata(args), "target": name,
          "structure": structure, "seconds": args.seconds,
          "train_seconds": train_seconds, "break_even": analysis,
          "results": results}, args, "budget", args.tag or name)


def sweep_tasks(args) -> list[tuple[dict, int]]:
    """Spread across the difficulty range rather than taking the n hardest --
    the claim is a crossover, so the easy end has to be represented too."""
    rows = [r for r in hit_rate_rows(args.out)
            if r["length"] <= args.sweep_max_length]
    if not rows:
        raise SystemExit("run `python run.py hitrate` first")
    rows.sort(key=lambda r: r.get("best_baseline", 1.0))
    positions = np.linspace(0, len(rows) - 1, min(args.n_targets, len(rows)))
    chosen = [rows[int(round(p))] for p in positions]
    return list(itertools.product(chosen, args.seeds))


def cmd_sweep(args) -> None:
    """Many targets, many seeds. One target is an anecdote.

    Built for Slurm arrays: --task-id runs ONE (target, seed) pair and writes
    its own shard, so N*M tasks run in parallel.
    """
    tasks = sweep_tasks(args)
    if args.list:
        print(f"{len(tasks)} tasks  ->  #SBATCH --array=0-{len(tasks) - 1}")
        for index, (row, seed) in enumerate(tasks):
            print(f"  {index:>3}  {row['target']:<22} seed {seed}")
        return

    import torch
    from model import TrajectoryBalanceTrainer, build_policy

    cfg = R.gflownet_from_args(args)
    selected = [tasks[args.task_id]] if args.task_id is not None else tasks
    records = []

    for row, seed in selected:
        structure = row["structure"]
        oracle = R.FoldingOracle(structure, workers=args.workers)

        torch.manual_seed(seed)
        trainer = TrajectoryBalanceTrainer(build_policy(structure, cfg), cfg,
                                           workers=args.workers)
        trainer.oracles[structure] = oracle

        started = time.time()
        trainer.fit(structure)
        elapsed = time.time() - started

        draws = trainer.sample(structure, args.eval_samples)
        hit = oracle.hits(draws)
        distinct = int(len(np.unique(draws[hit], axis=0))) if hit.any() else 0

        records.append({"target": row["target"], "structure": structure,
                        "length": row["length"], "seed": seed,
                        "gflownet_hit": float(hit.mean()),
                        "gflownet_distinct": distinct,
                        "best_baseline": row.get("best_baseline"),
                        "uniform_hit": row.get("uniform_hit"),
                        "boltzmann_hit": row.get("boltzmann_hit"),
                        "train_seconds": elapsed})
        print(f"{row['target']:<22} seed {seed}  hit {hit.mean() * 100:6.2f}%  "
              f"distinct {distinct:>5}  [{elapsed:.0f}s]")

    R.FoldingOracle.shutdown_pools()
    tag = args.tag or (f"task{args.task_id:03d}" if args.task_id is not None
                       else "all")
    save({"experiment": "sweep", "meta": metadata(args), "records": records},
         args, "sweep", tag)


def cmd_amortized(args) -> None:
    """One policy, many targets, zero-shot on held-out structures.

    No combinatorial sampler can do this at all, and it is the only argument
    that justifies paying a training cost -- so if `budget` does not favour the
    GFlowNet on a single target, this is where the contribution has to live.
    """
    import torch
    from model import TrajectoryBalanceTrainer, build_policy

    catalogue = R.load_targets(include_taneda=not args.no_taneda,
                               max_length=args.max_length)
    family = sorted(n for n in catalogue if n.startswith(args.family))
    if len(family) < args.holdout + 2:
        raise SystemExit(f"family '{args.family}' has only {len(family)} targets")

    holdout = family[1::3][:args.holdout]
    train_family = [n for n in family if n not in holdout]
    print(f"family '{args.family}': {len(family)} targets")
    print(f"  train:    {train_family}")
    print(f"  held out: {holdout}\n")

    cfg = R.gflownet_from_args(args)
    torch.manual_seed(args.seed)
    trainer = TrajectoryBalanceTrainer(
        build_policy([catalogue[n] for n in family], cfg), cfg,
        workers=args.workers)

    started = time.time()
    history = trainer.fit([catalogue[n] for n in train_family])
    train_seconds = time.time() - started
    print(f"\ntrained across {len(train_family)} targets in {train_seconds:.0f}s")

    if args.checkpoint:
        trainer.save(args.checkpoint)
        print(f"checkpoint -> {args.checkpoint}")

    baselines = {r["target"]: r.get("best_baseline")
                 for r in hit_rate_rows(args.out)}
    rows = []

    for name in family:
        structure = catalogue[name]
        oracle = R.FoldingOracle(structure, workers=args.workers)
        draws = trainer.sample(structure, args.eval_samples)
        hit = oracle.hits(draws)
        distinct = int(len(np.unique(draws[hit], axis=0))) if hit.any() else 0

        rows.append({"target": name, "structure": structure,
                     "split": "holdout" if name in holdout else "train",
                     "amortized_hit": float(hit.mean()),
                     "amortized_distinct": distinct,
                     "best_baseline": baselines.get(name)})
        print(f"{name:<22} {rows[-1]['split']:<8} hit {hit.mean() * 100:6.2f}%  "
              f"distinct {distinct:>5}")

    held = [r for r in rows if r["split"] == "holdout"]
    if held:
        mean_hit = float(np.mean([r["amortized_hit"] for r in held]))
        print(f"\nzero-shot on held-out targets: mean hit {mean_hit * 100:.2f}% "
              f"-- no per-target training was done for those rows.")

    R.FoldingOracle.shutdown_pools()
    save({"experiment": "amortized", "meta": metadata(args),
          "family": args.family, "holdout": holdout,
          "train_family": train_family, "train_seconds": train_seconds,
          "rows": rows, "history": history[-200:]},
         args, "amortized", args.tag or args.family)


def cmd_eterna(args) -> None:
    """Eterna100 with a negative control.

    Puzzles never solved under the unique-MFE criterion are the control: a
    method reporting a solution there is either making a discovery or has a
    bug. Self-correcting, because every claimed solution is re-verified.
    """
    frame = R.eterna100()
    frame = frame[frame["length"] <= args.max_puzzle_length]
    subset = frame.iloc[args.shard::args.n_shards]
    print(f"shard {args.shard}/{args.n_shards}: {len(subset)} puzzles "
          f"({int((subset['solved_umfe'] == 0).sum())} never solved)\n")

    rows = []
    for row in subset.itertuples():
        oracle = R.FoldingOracle(row.structure, workers=args.workers)
        result = R.run_under_budget(
            "adaptive walk", R.candidates_adaptive_walk, oracle, args.seconds,
            np.random.default_rng(args.seed + int(row.puzzle)))
        rows.append({"puzzle": int(row.puzzle), "name": row.name,
                     "length": int(row.length),
                     "known_solvable_umfe": bool(row.solved_umfe),
                     "known_solvable_mfe": bool(row.solved_mfe),
                     "n_distinct": result["distinct_solutions"],
                     "solved_here": result["distinct_solutions"] > 0})
        flag = "" if row.solved_umfe else "   <- NEGATIVE CONTROL"
        print(f"  #{row.puzzle:>3} {row.name[:38]:<40} "
              f"{result['distinct_solutions']:>6} distinct{flag}")

    solvable = [r for r in rows if r["known_solvable_umfe"]]
    unsolved = [r for r in rows if not r["known_solvable_umfe"]]
    breaches = [r for r in unsolved if r["solved_here"]]

    print(f"\nknown-solvable: {sum(r['solved_here'] for r in solvable)}"
          f"/{len(solvable)} solved")
    print(f"negative control: {len(breaches)}/{len(unsolved)} reported solved")
    if breaches:
        print("  Every solution was re-verified, so this is not a scoring "
              "slip. It is\n  either a real finding or a criterion difference: "
              "we accept the target as\n  *an* MFE structure, while "
              "solved_umfe needs the UNIQUE one. Check that\n  before claiming "
              "anything.")
        for r in breaches:
            print(f"    #{r['puzzle']} {r['name']}")

    R.FoldingOracle.shutdown_pools()
    save({"experiment": "eterna", "meta": metadata(args),
          "seconds": args.seconds, "rows": rows}, args, "eterna",
         args.tag or (f"shard{args.shard:02d}" if args.n_shards > 1 else ""))


def cmd_smoke(args) -> None:
    """Fast end-to-end check. Run before submitting anything to a queue.

    Asserts the invariants that have actually broken in this project: the
    folder must never report an MFE above a specific structure's energy, and
    sampled sequences must stay inside the compatible set.
    """
    structure = "(((.(((....))).(((....))).)))"
    started = time.time()
    rng = np.random.default_rng(0)

    def check(label, ok):
        print(f"  [{'ok' if ok else 'FAIL'}] {label}")
        if not ok:
            raise SystemExit(f"smoke test failed: {label}")

    print("structures")
    table = R.pair_table(structure)
    plan = R.GenerationPlan(table)
    check("plan covers every position",
          len(plan) == int((table >= 0).sum()) // 2 + int((table == -1).sum()))

    print("oracle")
    oracle = R.FoldingOracle(structure, workers=args.workers or 2)
    worst = R.soundness_gate(oracle, R.sample_uniform, 64, rng)
    check(f"soundness gate (worst gap {worst:+.4f} >= 0)",
          worst >= -R.HIT_TOLERANCE)

    print("samplers")
    allowed = set(R.CANONICAL_PAIRS)
    for name, sampler in R.SAMPLERS.items():
        seqs = sampler(table, 64, rng)
        pairs = {(int(s[i]), int(s[int(table[i])]))
                 for s in seqs for i in range(len(table)) if table[i] > i}
        check(f"{name} stays inside the compatible set", pairs <= allowed)
    check("stacking table is finite", bool(np.isfinite(R.stacking_table()).all()))

    print("local search")
    solutions, best = R.adaptive_walk(oracle, rng, n_walkers=32, n_steps=40)
    if len(solutions):
        check("every reported solution re-verifies",
              bool(oracle.hits(solutions).all()))
    else:
        print(f"  [--] no solution in 40 steps (best gap {best:.2f}); fine")

    print("budget harness")
    result = R.run_under_budget("boltzmann", R.candidates_boltzmann, oracle,
                                3.0, rng)
    check(f"harness respects its budget (overshoot {result['overshoot']:.1f}s)",
          result["overshoot"] < 0.5 * max(result["elapsed"], 1.0))

    try:
        import torch
        from model import TrajectoryBalanceTrainer, build_policy

        print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}")
        cfg = R.GFlowNetConfig(iterations=3, batch_size=16, d_model=32,
                               n_layers=1, n_heads=2, log_every=0)
        torch.manual_seed(0)
        trainer = TrajectoryBalanceTrainer(build_policy(structure, cfg), cfg,
                                           workers=2, progress=False)
        trainer.oracles[structure] = oracle
        history = trainer.fit(structure)
        check("training produced finite losses",
              all(np.isfinite(r["loss"]) for r in history))

        draws = trainer.sample(structure, 32)
        pairs = {(int(s[i]), int(s[int(table[i])]))
                 for s in draws for i in range(len(table)) if table[i] > i}
        check("policy samples stay inside the compatible set", pairs <= allowed)

        markov = cfg.replace(attention_window=1)
        torch.manual_seed(0)
        other = TrajectoryBalanceTrainer(build_policy(structure, markov),
                                         markov, workers=2, progress=False)
        other.oracles[structure] = oracle
        other.fit(structure)
        check("Markov ablation (attention window 1) runs", True)

        family = list(R.difficulty_ladder().values())[:3]
        torch.manual_seed(0)
        amortized = TrajectoryBalanceTrainer(build_policy(family, cfg), cfg,
                                             workers=2, progress=False)
        amortized.fit(family)
        check("amortised training over several targets runs", True)
    except ImportError:
        print("  [--] torch not installed; GPU path skipped")

    R.FoldingOracle.shutdown_pools()
    print(f"\nall checks passed in {time.time() - started:.1f}s")


# --------------------------------------------------------------------------
# Figures and summary
# --------------------------------------------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
YELLOW, PINK, VIOLET = "#eda100", "#e87ba4", "#4a3aa7"
INK, MUTED = "#0b0b0b", "#52514e"
METHOD_COLOURS = {
    "uniform (RNAblueprint)": BLUE, "Boltzmann (RNARedPrint)": ORANGE,
    "RNAinverse (Hofacker 1994)": YELLOW,
    "adaptive walk (local search)": PINK,
    "GFlowNet (training included)": VIOLET,
    "GFlowNet (inference only)": AQUA,
}


def cmd_figures(args) -> None:
    """Rebuild every figure from whatever results exist. Missing ones are
    skipped, so this works on partial array output."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="paper")
    except ImportError:
        pass

    os.makedirs(args.figures, exist_ok=True)

    def write(fig, name):
        for extension in ("png", "pdf"):
            fig.savefig(os.path.join(args.figures, f"{name}.{extension}"),
                        dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {args.figures}/{name}.png")

    # --- fig1: the collapse -------------------------------------------------
    rows = hit_rate_rows(args.out)
    if rows:
        payloads = load(args.out, "hitrate")
        n_samples = payloads[0].get("n_samples", 2000)
        frame = pd.DataFrame(rows).sort_values("best_baseline").reset_index(drop=True)
        uniform = frame["uniform_hit"].to_numpy() * 100
        boltzmann = frame["boltzmann_hit"].to_numpy() * 100
        floor = 100 / (n_samples * 3)
        y = np.arange(len(frame))

        fig, ax = plt.subplots(figsize=(9, 0.32 * len(frame) + 2.2))
        ax.barh(y - 0.2, np.maximum(uniform, floor), height=0.32, color=BLUE,
                label="uniform (RNAblueprint)", zorder=3)
        ax.barh(y + 0.2, np.maximum(boltzmann, floor), height=0.32,
                color=ORANGE, label="Boltzmann (RNARedPrint)", zorder=3)

        # Zeros cannot be drawn on a log axis; label them rather than drop them.
        for index, (u, b) in enumerate(zip(uniform, boltzmann)):
            for value, offset in ((u, -0.2), (b, 0.2)):
                if value == 0:
                    ax.text(floor * 1.25, index + offset, f"0 / {n_samples:,}",
                            va="center", fontsize=6.5, color=MUTED, zorder=4)

        ax.axvline(1.0, color=MUTED, linestyle=":", linewidth=1, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{r.target}{'  (Rfam)' if r.source == 'Rfam' else ''}"
             for r in frame.itertuples()], fontsize=7.5)
        for tick, source in zip(ax.get_yticklabels(), frame["source"]):
            tick.set_color(INK if source == "Rfam" else MUTED)
            tick.set_fontweight("bold" if source == "Rfam" else "normal")
        ax.set_xscale("log")
        ax.set_xlim(floor * 0.8, 160)
        ax.set_xlabel("compatible samples whose MFE is the target (%, log)")
        ax.set_title("Compatible-set sampling collapses as targets get harder\n"
                     f"ViennaRNA, {n_samples:,} samples per target; "
                     "bold labels are real Rfam structures",
                     fontsize=10, loc="left")
        ax.grid(axis="y", visible=False)
        ax.legend(fontsize=8, loc="lower right")
        write(fig, "fig1_collapse")

    # --- fig2: compute-matched ---------------------------------------------
    for payload in load(args.out, "budget"):
        results = payload["results"]
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

        for result in results:
            timeline = result["timeline"]
            axes[0].step([p[0] for p in timeline], [p[1] for p in timeline],
                         where="post", linewidth=2, label=result["method"],
                         color=METHOD_COLOURS.get(result["method"], MUTED))
        axes[0].set_xlabel("wall-clock seconds")
        axes[0].set_ylabel("distinct verified solutions")
        axes[0].set_title(f"{payload['target']}: solutions per unit compute",
                          loc="left", fontsize=10)
        axes[0].legend(fontsize=7.5, loc="upper left")

        frame = pd.DataFrame([{k: v for k, v in r.items() if k != "timeline"}
                              for r in results]).sort_values("distinct_solutions")
        bars = axes[1].barh(range(len(frame)), frame["distinct_solutions"],
                            height=0.6, zorder=3,
                            color=[METHOD_COLOURS.get(m, MUTED)
                                   for m in frame["method"]])
        axes[1].set_yticks(range(len(frame)))
        axes[1].set_yticklabels(frame["method"], fontsize=7.5)
        axes[1].set_xlabel(f"distinct solutions in {payload['seconds']:.0f}s")
        axes[1].set_title("Final count", loc="left", fontsize=10)
        axes[1].grid(axis="y", visible=False)
        for bar, value in zip(bars, frame["distinct_solutions"]):
            axes[1].text(bar.get_width() * 1.02,
                         bar.get_y() + bar.get_height() / 2, f"{value:,}",
                         va="center", fontsize=7.5, color=MUTED)
        write(fig, f"fig2_budget_{payload['target']}")

    # --- fig3: the crossover ------------------------------------------------
    records = [r for p in load(args.out, "sweep") for r in p.get("records", [])]
    if records:
        frame = pd.DataFrame(records)
        summary = (frame.groupby("target")
                   .agg(best_baseline=("best_baseline", "first"),
                        mean=("gflownet_hit", "mean"),
                        std=("gflownet_hit", "std"),
                        seeds=("seed", "nunique"))
                   .sort_values("best_baseline").reset_index())

        floor = 100 / (args.eval_samples * 3)
        baseline = np.maximum(summary["best_baseline"] * 100, floor)
        gflownet = np.maximum(summary["mean"] * 100, floor)

        fig, ax = plt.subplots(figsize=(6.5, 5.8))
        ax.errorbar(baseline, gflownet, yerr=summary["std"].fillna(0) * 100,
                    fmt="o", markersize=8, capsize=3, color=AQUA,
                    ecolor=MUTED, elinewidth=1, zorder=3)

        # Limits from the data, not a fixed floor -- otherwise most of the
        # panel is empty space below the lowest point.
        low = min(baseline.min(), gflownet.min()) / 2.5
        high = max(baseline.max(), gflownet.max()) * 2.5
        limits = [low, high]
        ax.plot(limits, limits, linestyle="--", color=MUTED, linewidth=1,
                zorder=1)

        # Points bunch up near saturation, so alternate the label side.
        order = np.argsort(baseline.to_numpy())
        side = {int(i): (1 if rank % 2 == 0 else -1)
                for rank, i in enumerate(order)}
        for i, row in summary.iterrows():
            x = max(row["best_baseline"] * 100, floor)
            y = max(row["mean"] * 100, floor)
            offset = (10, 7) if side[int(i)] > 0 else (-10, -15)
            ax.annotate(row["target"], (x, y), textcoords="offset points",
                        xytext=offset, fontsize=7, color=INK,
                        ha="left" if side[int(i)] > 0 else "right")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_xlabel("best compatible-set sampler, hit rate (%)")
        ax.set_ylabel("GFlowNet hit rate (%)")
        ax.set_title(f"Above the line = GFlowNet wins\n"
                     f"mean +/- sd over {int(summary['seeds'].max())} seeds",
                     loc="left", fontsize=10)
        write(fig, "fig3_crossover")
        print(f"GFlowNet ahead on {int((gflownet > baseline).sum())}"
              f"/{len(summary)} targets")


def cmd_summary(args) -> None:
    """Headline numbers from every experiment that has results."""
    import pandas as pd

    def section(title):
        print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")

    rows = hit_rate_rows(args.out)
    if rows:
        frame = pd.DataFrame(rows)
        section("hitrate")
        print(f"{len(frame)} targets; "
              f"{int((frame['best_baseline'] < 0.01).sum())} below 1%; "
              f"{int((frame['best_baseline'] == 0).sum())} with no solution")
        print(frame.nsmallest(8, "best_baseline")[
            ["target", "length", "uniform_hit", "boltzmann_hit"]
        ].to_string(index=False))

    rows = [r for p in load(args.out, "designability") for r in p.get("rows", [])]
    if rows:
        section("designability control")
        print(pd.DataFrame(rows)[["target", "designable", "n_distinct",
                                  "reverified", "seconds"]].to_string(index=False))

    for payload in load(args.out, "budget"):
        section(f"budget ({payload['target']}, {payload['seconds']:.0f}s each)")
        frame = pd.DataFrame([{k: v for k, v in r.items() if k != "timeline"}
                              for r in payload["results"]])
        print(frame[["method", "distinct_solutions", "per_second",
                     "candidates_evaluated"]]
              .sort_values("distinct_solutions", ascending=False)
              .to_string(index=False))
        analysis = payload.get("break_even")
        if analysis:
            print("break-even:",
                  "unreachable (trained policy is slower per second)"
                  if not analysis["reachable"]
                  else f"~{analysis['solutions']:.0f} solutions")

    records = [r for p in load(args.out, "sweep") for r in p.get("records", [])]
    if records:
        section("sweep")
        print(pd.DataFrame(records).groupby("target").agg(
            seeds=("seed", "nunique"), baseline=("best_baseline", "first"),
            gflownet=("gflownet_hit", "mean"), sd=("gflownet_hit", "std"),
        ).sort_values("baseline").to_string())

    for payload in load(args.out, "amortized"):
        section(f"amortisation (family '{payload['family']}')")
        print(pd.DataFrame(payload["rows"])[
            ["target", "split", "amortized_hit", "amortized_distinct",
             "best_baseline"]].to_string(index=False))

    rows = [r for p in load(args.out, "eterna") for r in p.get("rows", [])]
    if rows:
        frame = pd.DataFrame(rows).drop_duplicates("puzzle")
        solvable = frame[frame["known_solvable_umfe"]]
        unsolved = frame[~frame["known_solvable_umfe"]]
        section("Eterna100")
        print(f"known-solvable: {int(solvable['solved_here'].sum())}"
              f"/{len(solvable)} solved")
        print(f"negative control: {int(unsolved['solved_here'].sum())}"
              f"/{len(unsolved)} reported solved")
        if unsolved["solved_here"].any():
            print("  CHECK THESE -- verified solutions on never-solved puzzles:")
            print(unsolved[unsolved["solved_here"]][
                ["puzzle", "name", "n_distinct"]].to_string(index=False))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add(name, function, help_text):
        sub = subparsers.add_parser(name, help=help_text,
                                    description=function.__doc__)
        sub.set_defaults(func=function)
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--out", default="results")
        sub.add_argument("--figures", default="figures")
        sub.add_argument("--tag", default="")
        sub.add_argument("--workers", type=int, default=None,
                         help="folding processes; defaults to $SLURM_CPUS_PER_TASK")
        sub.add_argument("--max-length", type=int, default=90)
        sub.add_argument("--no-taneda", action="store_true")
        sub.add_argument("--eval-samples", type=int, default=2000)
        return sub

    sub = add("smoke", cmd_smoke, "end-to-end check; run this first")

    sub = add("hitrate", cmd_hitrate, "how often a compatible sequence is a solution")
    sub.add_argument("--n-samples", type=int, default=2000)
    sub.add_argument("--gate-samples", type=int, default=200)
    sub.add_argument("--chunk", type=int, default=2000)
    sub.add_argument("--shard", type=int, default=0)
    sub.add_argument("--n-shards", type=int, default=1)

    sub = add("designability", cmd_designability, "are the zero-hit targets solvable")
    sub.add_argument("--n-walkers", type=int, default=256)
    sub.add_argument("--n-steps", type=int, default=400)
    sub.add_argument("--n-targets", type=int, default=4)
    sub.add_argument("--targets", nargs="*", default=None)

    sub = add("budget", cmd_budget, "compute-matched comparison (decisive)")
    sub.add_argument("--seconds", type=float, default=180.0)
    sub.add_argument("--target", default=None)
    sub.add_argument("--methods", nargs="*",
                     default=list(R.CANDIDATE_GENERATORS)
                     + ["gflownet_training", "gflownet_inference"])
    R.add_gflownet_args(sub)

    sub = add("sweep", cmd_sweep, "many targets x many seeds (Slurm array)")
    sub.add_argument("--n-targets", type=int, default=6)
    sub.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    sub.add_argument("--sweep-max-length", type=int, default=60)
    sub.add_argument("--task-id", type=int, default=None)
    sub.add_argument("--list", action="store_true")
    R.add_gflownet_args(sub)

    sub = add("amortized", cmd_amortized, "one policy, zero-shot on held-out")
    sub.add_argument("--family", default="junction")
    sub.add_argument("--holdout", type=int, default=2)
    sub.add_argument("--checkpoint", default=None)
    R.add_gflownet_args(sub)

    sub = add("eterna", cmd_eterna, "Eterna100 + negative control")
    sub.add_argument("--seconds", type=float, default=30.0)
    sub.add_argument("--max-puzzle-length", type=int, default=100)
    sub.add_argument("--shard", type=int, default=0)
    sub.add_argument("--n-shards", type=int, default=1)

    add("figures", cmd_figures, "rebuild all figures from results/")
    add("summary", cmd_summary, "print headline numbers")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
