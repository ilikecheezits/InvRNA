"""GFlowNets for RNA inverse folding -- core library (numpy only).

Deliberately free of torch so CPU jobs need no GPU build: the model lives in
model.py, the only file that imports it.

Contents, in order: constants, configuration, structure parsing and the
generation plan, the parallel folding oracle, target catalogues, the four
baseline samplers, and the compute-matched budget harness.
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import subprocess
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields

import numpy as np

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

import numpy as np

NUCLEOTIDES = "ACGU"
A, C, G, U = 0, 1, 2, 3
N_BASES = 4

# Canonical pairs including the GU wobbles that ViennaRNA models. Excluding
# them shrinks the compatible set and inflates baseline hit rates, so results
# are not comparable across that choice.
CANONICAL_PAIRS: list[tuple[int, int]] = [
    (C, G), (G, C), (A, U), (U, A), (G, U), (U, G),
]
N_PAIR_TYPES = len(CANONICAL_PAIRS)
MAX_ACTIONS = max(N_PAIR_TYPES, N_BASES)

PAIR_FIRST = np.array([p[0] for p in CANONICAL_PAIRS])
PAIR_SECOND = np.array([p[1] for p in CANONICAL_PAIRS])

GAS_CONSTANT_T = 0.6163        # kcal/mol at 37 C

# A sequence is a solution when the target's energy is within this of the
# minimum free energy, i.e. the target is *an* MFE structure. This is the
# uMFE-style criterion and is more permissive than "RNAfold returns exactly
# the target" -- say which one you used when you report numbers.
HIT_TOLERANCE = 1e-4


def sequence_to_array(seq: str) -> np.ndarray:
    return np.array([NUCLEOTIDES.index(c) for c in seq], dtype=np.int64)


def array_to_sequence(arr) -> str:
    return "".join(NUCLEOTIDES[int(x)] for x in np.asarray(arr))


def arrays_to_sequences(batch: np.ndarray) -> list[str]:
    """Vectorised batch conversion. Worth it: the oracle sees millions."""
    lookup = np.frombuffer(NUCLEOTIDES.encode(), dtype=np.uint8)
    return [row.tobytes().decode() for row in lookup[np.asarray(batch)]]

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

import argparse
from dataclasses import asdict, dataclass, fields


@dataclass
class GFlowNetConfig:
    """One training iteration is: one no-grad rollout, one teacher-forced
    forward/backward, and `batch_size` folding calls.

    Folding dominates and parallelises across cores, so a large batch with few
    iterations beats a small batch with many -- the same number of folds, far
    fewer sequential GPU launches.
    """

    iterations: int = 400
    batch_size: int = 512
    learning_rate: float = 2e-3
    learning_rate_logz: float = 1e-1
    reward_beta: float = 1.0
    epsilon: float = 0.05
    grad_clip: float = 1.0
    log_every: int = 50

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    attention_window: int | None = None   # None = full causal; 1 = Markov

    def replace(self, **changes) -> "GFlowNetConfig":
        return GFlowNetConfig(**{**asdict(self), **changes})


def add_gflownet_args(parser: argparse.ArgumentParser) -> None:
    for field in fields(GFlowNetConfig):
        if field.name == "attention_window":
            parser.add_argument("--attention-window", type=int, default=None,
                                help="1 = Markov ablation; omit for full causal")
            continue
        parser.add_argument(f"--{field.name.replace('_', '-')}",
                            type=type(field.default), default=field.default)


def gflownet_from_args(args) -> GFlowNetConfig:
    values = {f.name: getattr(args, f.name) for f in fields(GFlowNetConfig)
              if hasattr(args, f.name)}
    return GFlowNetConfig(**values)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results",
                        help="directory for result JSON")
    parser.add_argument("--tag", type=str, default="",
                        help="suffix appended to the output filename")
    parser.add_argument("--workers", type=int, default=None,
                        help="folding processes; defaults to $SLURM_CPUS_PER_TASK")
    parser.add_argument("--max-length", type=int, default=90,
                        help="skip targets longer than this")
    parser.add_argument("--no-taneda", action="store_true",
                        help="synthetic ladder only")

# -------------------------------------------------------------------------
# Structures and the generation plan
# -------------------------------------------------------------------------

from dataclasses import dataclass

import numpy as np



def pair_table(dot_bracket: str) -> np.ndarray:
    """Dot-bracket -> array where entry i is i's partner, or -1 if unpaired."""
    table = np.full(len(dot_bracket), -1, dtype=np.int64)
    stack: list[int] = []

    for i, char in enumerate(dot_bracket):
        if char in "([{<":
            stack.append(i)
        elif char in ")]}>":
            if not stack:
                raise ValueError(f"unbalanced structure at position {i}")
            j = stack.pop()
            table[i], table[j] = j, i

    if stack:
        raise ValueError("unbalanced structure: unclosed brackets")
    return table


def helices(table: np.ndarray) -> list[list[tuple[int, int]]]:
    """Maximal stacked runs of pairs, each listed outermost first."""
    opening = [i for i in range(len(table)) if table[i] > i]
    starts = [i for i in opening
              if not (i - 1 >= 0 and table[i - 1] == table[i] + 1)]

    runs = []
    for start in starts:
        run, i, j = [], start, int(table[start])
        while i < j and table[i] == j:
            run.append((i, j))
            i, j = i + 1, j - 1
        runs.append(run)
    return runs


@dataclass
class Variable:
    kind: str                     # "pair" or "base"
    positions: tuple[int, ...]
    parent: int                   # previous variable in the same helix, or -1

    @property
    def n_actions(self) -> int:
        return N_PAIR_TYPES if self.kind == "pair" else N_BASES

    @property
    def anchor(self) -> int:
        return self.positions[0]


class GenerationPlan:
    """Fixed, deterministic assignment order.

    Helix pairs first (outermost to innermost, so stacking partners are
    adjacent in the order), then unpaired positions. Two consequences:
    a non-canonical pair is never representable, so no probability mass is
    wasted outside the compatible set; and every finished sequence has exactly
    one trajectory, so the backward policy is identically 1 and trajectory
    balance loses a term.
    """

    def __init__(self, table: np.ndarray):
        self.table = table
        self.length = len(table)
        self.variables: list[Variable] = []

        for run in helices(table):
            for depth, (i, j) in enumerate(run):
                self.variables.append(
                    Variable("pair", (i, j),
                             len(self.variables) - 1 if depth else -1)
                )
        for i in range(self.length):
            if table[i] == -1:
                self.variables.append(Variable("base", (i,), -1))

        self.anchors = np.array([v.anchor for v in self.variables])
        self.n_actions = np.array([v.n_actions for v in self.variables])
        self.is_pair = np.array([v.kind == "pair" for v in self.variables])
        self.partners = np.array([
            v.positions[1] if v.kind == "pair" else v.anchor
            for v in self.variables
        ])

    def __len__(self) -> int:
        return len(self.variables)

    def __iter__(self):
        return iter(self.variables)


def structure_marks(table: np.ndarray) -> np.ndarray:
    """0 unpaired, 1 opens a pair, 2 closes one."""
    marks = np.zeros(len(table), dtype=np.int64)
    for i in range(len(table)):
        if table[i] > i:
            marks[i] = 1
        elif table[i] >= 0:
            marks[i] = 2
    return marks


def n_pairs(dot_bracket: str) -> int:
    return dot_bracket.count("(")

# -------------------------------------------------------------------------
# Folding oracle -- the hot path of every experiment
# -------------------------------------------------------------------------

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np


_WORKER_TARGET: str | None = None


def default_workers() -> int:
    """Slurm tells us how many cores we actually own; os.cpu_count does not."""
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        if os.environ.get(key):
            return max(1, int(os.environ[key]))
    return max(1, (os.cpu_count() or 2))


def _init_worker(target: str) -> None:
    global _WORKER_TARGET
    _WORKER_TARGET = target


def _fold_chunk(sequences: list[str]) -> list[float]:
    import RNA

    out = []
    for seq in sequences:
        fold = RNA.fold_compound(seq)
        _, mfe = fold.mfe()
        out.append(fold.eval_structure(_WORKER_TARGET) - mfe)
    return out


class FoldingOracle:
    """Scores sequences against one fixed target structure.

    The central quantity is

        gap(s) = E(target | s) - E(MFE | s)  >= 0

    which is zero exactly when the target is a minimum-free-energy structure.
    """

    _pools: dict[str, ProcessPoolExecutor] = {}

    def __init__(self, target: str, workers: int | None = None,
                 cache: bool = True):
        self.target = target
        self.length = len(target)
        self.table = pair_table(target)

        self._cache: dict[str, float] = {} if cache else None
        self.n_folds = 0
        self.n_lookups = 0

        self.workers = max(1, min(workers or default_workers(), 32))
        self._pool = self._get_pool()

    # -- pool management -----------------------------------------------------

    def _get_pool(self):
        if self.workers <= 1:
            return None
        key = f"{self.target}:{self.workers}"
        if key not in FoldingOracle._pools:
            try:
                context = mp.get_context("fork")
                FoldingOracle._pools[key] = ProcessPoolExecutor(
                    max_workers=self.workers, mp_context=context,
                    initializer=_init_worker, initargs=(self.target,),
                )
            except Exception:
                return None
        return FoldingOracle._pools[key]

    @classmethod
    def shutdown_pools(cls) -> None:
        for pool in cls._pools.values():
            pool.shutdown(wait=False, cancel_futures=True)
        cls._pools.clear()

    # -- folding -------------------------------------------------------------

    def _fold_serial(self, sequences: list[str]) -> list[float]:
        import RNA

        out = []
        for seq in sequences:
            fold = RNA.fold_compound(seq)
            _, mfe = fold.mfe()
            out.append(fold.eval_structure(self.target) - mfe)
        return out

    def _fold_many(self, sequences: list[str]) -> list[float]:
        if self._pool is None or len(sequences) < 4 * self.workers:
            return self._fold_serial(sequences)

        size = max(32, (len(sequences) + self.workers - 1) // self.workers)
        chunks = [sequences[i:i + size] for i in range(0, len(sequences), size)]
        try:
            return [gap for chunk in self._pool.map(_fold_chunk, chunks)
                    for gap in chunk]
        except Exception:
            self._pool = None          # fall back permanently rather than die
            return self._fold_serial(sequences)

    # -- public --------------------------------------------------------------

    def gaps(self, seqs) -> np.ndarray:
        strings = arrays_to_sequences(np.atleast_2d(np.asarray(seqs)))
        self.n_lookups += len(strings)

        if self._cache is None:
            self.n_folds += len(strings)
            return np.array(self._fold_many(strings))

        missing = list({s for s in strings if s not in self._cache})
        if missing:
            for seq, gap in zip(missing, self._fold_many(missing)):
                self._cache[seq] = gap
            self.n_folds += len(missing)
        return np.array([self._cache[s] for s in strings])

    def hits(self, seqs) -> np.ndarray:
        return self.gaps(seqs) <= HIT_TOLERANCE

    def log_reward(self, seqs, beta: float = 1.0):
        """R = exp(-gap / (RT*beta)): exactly 1 on a solution, decaying smoothly
        away. A binary reward is zero almost everywhere on hard targets and
        gives the policy nothing to climb."""
        gaps = np.maximum(self.gaps(seqs), 0.0)
        return -gaps / (GAS_CONSTANT_T * beta), gaps

    def reset_counter(self) -> None:
        self.n_folds = self.n_lookups = 0

    def cache_info(self) -> str:
        avoided = self.n_lookups - self.n_folds
        rate = avoided / self.n_lookups if self.n_lookups else 0.0
        return (f"{self.n_lookups:,} lookups, {self.n_folds:,} folds, "
                f"{rate:.0%} avoided")


def soundness_gate(oracle: FoldingOracle, sampler, n_samples: int,
                   rng) -> float:
    """Worst gap over random compatible sequences; must be >= 0.

    The minimum free energy can never exceed the energy of a specific structure
    on the same sequence. A negative gap means the folder missed the target,
    which forces a 0% hit rate for reasons unrelated to the biology. ViennaRNA
    passes trivially; the gate exists because a hand-written folder used
    earlier in this project did not, and the fake zeros it produced were
    indistinguishable from a real result.
    """
    return float(oracle.gaps(sampler(oracle.table, n_samples, rng)).min())

# -------------------------------------------------------------------------
# Targets
# -------------------------------------------------------------------------

import glob
import os
import subprocess
import tarfile

# Datasets are cached next to this file. Override with $RNAGFN_DATA, e.g. to
# put them on scratch rather than in the repo.
DATA_ROOT = os.environ.get(
    "RNAGFN_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)


def hairpin(stem: int, loop: int = 4) -> str:
    return "(" * stem + "." * loop + ")" * stem


def junction(degree: int, stem: int, loop: int = 4, spacer: int = 1) -> str:
    """Multiloop of the given degree. Degree and stem length are the two
    difficulty knobs in the synthetic ladder."""
    inner = ("." * spacer).join(hairpin(stem, loop) for _ in range(degree - 1))
    return "(" * stem + "." * spacer + inner + "." * spacer + ")" * stem


def chain(count: int, stem: int, loop: int = 4, spacer: int = 2) -> str:
    return ("." * spacer).join(hairpin(stem, loop) for _ in range(count))


def difficulty_ladder() -> dict[str, str]:
    targets: dict[str, str] = {}
    for stem in (4, 6, 8, 10):
        targets[f"hairpin_s{stem}"] = hairpin(stem)
    for degree in (2, 3, 4, 5):
        for stem in (3, 5):
            targets[f"junction_d{degree}_s{stem}"] = junction(degree, stem)
    for count in (2, 3, 4):
        for stem in (4, 6):
            targets[f"chain_m{count}_s{stem}"] = chain(count, stem)
    return targets


def clone_once(url: str, name: str) -> str:
    """Clone into DATA_ROOT once. On a cluster login node run
    `python -m rnagfn.targets` first: compute nodes are often offline."""
    os.makedirs(DATA_ROOT, exist_ok=True)
    path = os.path.join(DATA_ROOT, name)
    if not os.path.isdir(path):
        subprocess.run(["git", "clone", "--depth", "1", "-q", url, path],
                       check=True)
    return path


def rfam_taneda(max_length: int | None = 90) -> dict[str, str]:
    root = clone_once("https://github.com/automl/learna", "learna")
    extracted = os.path.join(root, "_taneda")

    if not os.path.isdir(extracted):
        archive = tarfile.open(os.path.join(root, "data/rfam_taneda.tar.gz"))
        try:
            archive.extractall(extracted, filter="data")
        except TypeError:
            archive.extractall(extracted)

    targets: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(extracted, "**", "*.rna"),
                                 recursive=True)):
        found = [t for t in open(path).read().split()
                 if set(t) <= set(".()") and "(" in t]
        if found and (max_length is None or len(found[0]) <= max_length):
            name = "taneda_" + os.path.basename(path).removesuffix(".rna")
            targets[name] = found[0]
    return targets


def eterna100():
    """Eterna100 with per-puzzle solved/unsolved flags from RNA-Undesign.

    `solved_umfe == 0` marks puzzles never solved under the unique-MFE
    criterion; a subset are proven undesignable (Zhou et al.). They are the
    negative control: a method reporting a solution there is either making a
    discovery or has a bug.
    """
    import pandas as pd

    root = clone_once("https://github.com/shanry/RNA-Undesign", "RNA-Undesign")
    frame = pd.read_csv(os.path.join(root, "data/eterna100_design.csv"))
    frame = frame.rename(columns={
        "Puzzle #": "puzzle", "Puzzle Name": "name",
        "Secondary Structure": "structure",
    })[["puzzle", "name", "structure", "solved_mfe", "solved_umfe"]]
    frame["length"] = frame["structure"].str.len()
    return frame


def load_targets(include_taneda: bool = True,
                 max_length: int | None = 90) -> dict[str, str]:
    targets = dict(difficulty_ladder())
    if include_taneda:
        targets.update(rfam_taneda(max_length=max_length))
    return targets


if __name__ == "__main__":
    # Prefetch on a login node, where the network works.
    print(f"data root: {DATA_ROOT}")
    print(f"{len(load_targets())} ladder + Taneda targets")
    print(f"{len(eterna100())} Eterna100 puzzles")
    print("datasets cached")

# -------------------------------------------------------------------------
# Baseline samplers
# -------------------------------------------------------------------------

import numpy as np


_STACK_TABLE: np.ndarray | None = None


def stacking_table() -> np.ndarray:
    """Stacking free energies measured from ViennaRNA itself.

    For each (outer, inner) pair combination, evaluate a minimal two-pair helix
    closing a GAAA tetraloop and subtract the same hairpin without the outer
    pair, dangles off so the difference isolates the stack. Measuring rather
    than hard-coding keeps the baseline consistent with whatever parameters are
    loaded, and covers the GU wobbles for free.
    """
    global _STACK_TABLE
    if _STACK_TABLE is not None:
        return _STACK_TABLE

    import RNA

    details = RNA.md()
    details.dangles = 0
    table = np.zeros((N_PAIR_TYPES, N_PAIR_TYPES))

    for outer, (a, b) in enumerate(CANONICAL_PAIRS):
        for inner, (c, d) in enumerate(CANONICAL_PAIRS):
            two = (NUCLEOTIDES[a] + NUCLEOTIDES[c] + "GAAA"
                   + NUCLEOTIDES[d] + NUCLEOTIDES[b])
            one = NUCLEOTIDES[c] + "GAAA" + NUCLEOTIDES[d]
            table[outer, inner] = (
                RNA.fold_compound(two, details).eval_structure("((....))")
                - RNA.fold_compound(one, details).eval_structure("(....)")
            )

    _STACK_TABLE = table
    return table


def sample_uniform(table: np.ndarray, n_samples: int, rng) -> np.ndarray:
    """Uniform over the compatible set.

    For a single pseudoknot-free target the base-pair dependency graph is a
    perfect matching (treewidth 1), so this is not an approximation of
    RNAblueprint -- it is RNAblueprint's distribution for this case.
    """
    length = len(table)
    seqs = rng.integers(0, N_BASES, size=(n_samples, length))

    opening = np.flatnonzero(table > np.arange(length))
    if len(opening):
        choice = rng.integers(0, N_PAIR_TYPES, size=(n_samples, len(opening)))
        seqs[:, opening] = PAIR_FIRST[choice]
        seqs[:, table[opening]] = PAIR_SECOND[choice]
    return seqs


def _categorical(probabilities: np.ndarray, rng) -> np.ndarray:
    draws = (rng.random(probabilities.shape[0])[:, None]
             > probabilities.cumsum(axis=1)).sum(axis=1)
    return np.clip(draws, 0, probabilities.shape[1] - 1)


def sample_boltzmann(table: np.ndarray, n_samples: int, rng,
                     temperature: float = 1.0) -> np.ndarray:
    """RNARedPrint for a single target: each helix is a chain whose links are
    stacking interactions, so its exact Boltzmann distribution factorises and
    is sampled by a transfer-matrix pass plus a stochastic backtrack."""
    length = len(table)
    seqs = rng.integers(0, N_BASES, size=(n_samples, length))
    weights = np.exp(-stacking_table() / (GAS_CONSTANT_T * temperature))

    for run in helices(table):
        depth = len(run)
        messages = [np.ones(N_PAIR_TYPES) for _ in range(depth)]
        for k in range(depth - 2, -1, -1):
            messages[k] = weights @ messages[k + 1]

        root = messages[0] / messages[0].sum()
        types = _categorical(np.tile(root, (n_samples, 1)), rng)
        i, j = run[0]
        seqs[:, i], seqs[:, j] = PAIR_FIRST[types], PAIR_SECOND[types]

        for k in range(1, depth):
            probabilities = weights[types] * messages[k][None, :]
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            types = _categorical(probabilities, rng)
            i, j = run[k]
            seqs[:, i], seqs[:, j] = PAIR_FIRST[types], PAIR_SECOND[types]

    return seqs


SAMPLERS = {
    "uniform": sample_uniform,
    "boltzmann": sample_boltzmann,
}
SAMPLER_LABELS = {
    "uniform": "uniform (RNAblueprint)",
    "boltzmann": "Boltzmann (RNARedPrint)",
}


# --- generators, for the compute-matched comparison -------------------------
# Each yields batches of candidates forever; the budget harness stops them.

def candidates_uniform(oracle, rng, batch_size: int = 512):
    while True:
        yield sample_uniform(oracle.table, batch_size, rng)


def candidates_boltzmann(oracle, rng, batch_size: int = 512):
    while True:
        yield sample_boltzmann(oracle.table, batch_size, rng)


def candidates_adaptive_walk(oracle, rng, batch_size: int = 256,
                             pair_rate: float = 0.7):
    """Reward-guided local search, vectorised across walkers. Solved walkers
    restart, so one run harvests many distinct solutions.

    `batch_size` is the walker count: one yield per step, so it is also how
    often the budget harness gets to look at the clock."""
    n_walkers = batch_size
    table = oracle.table
    paired = np.flatnonzero(table > np.arange(oracle.length))
    unpaired = np.flatnonzero(table == -1)
    rows = np.arange(n_walkers)

    seqs = sample_uniform(table, n_walkers, rng)
    gaps = oracle.gaps(seqs)

    while True:
        proposal = seqs.copy()
        use_pair = (rng.random(n_walkers) < pair_rate) if len(paired) \
            else np.zeros(n_walkers, dtype=bool)
        if not len(unpaired):
            use_pair[:] = True

        if use_pair.any():
            sites = paired[rng.integers(0, len(paired), int(use_pair.sum()))]
            kinds = rng.integers(0, N_PAIR_TYPES, int(use_pair.sum()))
            proposal[rows[use_pair], sites] = PAIR_FIRST[kinds]
            proposal[rows[use_pair], table[sites]] = PAIR_SECOND[kinds]
        if (~use_pair).any():
            sites = unpaired[rng.integers(0, len(unpaired),
                                          int((~use_pair).sum()))]
            proposal[rows[~use_pair], sites] = rng.integers(
                0, N_BASES, int((~use_pair).sum()))

        proposal_gaps = oracle.gaps(proposal)
        improved = proposal_gaps <= gaps
        seqs[improved] = proposal[improved]
        gaps[improved] = proposal_gaps[improved]

        yield seqs.copy()

        solved = gaps <= HIT_TOLERANCE
        if solved.any():
            restart = sample_uniform(table, int(solved.sum()), rng)
            seqs[solved] = restart
            gaps[solved] = oracle.gaps(restart)


def candidates_rnainverse(oracle, rng, batch_size: int = 8):
    """RNAinverse (Hofacker et al. 1994).

    A small batch on purpose: each call is a full restart and the harness can
    only check the clock between yields, so a large batch overshoots.
    """
    import RNA

    while True:
        starts = sample_uniform(oracle.table, batch_size, rng)
        designed = [RNA.inverse_fold(array_to_sequence(s), oracle.target)[0]
                    for s in starts]
        yield np.array([sequence_to_array(d) for d in designed])


def adaptive_walk(oracle, rng, n_walkers: int = 256, n_steps: int = 300,
                  harvest: bool = True):
    """Bounded-step version used by the designability control."""
    generator = candidates_adaptive_walk(oracle, rng, batch_size=n_walkers)
    solutions: list[np.ndarray] = []
    best = float("inf")

    for _ in range(n_steps):
        seqs = next(generator)
        gaps = oracle.gaps(seqs)
        best = min(best, float(gaps.min()))
        solved = gaps <= HIT_TOLERANCE
        if solved.any():
            solutions.extend(row.copy() for row in seqs[solved])
            if not harvest:
                break

    found = (np.unique(np.array(solutions), axis=0) if solutions
             else np.zeros((0, oracle.length), dtype=np.int64))
    return found, best


CANDIDATE_GENERATORS = {
    "uniform": candidates_uniform,
    "boltzmann": candidates_boltzmann,
    "rnainverse": candidates_rnainverse,
    "adaptive_walk": candidates_adaptive_walk,
}
GENERATOR_LABELS = {
    "uniform": "uniform (RNAblueprint)",
    "boltzmann": "Boltzmann (RNARedPrint)",
    "rnainverse": "RNAinverse (Hofacker 1994)",
    "adaptive_walk": "adaptive walk (local search)",
}

# -------------------------------------------------------------------------
# Compute-matched budget harness
# -------------------------------------------------------------------------

import time

import numpy as np


def calibrate_batch(oracle, rng, seconds: float, probe: int = 32,
                    low: int = 32, high: int = 2048) -> int:
    """Pick a batch size so one batch costs about 2% of the budget.

    The harness can only check the clock between yields, so a batch that takes
    longer than the budget blows straight through it -- which is easy to hit on
    a long target or a slow node. Timing a small probe makes the harness
    budget-respecting without per-target tuning.
    """
    started = time.time()
    oracle.gaps(sample_uniform(oracle.table, probe, rng))
    per_sequence = (time.time() - started) / probe

    if per_sequence <= 0:
        return high
    return int(np.clip(0.02 * seconds / per_sequence, low, high))


def run_under_budget(label: str, factory, oracle, seconds: float, rng,
                     batch_size: int | None = None,
                     max_timeline: int = 20_000) -> dict:
    oracle.reset_counter()

    if batch_size is None:
        batch_size = calibrate_batch(oracle, rng, seconds)

    started = time.time()
    seen: set[bytes] = set()
    timeline: list[tuple[float, int]] = [(0.0, 0)]
    n_candidates = 0

    try:
        generator = factory(oracle, rng, batch_size=batch_size)
    except TypeError:
        generator = factory(oracle, rng)        # generator fixes its own size

    for batch in generator:
        batch = np.atleast_2d(batch)
        n_candidates += len(batch)

        for row in batch[oracle.hits(batch)]:
            key = row.astype(np.int8).tobytes()
            if key not in seen:
                seen.add(key)
                if len(timeline) < max_timeline:
                    timeline.append((time.time() - started, len(seen)))

        elapsed = time.time() - started
        if timeline[-1][0] < elapsed:
            timeline.append((elapsed, len(seen)))
        if elapsed >= seconds:
            break

    elapsed = time.time() - started
    return {
        "method": label,
        "batch_size": batch_size,
        "overshoot": max(0.0, elapsed - seconds),
        "distinct_solutions": len(seen),
        "candidates_evaluated": n_candidates,
        "folds": oracle.n_folds,
        "elapsed": elapsed,
        "per_second": len(seen) / max(elapsed, 1e-9),
        "timeline": timeline,
    }


def break_even(train_seconds: float, trained_rate: float,
               baseline_rate: float) -> dict:
    """How many solutions you must want before paying for training is rational.

    Returns `reachable=False` when the trained policy is slower per second than
    the best baseline -- in that case training can never be amortised away on a
    single target, however long you run.
    """
    if trained_rate <= baseline_rate:
        return {"reachable": False, "solutions": None,
                "trained_rate": trained_rate, "baseline_rate": baseline_rate,
                "train_seconds": train_seconds}

    solutions = train_seconds / (1.0 / baseline_rate - 1.0 / trained_rate)
    return {"reachable": True, "solutions": solutions,
            "trained_rate": trained_rate, "baseline_rate": baseline_rate,
            "train_seconds": train_seconds}
