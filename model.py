"""Policy and trainer -- the only file that imports torch.

An earlier version re-encoded the whole partial sequence at every one of the K
decisions, with gradients, building a K-deep autograd graph per rollout. That
was the GPU bottleneck. This is a causal transformer over the *decision
sequence* instead:

    sampling   K incremental forwards under no_grad, no graph at all
    loss       ONE teacher-forced forward and ONE backward over all K

Attention over previous decisions carries the same information as re-reading
the partial sequence, so nothing is given up -- and the Markov ablation (the
Boltzmann baseline's model family, with learned rather than hand-set weights)
becomes an attention window of 1 rather than a second model class.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rnagfn import (CANONICAL_PAIRS, MAX_ACTIONS, FoldingOracle,
                    GenerationPlan, GFlowNetConfig, HIT_TOLERANCE, pair_table,
                    structure_marks)

import numpy as np
import torch
import torch.nn as nn


START_ACTION = MAX_ACTIONS          # "no previous decision yet"


def causal_mask(size: int, window: int | None, device) -> torch.Tensor:
    """True where attention is NOT allowed."""
    index = torch.arange(size, device=device)
    mask = index[None, :] > index[:, None]
    if window is not None:
        mask = mask | ((index[:, None] - index[None, :]) >= window)
    return mask


class PlanContext:
    """Everything about one target that can be precomputed once."""

    def __init__(self, structure: str, attention_window: int | None, device):
        self.structure = structure
        self.table = pair_table(structure)
        self.plan = GenerationPlan(self.table)
        self.length = self.plan.length
        self.n_decisions = len(self.plan)

        self.marks = torch.tensor(structure_marks(self.table), device=device)
        self.anchors = torch.tensor(self.plan.anchors, device=device)
        self.partners = torch.tensor(self.plan.partners, device=device)
        self.is_pair = torch.tensor(self.plan.is_pair.astype(np.int64),
                                    device=device)
        self.n_actions = torch.tensor(self.plan.n_actions, device=device)

        self.action_mask = (
            torch.arange(MAX_ACTIONS, device=device)[None, :]
            >= self.n_actions[:, None]
        )
        self.attention_mask = causal_mask(self.n_decisions, attention_window,
                                          device)
        self.pair_first = torch.tensor([p[0] for p in CANONICAL_PAIRS],
                                       device=device)
        self.pair_second = torch.tensor([p[1] for p in CANONICAL_PAIRS],
                                        device=device)


class DesignPolicy(nn.Module):
    """Structure encoder plus causal decision decoder.

    The same class serves per-target and amortised training: the target enters
    only through the encoded marks, so nothing is tied to one structure.
    """

    def __init__(self, max_length: int, max_decisions: int,
                 cfg: GFlowNetConfig):
        super().__init__()
        self.attention_window = cfg.attention_window
        self.d_model = cfg.d_model

        self.mark_embedding = nn.Embedding(3, cfg.d_model)
        self.structure_position = nn.Parameter(
            torch.randn(1, max_length, cfg.d_model) * 0.02
        )
        self.structure_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.n_heads,
                dim_feedforward=4 * cfg.d_model, dropout=0.0,
                batch_first=True, norm_first=True,
            ), 2, enable_nested_tensor=False,
        )

        self.action_embedding = nn.Embedding(MAX_ACTIONS + 1, cfg.d_model)
        self.kind_embedding = nn.Embedding(2, cfg.d_model)
        self.decision_position = nn.Parameter(
            torch.randn(1, max_decisions, cfg.d_model) * 0.02
        )
        self.decision_decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.n_heads,
                dim_feedforward=4 * cfg.d_model, dropout=0.0,
                batch_first=True, norm_first=True,
            ), cfg.n_layers, enable_nested_tensor=False,
        )

        self.head = nn.Linear(cfg.d_model, MAX_ACTIONS)
        self.log_z_head = nn.Linear(cfg.d_model, 1)

    def encode_structure(self, context: PlanContext) -> torch.Tensor:
        hidden = (self.mark_embedding(context.marks).unsqueeze(0)
                  + self.structure_position[:, :context.length, :])
        return self.structure_encoder(hidden)              # (1, L, d)

    def log_z(self, structure: torch.Tensor) -> torch.Tensor:
        return self.log_z_head(structure.mean(dim=1)).squeeze(-1).squeeze(0)

    def decision_features(self, context: PlanContext,
                          structure: torch.Tensor) -> torch.Tensor:
        """Per-decision features, independent of the actions taken. (K, d)"""
        flat = structure.squeeze(0)
        return (flat[context.anchors]
                + flat[context.partners] * context.is_pair.unsqueeze(1)
                + self.kind_embedding(context.is_pair)
                + self.decision_position[0, :context.n_decisions, :])

    def logits(self, tokens: torch.Tensor, context: PlanContext,
               upto: int | None = None) -> torch.Tensor:
        steps = tokens.shape[1]
        hidden = self.decision_decoder(
            tokens, mask=context.attention_mask[:steps, :steps]
        )
        limit = steps if upto is None else upto
        return self.head(hidden).masked_fill(
            context.action_mask[:limit].unsqueeze(0), float("-inf")
        )


def build_policy(structures, cfg: GFlowNetConfig) -> DesignPolicy:
    """Size the embeddings to the largest structure and plan ever seen."""
    if isinstance(structures, str):
        structures = [structures]
    max_length = max(len(s) for s in structures)
    max_decisions = max(len(GenerationPlan(pair_table(s))) for s in structures)
    return DesignPolicy(max_length, max_decisions, cfg)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return device


class TrajectoryBalanceTrainer:

    def __init__(self, policy: DesignPolicy, cfg: GFlowNetConfig,
                 device=None, workers: int | None = None,
                 use_amp: bool | None = None, progress: bool = True):
        self.device = device or get_device()
        self.policy = policy.to(self.device)
        self.cfg = cfg
        self.workers = workers
        self.progress = progress

        self.contexts: dict[str, PlanContext] = {}
        self.oracles: dict[str, FoldingOracle] = {}

        body = [p for name, p in policy.named_parameters()
                if "log_z_head" not in name]
        self.optimizer = torch.optim.Adam([
            {"params": body, "lr": cfg.learning_rate},
            {"params": policy.log_z_head.parameters(),
             "lr": cfg.learning_rate_logz},
        ])
        self.use_amp = (self.device.type == "cuda") if use_amp is None else use_amp

    def context(self, structure: str) -> PlanContext:
        if structure not in self.contexts:
            self.contexts[structure] = PlanContext(
                structure, self.cfg.attention_window, self.device
            )
        return self.contexts[structure]

    def oracle(self, structure: str) -> FoldingOracle:
        if structure not in self.oracles:
            self.oracles[structure] = FoldingOracle(structure,
                                                    workers=self.workers)
        return self.oracles[structure]

    # -- rollout, no gradients ----------------------------------------------

    @torch.no_grad()
    def rollout(self, context: PlanContext, batch_size: int, epsilon: float):
        policy = self.policy
        steps = context.n_decisions
        amp = self.use_amp and self.device.type == "cuda"

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            structure = policy.encode_structure(context)
            features = policy.decision_features(context, structure).float()

        tokens = torch.zeros(batch_size, steps, policy.d_model,
                             device=self.device)
        actions = torch.zeros(batch_size, steps, dtype=torch.long,
                              device=self.device)
        previous = torch.full((batch_size,), START_ACTION, dtype=torch.long,
                              device=self.device)

        valid = (~context.action_mask).float()
        uniform = valid / context.n_actions.unsqueeze(1).float()

        for step in range(steps):
            tokens[:, step] = features[step] + policy.action_embedding(previous)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                scores = policy.logits(tokens[:, :step + 1], context,
                                       upto=step + 1)[:, step]
            probs = F.softmax(scores.float(), dim=-1)
            behaviour = (1 - epsilon) * probs + epsilon * uniform[step]
            previous = torch.multinomial(behaviour, 1).squeeze(1)
            actions[:, step] = previous

        return self.assemble(context, actions), actions

    @staticmethod
    def assemble(context: PlanContext, actions: torch.Tensor) -> np.ndarray:
        batch = actions.shape[0]
        seqs = torch.zeros(batch, context.length, dtype=torch.long,
                           device=actions.device)
        pair_steps = context.is_pair.bool()
        base_steps = ~pair_steps

        if pair_steps.any():
            chosen = actions[:, pair_steps]
            seqs[:, context.anchors[pair_steps]] = context.pair_first[chosen]
            seqs[:, context.partners[pair_steps]] = context.pair_second[chosen]
        if base_steps.any():
            seqs[:, context.anchors[base_steps]] = actions[:, base_steps]
        return seqs.cpu().numpy()

    # -- loss: one forward, one backward -------------------------------------

    def log_prob(self, context: PlanContext, actions: torch.Tensor):
        policy = self.policy
        batch = actions.shape[0]

        structure = policy.encode_structure(context)
        features = policy.decision_features(context, structure)

        start = torch.full((batch, 1), START_ACTION, dtype=torch.long,
                           device=self.device)
        previous = torch.cat([start, actions[:, :-1]], dim=1)

        tokens = features.unsqueeze(0) + policy.action_embedding(previous)
        scores = policy.logits(tokens, context)

        log_probs = F.log_softmax(scores.float(), dim=-1)
        chosen = log_probs.gather(2, actions.unsqueeze(2)).squeeze(2)
        return chosen.sum(dim=1), policy.log_z(structure)

    # -- training ------------------------------------------------------------

    def fit(self, structures, iterations: int | None = None):
        records = [record for record, _ in self.fit_iter(structures, iterations)]
        return records

    def fit_iter(self, structures, iterations: int | None = None):
        """Yields (record, sequences) per iteration, so a caller can watch the
        batches go by -- that is how the budget experiment credits the method
        with solutions found *during* training."""
        if isinstance(structures, str):
            structures = [structures]
        cfg = self.cfg
        total = cfg.iterations if iterations is None else iterations

        steps = range(total)
        if self.progress:
            try:
                from tqdm.auto import trange

                steps = trange(total, desc="trajectory balance")
            except ImportError:
                pass

        for iteration in steps:
            structure = structures[iteration % len(structures)]
            context = self.context(structure)
            oracle = self.oracle(structure)

            seqs, actions = self.rollout(context, cfg.batch_size, cfg.epsilon)
            log_reward, gaps = oracle.log_reward(seqs, beta=cfg.reward_beta)
            log_reward = torch.as_tensor(log_reward, dtype=torch.float32,
                                         device=self.device)

            log_pf, log_z = self.log_prob(context, actions)
            loss = ((log_z + log_pf - log_reward) ** 2).mean()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip)
            self.optimizer.step()

            record = {
                "iteration": iteration,
                "structure": structure,
                "loss": float(loss.item()),
                "batch_hit_rate": float((gaps <= HIT_TOLERANCE).mean()),
                "mean_gap": float(gaps.mean()),
                "log_z": float(log_z.item()),
            }
            if cfg.log_every and hasattr(steps, "set_postfix") \
                    and iteration % cfg.log_every == 0:
                steps.set_postfix(loss=f"{loss.item():.1f}",
                                  hit=f"{record['batch_hit_rate']:.1%}",
                                  gap=f"{gaps.mean():.2f}")
            yield record, seqs

    @torch.no_grad()
    def sample(self, structure: str, n_samples: int,
               chunk_size: int = 1024) -> np.ndarray:
        context = self.context(structure)
        drawn, out = 0, []
        while drawn < n_samples:
            size = min(chunk_size, n_samples - drawn)
            seqs, _ = self.rollout(context, size, epsilon=0.0)
            out.append(seqs)
            drawn += size
        return np.concatenate(out)

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.policy.state_dict(),
                    "config": self.cfg.__dict__}, path)


def gflownet_generator(oracle, rng, cfg: GFlowNetConfig, seed: int = 0,
                      workers: int | None = None, batch_size: int = 512):
    """Generator for the budget harness: trains inside the budget, yielding its
    training batches, then samples. Solutions found during training count --
    not crediting them would understate the method."""
    torch.manual_seed(seed)
    trainer = TrajectoryBalanceTrainer(build_policy(oracle.target, cfg), cfg,
                                       workers=workers, progress=False)
    trainer.oracles[oracle.target] = oracle

    for _, seqs in trainer.fit_iter(oracle.target):
        yield seqs
    while True:
        yield trainer.sample(oracle.target, batch_size, chunk_size=batch_size)
