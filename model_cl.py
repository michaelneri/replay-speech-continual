"""
model_cl.py — Three continual-learning Lightning modules for ReMASC anti-spoofing.

All three have the same constructor signature as AudioDeepFakeDetectionModelModule
and can be plugged into train_cl.py with minimal additions.

Modules
───────
EWCModelModule   Regularization-based  — Elastic Weight Consolidation
GPMModelModule   Optimization-based    — Gradient Projection Memory
TSBModelModule   Architecture-based    — Task-Specific Beamformer + shared backbone

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Integration notes (minimal additions to train_cl.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1 — add an import at the top of train_cl.py:
    from model_cl import (
        EWCModelModule, GPMModelModule, TSBModelModule,
        consolidate_ewc, consolidate_gpm,
    )

Step 2 — choose ONE variant.  Replace _build_model() in train_cl.py with:

    # ── EWC ──
    def _build_model(label_0, label_1, tag):
        return EWCModelModule(
            input_channels=INPUT_CH, lr=CFG["lr"],
            train_label_0=label_0, train_label_1=label_1,
            n_fft=CFG["n_fft"], hop_length=CFG["hop_length"],
            fs=CFG["target_fs"], info=tag,
            ewc_lambda=5_000.0,
        )

    # ── GPM ──
    def _build_model(label_0, label_1, tag):
        return GPMModelModule(
            input_channels=INPUT_CH, lr=CFG["lr"],
            train_label_0=label_0, train_label_1=label_1,
            n_fft=CFG["n_fft"], hop_length=CFG["hop_length"],
            fs=CFG["target_fs"], info=tag,
            gpm_threshold=0.97,
        )

    # ── TSB ──  (also needs task_idx, see step 3b)
    def _build_model(label_0, label_1, tag, task_idx=0):
        return TSBModelModule(
            input_channels=INPUT_CH, lr=CFG["lr"],
            train_label_0=label_0, train_label_1=label_1,
            n_fft=CFG["n_fft"], hop_length=CFG["hop_length"],
            fs=CFG["target_fs"], info=tag,
            task_idx=task_idx, num_tasks=len(ENVS),
        )

Step 3a — EWC / GPM: after each call to train_one() in _get_or_train_prefix,
          save consolidation state and load it into the next task's model:

    # inside _get_or_train_prefix, after  ckpt = train_one(...)
    if len(prefix) == 1:
        consolidate_ewc(ckpt, train_dm.train_dataloader(), device="cuda")
        # (or consolidate_gpm for GPM)

    # and when creating the model for the NEXT step (len(prefix) > 1):
    model = _build_model(int(label_0), int(label_1), tag)
    _load_weights_into(model, prev_ckpt)     # existing line
    model.load_consolidation(prev_ckpt)      # NEW: loads Fisher / gradient bases

Step 3b — TSB: pass task_idx to _build_model.  In _get_or_train_prefix
          the task index equals len(prefix) - 1:

    task_idx = len(prefix) - 1
    model    = _build_model(int(label_0), int(label_1), tag, task_idx=task_idx)
    _load_weights_into(model, prev_ckpt)     # beamformer bank fully restored
    # activate_task() is already called inside __init__ via task_idx
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning import LightningModule

from model import (
    AudioDeepFakeDetectionModel,
    AdaptiveComplexBeamformer,
    orthogonality_regularization,
    l1_regularization,
)
from utils import EER


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_criterion(label_1: int, label_0: int) -> nn.CrossEntropyLoss:
    """Weighted CE with minimum weight = 1 to prevent 0 × −∞ = NaN."""
    return nn.CrossEntropyLoss(weight=torch.tensor([
        max(float(label_1), 1.0),
        max(float(label_0), 1.0),
    ]))


def _make_optimizer(params, lr: float):
    opt = optim.Adam(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=100, eta_min=0.1 * float(lr)
    )
    return {"optimizer": opt, "lr_scheduler": {"scheduler": sched}}


def _base_step(module, batch, eer_metric):
    """Shared forward + loss computation for val/test steps."""
    inputs, labels = batch['waveform'], batch['label']
    outputs, cw, _ = module(inputs)
    loss = (module.criterion(outputs, labels)
            + orthogonality_regularization(cw)
            + l1_regularization(cw))
    if torch.isnan(loss):
        return None, None, None
    loss    = torch.clamp(loss, max=1e6)
    outputs = torch.nan_to_num(outputs, nan=0.0)
    eer_metric.update(outputs.detach(), labels)
    return loss, outputs, labels


# ─────────────────────────────────────────────────────────────────────────────
# 1. EWC — Elastic Weight Consolidation  (Kirkpatrick et al., 2017)
# ─────────────────────────────────────────────────────────────────────────────

class EWCModelModule(LightningModule):
    """
    Adds a quadratic penalty  (λ/2) Σ_i  F_i (θ_i − θ*_i)²  to the loss.

    F_i  = diagonal Fisher Information (expected squared gradient),
           computed on the previous task's training data after training.
    θ*_i = model parameters after training on the previous task.
    """

    def __init__(
        self,
        input_channels: int,
        lr: float,
        train_label_1: int,
        train_label_0: int,
        n_fft: int,
        hop_length: int,
        fs: int = 44_100,
        info: Optional[str] = None,
        ewc_lambda: float = 5_000.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.ewc_lambda = ewc_lambda
        self.model      = AudioDeepFakeDetectionModel(input_channels, n_fft, hop_length, fs)
        self.lr         = lr
        self.criterion  = _make_criterion(train_label_1, train_label_0)
        self.eer_computation_train = EER()
        self.eer_computation_val   = EER()
        self.eer_computation_test  = EER()
        # Populated by consolidate() / load_consolidation()
        self._ewc_means:  dict[str, torch.Tensor] = {}
        self._ewc_fisher: dict[str, torch.Tensor] = {}

    def forward(self, x):
        return self.model(x)

    # ── EWC penalty ──────────────────────────────────────────────────────────
    def _ewc_penalty(self) -> torch.Tensor:
        if not self._ewc_means:
            return torch.zeros(1, device=self.device).squeeze()
        penalty = torch.zeros(1, device=self.device).squeeze()
        for name, param in self.model.named_parameters():
            if name in self._ewc_fisher:
                f = self._ewc_fisher[name].to(param.device)
                m = self._ewc_means[name].to(param.device)
                penalty = penalty + (f * (param - m).pow(2)).sum()
        return (self.ewc_lambda / 2.0) * penalty

    # ── steps ────────────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        inputs, labels = batch['waveform'], batch['label']
        outputs, cw, _ = self(inputs)
        loss = (self.criterion(outputs, labels)
                + orthogonality_regularization(cw)
                + l1_regularization(cw)
                + self._ewc_penalty())
        if torch.isnan(loss):
            return None
        loss    = torch.clamp(loss, max=1e6)
        outputs = torch.nan_to_num(outputs, nan=0.0)
        self.eer_computation_train.update(outputs.detach(), labels)
        self.log('train_loss',    loss,                       prog_bar=True,  on_epoch=True, on_step=True)
        self.log('train_eer',     self.eer_computation_train, prog_bar=True,  on_epoch=True, on_step=False)
        self.log('ewc_penalty',   self._ewc_penalty().detach(), prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = _base_step(self, batch, self.eer_computation_val)
        if loss is None:
            return None
        self.log('val_loss', loss,                     prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_eer',  self.eer_computation_val, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _, _ = _base_step(self, batch, self.eer_computation_test)
        if loss is None:
            return None
        self.log('test_loss', loss,                      prog_bar=True, on_epoch=True, on_step=False)
        self.log('test_err',  self.eer_computation_test, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        return _make_optimizer(self.parameters(), self.lr)

    # ── CL state management ──────────────────────────────────────────────────
    def consolidate(self, dataloader, device: str = "cuda") -> None:
        """
        Compute and store diagonal Fisher + optimal params after training.
        Call ONCE after trainer.fit() on the current task.
        """
        self.model.train()
        self.model.to(device)
        fisher   = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        n_samples = 0

        for batch in dataloader:
            inp = batch['waveform'].to(device)
            lbl = batch['label'].to(device)
            self.model.zero_grad()
            out, _, _ = self.model(inp)
            F.cross_entropy(out, lbl).backward()
            bs = inp.size(0)
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n].add_(p.grad.data.pow(2).mul_(bs))
            n_samples += bs

        for n in fisher:
            fisher[n].div_(max(n_samples, 1))

        self._ewc_means  = {n: p.detach().clone().cpu()
                            for n, p in self.model.named_parameters()}
        self._ewc_fisher = {n: f.cpu() for n, f in fisher.items()}

    def save_consolidation(self, ckpt_path: str) -> None:
        torch.save({'means': self._ewc_means, 'fisher': self._ewc_fisher},
                   ckpt_path + ".ewc.pt")

    def load_consolidation(self, ckpt_path: str) -> None:
        path = ckpt_path + ".ewc.pt"
        if os.path.exists(path):
            data = torch.load(path, map_location='cpu', weights_only=False)
            self._ewc_means  = data['means']
            self._ewc_fisher = data['fisher']


def consolidate_ewc(ckpt_path: str, dataloader, device: str = "cuda") -> None:
    """
    Top-level helper: load checkpoint, compute Fisher, save .ewc.pt file.
    Call from train_cl.py right after train_one() for steps k ≥ 0.
    """
    saved = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hparams = saved['hyper_parameters']
    model = EWCModelModule(
        input_channels = hparams['input_channels'],
        lr             = hparams['lr'],
        train_label_1  = hparams['train_label_1'],
        train_label_0  = hparams['train_label_0'],
        n_fft          = hparams['n_fft'],
        hop_length     = hparams['hop_length'],
        fs             = hparams.get('fs', 16_000),
        ewc_lambda     = hparams.get('ewc_lambda', 5_000.0),
    )
    model.load_state_dict(saved['state_dict'])
    model.consolidate(dataloader, device=device)
    model.save_consolidation(ckpt_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. GPM — Gradient Projection Memory  (Saha et al., ICLR 2021)
# ─────────────────────────────────────────────────────────────────────────────

class GPMModelModule(LightningModule):
    """
    After each task, computes the principal gradient directions (via SVD)
    for every layer and stores them as a memory basis.  On subsequent tasks,
    gradients are projected onto the null space of the accumulated memory so
    that parameters important to previous tasks are not disrupted.

    No replay buffer — no privacy concerns.
    """

    def __init__(
        self,
        input_channels: int,
        lr: float,
        train_label_1: int,
        train_label_0: int,
        n_fft: int,
        hop_length: int,
        fs: int = 44_100,
        info: Optional[str] = None,
        gpm_threshold: float = 0.97,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.gpm_threshold = gpm_threshold
        self.model         = AudioDeepFakeDetectionModel(input_channels, n_fft, hop_length, fs)
        self.lr            = lr
        self.criterion     = _make_criterion(train_label_1, train_label_0)
        self.eer_computation_train = EER()
        self.eer_computation_val   = EER()
        self.eer_computation_test  = EER()
        # {layer_name: basis [d, k]} — populated by consolidate()
        self._gradient_bases: dict[str, torch.Tensor] = {}

    def forward(self, x):
        return self.model(x)

    # ── gradient projection ──────────────────────────────────────────────────
    def on_before_optimizer_step(self, optimizer) -> None:
        """Project each layer's gradient onto the null space of the memory."""
        if not self._gradient_bases:
            return
        for name, param in self.model.named_parameters():
            if param.grad is None or name not in self._gradient_bases:
                continue
            basis   = self._gradient_bases[name].to(param.device)  # [d, k]
            g_flat  = param.grad.data.view(-1)                      # [d]
            # g_proj = g − basis (basisᵀ g)
            coeffs  = basis.T @ g_flat                              # [k]
            g_flat  = g_flat - basis @ coeffs
            param.grad.data = g_flat.view(param.grad.data.shape)

    # ── steps ────────────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        inputs, labels = batch['waveform'], batch['label']
        outputs, cw, _ = self(inputs)
        loss = (self.criterion(outputs, labels)
                + orthogonality_regularization(cw)
                + l1_regularization(cw))
        if torch.isnan(loss):
            return None
        loss    = torch.clamp(loss, max=1e6)
        outputs = torch.nan_to_num(outputs, nan=0.0)
        self.eer_computation_train.update(outputs.detach(), labels)
        self.log('train_loss', loss,                       prog_bar=True, on_epoch=True, on_step=True)
        self.log('train_eer',  self.eer_computation_train, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = _base_step(self, batch, self.eer_computation_val)
        if loss is None:
            return None
        self.log('val_loss', loss,                     prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_eer',  self.eer_computation_val, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _, _ = _base_step(self, batch, self.eer_computation_test)
        if loss is None:
            return None
        self.log('test_loss', loss,                      prog_bar=True, on_epoch=True, on_step=False)
        self.log('test_err',  self.eer_computation_test, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        return _make_optimizer(self.parameters(), self.lr)

    # ── CL state management ──────────────────────────────────────────────────
    def consolidate(self, dataloader, device: str = "cuda") -> None:
        """
        Compute per-layer gradient bases via SVD after training on a task.
        Existing bases are extended (online GPM): new directions orthogonal
        to already-stored ones are appended.
        Call ONCE after trainer.fit() on the current task.
        """
        # train() (not eval()) — cuDNN's RNN backward kernel requires it,
        # and the model has no Dropout so BatchNorm batch-stats are harmless here.
        self.model.train()
        self.model.to(device)

        # Accumulate one gradient vector per batch for each parameter
        grad_lists: dict[str, list[torch.Tensor]] = {
            n: [] for n, p in self.model.named_parameters() if p.requires_grad
        }

        for batch in dataloader:
            inp = batch['waveform'].to(device)
            lbl = batch['label'].to(device)
            self.model.zero_grad()
            out, cw, _ = self.model(inp)
            loss = (F.cross_entropy(out, lbl)
                    + orthogonality_regularization(cw)
                    + l1_regularization(cw))
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    grad_lists[n].append(p.grad.data.view(-1).clone().cpu())

        new_bases: dict[str, torch.Tensor] = {}

        for name, grads in grad_lists.items():
            if len(grads) < 2:
                continue

            G = torch.stack(grads, dim=1)   # [d, N]

            try:
                U, S, _ = torch.linalg.svd(G, full_matrices=False)   # U: [d, r]
            except RuntimeError:
                continue

            # Select k = fewest vectors that explain >= threshold variance
            cumvar = S.cumsum(0) / (S.sum() + 1e-12)
            k = int((cumvar < self.gpm_threshold).sum()) + 1
            k = min(k, U.shape[1])
            U_new = U[:, :k]    # [d, k_new]  (orthonormal columns)

            # Online merge: project new directions onto null space of old basis
            if name in self._gradient_bases:
                B_old = self._gradient_bases[name].cpu()   # [d, k_old]
                # Remove component already covered by B_old
                U_new_orth = U_new - B_old @ (B_old.T @ U_new)
                norms = U_new_orth.norm(dim=0, keepdim=True).clamp(min=1e-8)
                U_new_orth = U_new_orth / norms
                # Keep only directions with sufficient residual magnitude
                valid = U_new_orth.norm(dim=0) > 0.1
                if valid.any():
                    new_bases[name] = torch.cat([B_old, U_new_orth[:, valid]], dim=1)
                else:
                    new_bases[name] = B_old
            else:
                new_bases[name] = U_new

        self._gradient_bases.update(new_bases)

    def save_consolidation(self, ckpt_path: str) -> None:
        torch.save(self._gradient_bases, ckpt_path + ".gpm.pt")

    def load_consolidation(self, ckpt_path: str) -> None:
        path = ckpt_path + ".gpm.pt"
        if os.path.exists(path):
            self._gradient_bases = torch.load(path, map_location='cpu',
                                              weights_only=False)


def consolidate_gpm(ckpt_path: str, dataloader, device: str = "cuda") -> None:
    """
    Top-level helper: load checkpoint, compute gradient bases, save .gpm.pt.
    Call from train_cl.py right after train_one() for steps k ≥ 0.
    """
    saved   = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hparams = saved['hyper_parameters']
    model   = GPMModelModule(
        input_channels = hparams['input_channels'],
        lr             = hparams['lr'],
        train_label_1  = hparams['train_label_1'],
        train_label_0  = hparams['train_label_0'],
        n_fft          = hparams['n_fft'],
        hop_length     = hparams['hop_length'],
        fs             = hparams.get('fs', 44_100),
        gpm_threshold  = hparams.get('gpm_threshold', 0.97),
    )
    model.load_state_dict(saved['state_dict'])
    model.consolidate(dataloader, device=device)
    model.save_consolidation(ckpt_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. TSB — Task-Specific Beamformer  (architecture-based, no task ID at test)
# ─────────────────────────────────────────────────────────────────────────────

class TSBModelModule(LightningModule):
    """
    Task-Specific Beamformer with shared classification backbone.

    Physical motivation
    ───────────────────
    Each recording environment has distinct acoustic conditions (room geometry,
    background noise, reverberation).  The spatial filtering stage
    (AdaptiveComplexBeamformer) should specialise per environment, while the
    classification backbone (what distinguishes replay from genuine acoustically)
    should generalise.

    Architecture
    ────────────
    • num_tasks independent AdaptiveComplexBeamformer heads (one per env)
    • Shared backbone: feature extraction → attention heatmap → CNN → GRU → linear
    • During training on task k:   only beamformers[k] is updated
    • During inference (no task ID): ensemble logits from all trained heads

    The shared backbone receives gradient updates at every task step, allowing
    continual adaptation of the classification decision while spatial forgetting
    is prevented by freezing old beamformers.
    """

    def __init__(
        self,
        input_channels: int,
        lr: float,
        train_label_1: int,
        train_label_0: int,
        n_fft: int,
        hop_length: int,
        fs: int = 44_100,
        info: Optional[str] = None,
        task_idx: int = 0,
        num_tasks: int = 4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # ── shared backbone (entire AudioDeepFakeDetectionModel is kept
        #    but its built-in bm_weights is REPLACED by the per-task bank below)
        self._backbone = AudioDeepFakeDetectionModel(input_channels, n_fft, hop_length, fs)

        freq_bins  = n_fft // 2 + 1
        hidden_dim = 64

        # ── per-task beamformer bank: all initialised with the same weights
        #    as _backbone.bm_weights so that task-0 fine-tuning is equivalent
        #    to the baseline (no architecture change on first task)
        self.beamformers = nn.ModuleList([
            AdaptiveComplexBeamformer(input_channels, freq_bins, hidden_dim)
            for _ in range(num_tasks)
        ])
        init_state = self._backbone.bm_weights.state_dict()
        for bf in self.beamformers:
            bf.load_state_dict(init_state)

        self.lr        = lr
        self.criterion = _make_criterion(train_label_1, train_label_0)
        self.eer_computation_train = EER()
        self.eer_computation_val   = EER()
        self.eer_computation_test  = EER()
        self._num_trained = 0   # tasks fully trained so far (set by activate_task)

        # Activate current task immediately
        self.activate_task(task_idx)

    # ── spatial + backbone split ─────────────────────────────────────────────
    def _beamform(self, stft_audio, task_idx: int):
        return self.beamformers[task_idx](stft_audio)

    def _shared_forward(self, weighted_sum, complex_weights):
        """Feature extraction → heatmap → CNN → GRU → linear (shared path)."""
        m = self._backbone

        _mag2    = weighted_sum.real ** 2 + weighted_sum.imag ** 2
        _mag     = torch.sqrt(_mag2 + 1e-12)
        features = torch.stack([
            torch.log(_mag2 + 1e-6),
            weighted_sum.imag / (_mag + 1e-8),
            weighted_sum.real / (_mag + 1e-8),
        ], dim=1)

        hm       = m.heatmap(features)
        features = features * hm

        features = m.activation(m.bn1(m.conv1(features)))
        features = m.max_pool1(features) + m.avg_pool1(features)
        features = m.activation(m.bn2(m.conv2(features)))
        features = m.max_pool2(features) + m.avg_pool2(features)
        features = m.activation(m.bn3(m.conv3(features)))
        features = m.max_pool3(features) + m.avg_pool3(features)

        features = features.permute(0, 3, 2, 1)
        features = features.reshape(features.shape[0], features.shape[1], -1)

        features, _ = m.gru1(features)
        features, _ = m.gru2(features)

        latent = features[:, -1, :]
        logits = m.fc(latent)
        return logits, latent

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        stft = [self._backbone.tf_transform(mic) for mic in x]
        stft = torch.stack(stft, dim=1)

        if self.training:
            # Single beamformer (current task)
            ws, cw = self._beamform(stft, self.hparams.task_idx)
            logits, latent = self._shared_forward(ws, cw)
            return logits, cw, latent
        else:
            # Ensemble: average logits across all trained beamformers
            n_heads = max(self._num_trained, 1)
            all_logits = []
            last_cw    = None
            for k in range(n_heads):
                ws, cw = self._beamform(stft, k)
                logits, _ = self._shared_forward(ws, cw)
                all_logits.append(logits)
                last_cw = cw
            ensemble = torch.stack(all_logits, dim=0).mean(0)
            return ensemble, last_cw, None

    # ── task management ──────────────────────────────────────────────────────
    def activate_task(self, task_idx: int) -> None:
        """
        Prepare for training on task `task_idx`.
        - Freezes all beamformers with index < task_idx  (no forgetting)
        - Unfreezes beamformers[task_idx]                 (new spatial head)
        - Backbone remains fully trainable at every step
        Call this before trainer.fit().  It is invoked automatically in __init__.
        """
        for i, bf in enumerate(self.beamformers):
            requires = (i == task_idx)
            for p in bf.parameters():
                p.requires_grad_(requires)
        self._num_trained = max(self._num_trained, task_idx + 1)

    # ── steps ────────────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        inputs, labels = batch['waveform'], batch['label']
        outputs, cw, _ = self(inputs)
        loss = (self.criterion(outputs, labels)
                + orthogonality_regularization(cw)
                + l1_regularization(cw))
        if torch.isnan(loss):
            return None
        loss    = torch.clamp(loss, max=1e6)
        outputs = torch.nan_to_num(outputs, nan=0.0)
        self.eer_computation_train.update(outputs.detach(), labels)
        self.log('train_loss', loss,                       prog_bar=True, on_epoch=True, on_step=True)
        self.log('train_eer',  self.eer_computation_train, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = _base_step(self, batch, self.eer_computation_val)
        if loss is None:
            return None
        self.log('val_loss', loss,                     prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_eer',  self.eer_computation_val, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _, _ = _base_step(self, batch, self.eer_computation_test)
        if loss is None:
            return None
        self.log('test_loss', loss,                      prog_bar=True, on_epoch=True, on_step=False)
        self.log('test_err',  self.eer_computation_test, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        # Only optimise parameters with requires_grad=True
        # (frozen beamformers are excluded automatically)
        params = [p for p in self.parameters() if p.requires_grad]
        return _make_optimizer(params, self.lr)
