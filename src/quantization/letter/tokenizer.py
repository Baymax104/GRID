"""对照作者 RQ-VAE/models/{rqvae,rq,vq,layers}.py 的独立实现。"""

import random
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

LETTER_COMMIT = "8d0154e28de37dbb6e24871c508ad8ddb1921cda"


def mlp(dimensions: Sequence[int], dropout: float = 0.0, batch_norm: bool = False) -> nn.Sequential:
    modules = []
    for index, (source, target) in enumerate(zip(dimensions[:-1], dimensions[1:], strict=True)):
        modules.extend([nn.Dropout(dropout), nn.Linear(source, target)])
        if batch_norm:
            modules.append(nn.BatchNorm1d(target))
        if index < len(dimensions) - 2:
            modules.append(nn.ReLU())
    model = nn.Sequential(*modules)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    return model


def squared_distances(values: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    return values.square().sum(-1, keepdim=True) + codes.square().sum(-1)[None] - 2 * values @ codes.T


@torch.no_grad()
def sinkhorn_assign(distances: torch.Tensor, epsilon: float, iterations: int) -> torch.Tensor:
    """作者距离中心化与行列平衡；double 降低指数下溢风险。"""
    if epsilon <= 0 or iterations < 1:
        raise ValueError("Sinkhorn requires positive epsilon and iterations.")
    middle = (distances.max() + distances.min()) / 2
    centered = (distances - middle) / (distances.max() - middle + 1e-5)
    weights = torch.exp(-centered.double() / epsilon)
    weights /= weights.sum()
    for _ in range(iterations):
        weights /= weights.sum(1, keepdim=True)
        weights /= distances.shape[0]
        weights /= weights.sum(0, keepdim=True)
        weights /= distances.shape[1]
    if not torch.isfinite(weights).all():
        raise ValueError("Non-finite Sinkhorn assignments.")
    return weights.argmax(-1)


def constrained_clusters(values: torch.Tensor, count: int, *, initialization: bool, seed: int, n_jobs: int = 10):
    """保持作者 constrained K-means 的容量、迭代数与重复次数。"""
    from k_means_constrained import KMeansConstrained

    if len(values) < count * 2:
        raise ValueError("Constrained clustering requires at least two values per cluster.")
    minimum = min(len(values) // (count * 2), 50 if initialization else 10)
    maximum = minimum * 4 if initialization else count * 6
    if count * maximum < len(values):
        raise ValueError("Constrained clustering capacity is too small for these dimensions.")
    cluster = KMeansConstrained(
        n_clusters=count,
        size_min=minimum,
        size_max=maximum,
        max_iter=10,
        n_init=10,
        n_jobs=n_jobs,
        random_state=seed,
    ).fit(values.detach().cpu().float().numpy())
    return (
        torch.as_tensor(cluster.cluster_centers_, device=values.device, dtype=values.dtype),
        torch.as_tensor(cluster.labels_, device=values.device, dtype=torch.long),
    )


def diversity_loss(
    codes: torch.Tensor, ids: torch.Tensor, labels: torch.Tensor, positives: torch.Tensor | None = None
) -> torch.Tensor:
    if positives is None:
        # 批量读取采样输入，避免按样本读取 CUDA label；成员顺序与原 nonzero 一致。
        cpu_labels = labels.tolist()
        groups = {}
        for index, group in enumerate(cpu_labels):
            groups.setdefault(group, []).append(index)
        sampled = []
        for index in ids.tolist():
            choices = [other for other in groups[cpu_labels[index]] if other != index]
            if not choices:
                raise ValueError("Diversity positive cluster must contain a non-self code.")
            sampled.append(random.choice(choices))
        positives = torch.tensor(sampled, device=codes.device)
    if (positives == ids).any() or not torch.equal(labels[positives], labels[ids]):
        raise ValueError("Diversity positives must be distinct codes in the same cluster.")
    logits = codes[ids] @ codes.T
    logits = logits.scatter(1, ids[:, None], -1e12)
    return F.cross_entropy(logits, positives)


class LetterTokenizer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        codebook_size: int = 256,
        num_layers: int = 4,
        hidden_sizes: Sequence[int] = (2048, 1024, 512, 256, 128, 64),
        alpha: float = 0.01,
        beta: float = 0.0001,
        mu: float = 0.25,
        quant_loss_weight: float = 1.0,
        num_groups: int = 10,
        sk_epsilons: Sequence[float] | None = None,
        sk_iterations: int = 50,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()
        if min(input_dim, latent_dim, codebook_size, num_layers, num_groups) < 1:
            raise ValueError("Tokenizer dimensions must be positive.")
        if min(alpha, beta, mu, quant_loss_weight) < 0:
            raise ValueError("Tokenizer loss weights must be nonnegative.")
        self.latent_dim, self.codebook_size, self.num_layers = latent_dim, codebook_size, num_layers
        self.alpha, self.beta, self.mu = alpha, beta, mu
        self.quant_loss_weight, self.num_groups = quant_loss_weight, num_groups
        self.sk_epsilons = tuple(sk_epsilons or ([0.0] * (num_layers - 1) + [0.003]))
        if len(self.sk_epsilons) != num_layers:
            raise ValueError("One Sinkhorn epsilon is required per codebook.")
        self.sk_iterations = sk_iterations
        dimensions = [input_dim, *hidden_sizes, latent_dim]
        self.encoder = mlp(dimensions, dropout, batch_norm)
        self.decoder = mlp(dimensions[::-1], dropout, batch_norm)
        self.codebooks = nn.Parameter(torch.zeros(num_layers, codebook_size, latent_dim))
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("group_labels", torch.full((num_layers, codebook_size), -1, dtype=torch.long))

    @torch.no_grad()
    def initialize(self, features: torch.Tensor, seed: int = 42):
        residual = self.encoder(features)
        for level in range(self.num_layers):
            centers, _ = constrained_clusters(residual, self.codebook_size, initialization=True, seed=seed)
            self.codebooks[level].copy_(centers)
            ids = self._assign(residual, level, use_sk=True)
            residual = residual - centers[ids]
        self.initialized.fill_(True)
        self.update_groups(seed)

    @torch.no_grad()
    def update_groups(self, seed: int = 42):
        for level in range(self.num_layers):
            _, labels = constrained_clusters(self.codebooks[level], self.num_groups, initialization=False, seed=seed)
            counts = torch.bincount(labels, minlength=self.num_groups)
            if (counts < 2).any():
                raise ValueError("Diversity grouping contains singleton clusters.")
            self.group_labels[level].copy_(labels)

    def _assign(self, residual: torch.Tensor, level: int, use_sk: bool):
        distances = squared_distances(residual, self.codebooks[level])
        epsilon = self.sk_epsilons[level]
        if use_sk and epsilon > 0:
            return sinkhorn_assign(distances, epsilon, self.sk_iterations)
        return distances.argmin(-1)

    def quantize(
        self, latent: torch.Tensor, *, use_sk: bool, compute_loss: bool, positives: Sequence[torch.Tensor] | None = None
    ):
        if not self.initialized:
            raise ValueError("LETTER codebooks must be initialized before quantization.")
        residual, total = latent, torch.zeros_like(latent)
        indices, vq_losses, div_losses = [], [], []
        for level in range(self.num_layers):
            ids = self._assign(residual, level, use_sk)
            values = self.codebooks[level][ids]
            if compute_loss:
                if (self.group_labels[level] < 0).any():
                    raise ValueError("Codebook groups have not been initialized.")
                div = diversity_loss(
                    self.codebooks[level],
                    ids,
                    self.group_labels[level],
                    None if positives is None else positives[level],
                )
                loss = F.mse_loss(values, residual.detach()) + self.mu * F.mse_loss(values.detach(), residual)
                vq_losses.append(loss + self.beta * div)
                div_losses.append(div)
            # 保留作者逐层 STE：CF 不直接更新 codebook，后层 commitment 不回传 encoder。
            straight_through = residual + (values - residual).detach()
            total = total + straight_through
            residual = residual - straight_through
            indices.append(ids)
        zero = latent.sum() * 0
        return (
            total,
            torch.stack(indices, -1),
            (torch.stack(vq_losses).mean() if vq_losses else zero),
            (torch.stack(div_losses).mean() if div_losses else zero),
        )

    def forward(self, features: torch.Tensor, cf: torch.Tensor, positives=None):
        if cf.shape != (len(features), self.latent_dim) or not torch.isfinite(cf).all():
            raise ValueError("CF embeddings must be finite and match tokenizer latent dimensions.")
        latent = self.encoder(features)
        quantized, ids, vq, div = self.quantize(latent, use_sk=True, compute_loss=True, positives=positives)
        reconstruction = self.decoder(quantized)
        recon = F.mse_loss(reconstruction, features)
        cf_loss = F.cross_entropy(quantized @ cf.detach().T, torch.arange(len(features), device=features.device))
        return {
            "loss": recon + self.quant_loss_weight * vq + self.alpha * cf_loss,
            "reconstruction_loss": recon,
            "quantization_loss": vq,
            "cf_loss": cf_loss,
            "diversity_loss": div,
            "indices": ids,
            "quantized": quantized,
            "reconstruction": reconstruction,
        }

    @torch.no_grad()
    def encode(self, features: torch.Tensor, use_sk: bool = False):
        return self.quantize(self.encoder(features), use_sk=use_sk, compute_loss=False)[1]

    @torch.no_grad()
    def unique_codes(self, features: torch.Tensor, max_rounds: int = 20):
        """作者式修复后，GRID在末层容量内以硬匹配消除残余碰撞。"""
        latent = self.encoder(features)
        ids = self.encode(features)
        for _ in range(max_rounds):
            _, inverse, counts = torch.unique(ids, dim=0, return_inverse=True, return_counts=True)
            if counts.max() == 1:
                return ids
            for group in (counts > 1).nonzero().flatten():
                rows = (inverse == group).nonzero().flatten()
                residual = latent[rows]
                for level in range(self.num_layers):
                    distances = squared_distances(residual, self.codebooks[level])
                    assigned = (
                        sinkhorn_assign(distances, max(self.sk_epsilons[-1], 0.003), self.sk_iterations)
                        if level == self.num_layers - 1
                        else distances.argmin(-1)
                    )
                    ids[rows, level] = assigned
                    residual = residual - self.codebooks[level][assigned]
        if torch.unique(ids, dim=0).shape[0] != len(ids):
            from scipy.optimize import linear_sum_assignment

            _, prefix_inverse, prefix_counts = torch.unique(ids[:, :-1], dim=0, return_inverse=True, return_counts=True)
            if prefix_counts.max() > self.codebook_size:
                raise ValueError(
                    "LETTER SID export has unresolved collisions: prefix population exceeds final codebook capacity."
                )
            for group in (prefix_counts > 1).nonzero().flatten():
                rows = (prefix_inverse == group).nonzero().flatten()
                if ids[rows, -1].unique().numel() == len(rows):
                    continue
                residual = latent[rows].clone()
                for level in range(self.num_layers - 1):
                    residual -= self.codebooks[level][ids[rows, level]]
                distances = squared_distances(residual, self.codebooks[-1]).double().cpu().numpy()
                row_indices, codes = linear_sum_assignment(distances)
                ids[rows[torch.as_tensor(row_indices, device=rows.device)], -1] = torch.as_tensor(
                    codes, device=ids.device
                )
            if torch.unique(ids, dim=0).shape[0] != len(ids):
                raise ValueError("LETTER SID export has unresolved collisions after hard assignment.")
        return ids
