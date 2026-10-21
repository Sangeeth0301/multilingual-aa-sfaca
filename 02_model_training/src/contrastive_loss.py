"""
contrastive_loss.py
--------------------
Implements the contrastive objective for SFA-CA, adapted from OTBDetector's
contrastive strategy (La Cava & Tagarelli, EMNLP 2025).

In per-cluster training, batches contain texts from a single script/cluster,
so the contrastive loss operates within that cluster: within a batch, texts
from the SAME generator are pulled closer, while texts from DIFFERENT generators
are pushed apart. Note: only in the monolithic multi-cluster baseline (--cluster all)
do batches mix scripts, producing cross-script contrastive pairs.

We use a standard supervised contrastive loss (SupCon-style): within a
batch, for each sample, positives are all other samples with the same
generator label; negatives are all samples with a different label.
"""

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    embeddings: [batch_size, hidden_dim] -- pooled representations from
                SFACAModel.get_embedding(...)
    labels:     [batch_size] -- generator class ids (0..7)
    temperature: scaling factor for the similarity scores; lower = sharper

    Returns a scalar loss.
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]

    # Normalize embeddings so dot product = cosine similarity.
    embeddings = F.normalize(embeddings, p=2, dim=1)
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    # Mask out self-similarity (diagonal).
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    similarity_matrix.masked_fill_(self_mask, float("-inf"))

    # Positive mask: same label, not self.
    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.T) & (~self_mask)

    # For numerical stability, subtract row-wise max before exponentiating.
    row_max, _ = similarity_matrix.max(dim=1, keepdim=True)
    logits = similarity_matrix - row_max.detach()

    exp_logits = torch.exp(logits)
    exp_logits = exp_logits.masked_fill(self_mask, 0.0)

    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    # IMPORTANT: log_prob contains -inf on the diagonal (self-similarity,
    # intentionally excluded). Multiplying that -inf by positive_mask's 0
    # would give nan (IEEE float rule: 0 * -inf = nan), not 0. So instead
    # of `positive_mask * log_prob`, we use torch.where to safely zero out
    # every position we don't want, without ever multiplying by -inf.
    zeros = torch.zeros_like(log_prob)
    safe_log_prob = torch.where(positive_mask, log_prob, zeros)

    # Average log-probability over positive pairs per anchor.
    num_positives = positive_mask.sum(dim=1)
    # Avoid division by zero for anchors with no positives in this batch.
    safe_num_positives = num_positives.clamp(min=1)

    mean_log_prob_pos = safe_log_prob.sum(dim=1) / safe_num_positives

    # Anchors with zero positives contribute zero loss (can't be trained on).
    loss_per_anchor = -mean_log_prob_pos
    loss_per_anchor = loss_per_anchor * (num_positives > 0).float()

    valid_anchors = (num_positives > 0).sum().clamp(min=1)
    return loss_per_anchor.sum() / valid_anchors


def combined_loss(
    logits: torch.Tensor,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    contrastive_weight: float = 0.5,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Combines standard cross-entropy classification loss with the
    supervised contrastive loss. Tune `contrastive_weight` (0..1) as
    a hyperparameter -- start at 0.5 and adjust based on validation F1.
    """
    ce_loss = F.cross_entropy(logits, labels)
    con_loss = supervised_contrastive_loss(embeddings, labels, temperature)
    return (1 - contrastive_weight) * ce_loss + contrastive_weight * con_loss


if __name__ == "__main__":
    # Quick smoke test with random data.
    torch.manual_seed(0)
    emb = torch.randn(6, 32)
    lbl = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = supervised_contrastive_loss(emb, lbl)
    print("Contrastive loss (random data):", loss.item())
