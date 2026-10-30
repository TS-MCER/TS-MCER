"""Expert consensus, local residual refinement, and dual-view evidence fusion."""

import numpy as np
from scipy.special import expit, logit
from scipy.spatial.distance import cdist


def lg(q):
    return logit(np.clip(np.asarray(q, float), 1e-7, 1 - 1e-7))


def normalize(q):
    q = np.asarray(q, float)
    if q.ndim != 2 or not np.isfinite(q).all() or np.any(q < 0) or np.any(q.sum(1) <= 0):
        raise ValueError("Invalid probability matrix")
    return q / q.sum(1, keepdims=True)


def expert_consensus(opinions, clip):
    """Aggregate three context experts with five bounded Huber updates."""
    z = lg(opinions)
    mu = np.median(z, axis=2)
    for _ in range(5):
        mu += np.clip(z - mu[:, :, None], -clip, clip).mean(2)
    return expit(mu)


def risk_coordinates(z):
    return np.stack([lg(z["base_probabilities"]), lg(z["context_probabilities"]).mean(2)], axis=2)


def build_residual_memory(t, clip):
    """Pair fit-local OOF risk coordinates, labels, and consensus expectations."""
    x = risk_coordinates(t)
    mean = x.mean(0); std = x.std(0); std[std < 1e-6] = 1
    nclass = t["base_probabilities"].shape[1]
    if len(np.unique(t["row_index"])) != len(x) or len(np.unique(t["fold"])) < 2:
        raise ValueError("Memory requires unique rows with genuine OOF provenance")
    if not np.array_equal(np.unique(t["y"]), np.arange(nclass)):
        raise ValueError("Memory is missing a class")
    return dict(x=np.clip((x - mean) / std, -5, 5), mean=mean, std=std,
                labels=(t["y"][:, None] == np.arange(nclass)).astype(float),
                qsource=expert_consensus(t["context_probabilities"], clip), row_index=t["row_index"].copy(),
                local_fold=t["fold"].copy(), huber_clip=np.array(clip))


def neighbor_geometry(z, state, max_k):
    if np.intersect1d(z["row_index"], state["row_index"]).size:
        raise ValueError("Query rows overlap the local residual refinement fit memory")
    x = np.clip((risk_coordinates(z) - state["mean"]) / state["std"], -5, 5)
    orders, distances = [], []
    for c in range(x.shape[1]):
        d = cdist(x[:, c], state["x"][:, c], "sqeuclidean")
        order = np.argsort(d, axis=1, kind="stable")[:, :max_k]
        orders.append(order); distances.append(np.take_along_axis(d, order, axis=1))
    return np.stack(orders, axis=1), np.stack(distances, axis=1)


def local_residual(state, geom, k, class_power, pseudo):
    order, distance = geom
    order = order[:, :, :k]; distance = distance[:, :, :k]
    observed, expected, ess_list, raw_list = [], [], [], []
    for c in range(state["labels"].shape[1]):
        labels = state["labels"][:, c]
        counts = np.bincount(labels.astype(int), minlength=2)
        if not (counts > 0).all():
            raise ValueError("Local residual refinement needs positive and negative memory support")
        cw = (len(labels) / (2 * counts[labels.astype(int)])) ** class_power
        prior = float(cw @ labels / cw.sum())
        oi, di = order[:, c], distance[:, c]
        w = np.exp(-di / np.maximum(di[:, -1, None], 1e-8)) * cw[oi]
        w /= w.sum(1, keepdims=True)
        ess = 1 / (w * w).sum(1)
        obs = (ess * np.sum(w * labels[oi], 1) + pseudo * prior) / (ess + pseudo)
        exp = (ess * np.sum(w * state["qsource"][:, c][oi], 1) + pseudo * prior) / (ess + pseudo)
        observed.append(obs); expected.append(exp); ess_list.append(ess); raw_list.append(lg(obs) - lg(exp))
    raw = np.column_stack(raw_list)
    return np.clip(raw, -2, 2), dict(ess=np.column_stack(ess_list), raw_delta=raw,
                                   observed=np.column_stack(observed), expected=np.column_stack(expected))


def refine_consensus(z, state, params, geom=None):
    """Refine each class's consensus from local OOF residuals, then normalize."""
    if float(state["huber_clip"]) != params["huber_clip"]:
        raise ValueError("Memory expected probabilities use a different Huber clip")
    # Enforce disjointness even when alpha is zero.
    if np.intersect1d(z["row_index"], state["row_index"]).size:
        raise ValueError("Query rows overlap the local residual refinement fit memory")
    consensus_scores = expert_consensus(z["context_probabilities"], params["huber_clip"])
    consensus_probabilities = normalize(consensus_scores)
    if params["alpha"] == 0:
        return consensus_probabilities, dict(expert_consensus=consensus_probabilities,
            consensus_refined=consensus_probabilities, consensus_scores=consensus_scores,
            refined_scores=consensus_scores, local_residual=np.zeros_like(consensus_scores),
            effective_sample_size=np.zeros_like(consensus_scores))
    k = min(params["k"], len(state["row_index"]))
    if geom is not None and geom[0].shape[2] < k:
        raise ValueError("Cached neighbor geometry is too short for k")
    delta, diag = local_residual(state, neighbor_geometry(z, state, k) if geom is None else geom, k, params["class_power"], params["pseudo"])
    refined_scores = expit(lg(consensus_scores) + params["alpha"] * delta)
    refined_probabilities = normalize(refined_scores)
    return refined_probabilities, dict(expert_consensus=consensus_probabilities,
        consensus_refined=refined_probabilities, consensus_scores=consensus_scores,
        refined_scores=refined_scores, local_residual=delta, effective_sample_size=diag["ess"])


def fuse_dual_view_evidence(consensus_refined, specialized, complete, cfg):
    """Apply bounded positive evidence from specialized versus complete views."""
    evidence = np.clip(lg(specialized) - lg(complete), -cfg["evidence_clip"], cfg["evidence_clip"])
    increment = np.clip(cfg["beta_plus"] * np.maximum(evidence - cfg["tau"], 0),
                        -cfg["increment_clip"], cfg["increment_clip"])
    probabilities = normalize(expit(lg(consensus_refined) + increment))
    inactive = np.all(increment == 0, axis=1)
    probabilities[inactive] = consensus_refined[inactive]
    return probabilities, increment
