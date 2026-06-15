from scipy.spatial.distance import cdist
import numpy as np
import torch

def solve_ot(G, S, eps=0.05, tau=1.0, metric="sqeuclidean", num_iters=1000, tol=1e-9):

    C = torch.as_tensor(
        cdist(np.asarray(G, dtype=np.float64), np.asarray(S, dtype=np.float64), metric=metric),
        dtype=torch.float64,
    )
    n, m = C.shape
    p = torch.full((n,), 1.0 / n, dtype=torch.float64)
    q = torch.full((m,), 1.0 / m, dtype=torch.float64)
    reg = eps * tau
    u, v = torch.zeros(n, dtype=torch.float64), torch.zeros(m, dtype=torch.float64)
    H = lambda u, v: (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / reg
    for _ in range(num_iters):
        u_old, v_old = u, v
        u = reg * (torch.log(p + 1e-8) - torch.logsumexp(H(u, v), dim=-1)) + u
        v = reg * (torch.log(q + 1e-8) - torch.logsumexp(H(u, v).transpose(-1, -2), dim=-1)) + v
        if ((u - u_old).abs().sum() + (v - v_old).abs().sum()).item() < tol:
            break
    return torch.exp(H(u, v))


def solve_uot(G, S, eps=0.05, tau=1.0, reg_m=(1.0, 1.0), metric="sqeuclidean", num_iters=1000, tol=1e-9):

    C = torch.as_tensor(
        cdist(np.asarray(G, dtype=np.float64), np.asarray(S, dtype=np.float64), metric=metric),
        dtype=torch.float64,
    )
    n, m = C.shape
    p = torch.full((n,), 1.0 / n, dtype=torch.float64)
    q = torch.full((m,), 1.0 / m, dtype=torch.float64)
    reg = eps * tau
    beta_abstract, beta_neural = reg_m
    lam_abstract = beta_abstract / (beta_abstract + reg)   # relax variable marginal
    lam_neural = beta_neural / (beta_neural + reg)          # relax site marginal
    u, v = torch.zeros(n, dtype=torch.float64), torch.zeros(m, dtype=torch.float64)
    H = lambda u, v: (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / reg
    for _ in range(num_iters):
        u_old, v_old = u, v
        u = lam_abstract * (reg * (torch.log(p + 1e-8) - torch.logsumexp(H(u, v), dim=-1)) + u)
        v = lam_neural * (reg * (torch.log(q + 1e-8) - torch.logsumexp(H(u, v).transpose(-1, -2), dim=-1)) + v)
        if ((u - u_old).abs().sum() + (v - v_old).abs().sum()).item() < tol:
            break
    return torch.exp(H(u, v))