"""One check: the intervention penalty prices cross-word pairs only.

Hand-computed on 3 vectors (words A, A, B): the only cross-word cosines are
(v0,v2) and (v1,v2); same-word (v0,v1) and diagonals must not contribute.
"""
import torch
import torch.nn.functional as F


def penalty(resid, y, m, word_of):
    """Mirror of the lam branch in harness.train."""
    w = word_of[y[:, :-1].long()]
    sel = (m[:, :-1] > 0) & (w >= 0)
    H = F.normalize(resid[:, :-1][sel], dim=-1)
    ww = w[sel]
    cross = ww[:, None] != ww[None, :]
    return ((H @ H.T)[cross] ** 2).mean()


def main():
    # selected vectors ([:, :-1] drops t=2; m masks batch1 t=1):
    #   v0=(1,0) word A, v1=(1,1)/√2 word A, v2=(0,1) word B
    resid = torch.tensor([[[1.0, 0.0], [1.0, 1.0], [9.0, 9.0]],
                          [[0.0, 1.0], [5.0, 5.0], [9.0, 9.0]]])
    y = torch.tensor([[10, 11, 0], [12, 0, 0]])   # targets: cont A, cont A, cont B
    m = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    word_of = torch.full((13,), -1, dtype=torch.long)
    word_of[10] = word_of[11] = 0                 # word A
    word_of[12] = 1                               # word B
    got = penalty(resid, y, m, word_of).item()
    # cross pairs: (v0,v2) cos=0, (v1,v2) cos=1/√2; each twice by symmetry
    # -> mean of {0, 0, 0.5, 0.5} = 0.25. Same-word (v0,v1) cos²=0.5 excluded:
    # if it leaked in, mean would be 1/3.
    assert abs(got - 0.25) < 1e-6, got
    resid[0, 1] = torch.tensor([1.0, 0.0])        # now (v0,v1) cos=1 (same word)
    got = penalty(resid, y, m, word_of).item()
    # cross cosines now: (e1,e1)=1 twice, (e1,e2)... recompute: v0=e1, v1=e1, v2=e2
    # cross pairs (v0,v2)=0, (v1,v2)=0 -> mean 0; same-word cos=1 excluded
    assert abs(got - 0.0) < 1e-6, got
    # masked-out position (m=0) must not enter: flip its word to B, nothing changes
    y[1, 1] = 12
    assert abs(penalty(resid, y, m, word_of).item() - 0.0) < 1e-6
    print("penalty mask ok")


if __name__ == "__main__":
    main()
