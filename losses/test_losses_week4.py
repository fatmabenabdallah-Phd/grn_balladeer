# imports adjusted to match the grn_balladeer package convention

import torch

from grn_balladeer.losses.harmonic_loss import compute_consonance_degree, harmonic_loss, all_pairs_edge_index, CONSONANCE_RATIOS
from grn_balladeer.losses.symbolic_loss import get_frontal_pairs, determine_rule_direction, symbolic_implication_loss, FRONTAL_CHANNELS
from grn_balladeer.losses.total_loss import total_loss

print("=== TOY TESTS ===")

# 1. compute_consonance_degree: exact ratio -> mu should be ~1.0
omega_i = torch.tensor([2.0])
omega_j = torch.tensor([1.0])  # ratio 2.0, exactly in CONSONANCE_RATIOS
mu = compute_consonance_degree(omega_i, omega_j, sigma=0.1)
print("mu (exact octave ratio, expect ~1.0):", mu.item())
assert mu.item() > 0.99

# far-from-any-ratio case -> mu should be near 0
omega_i2 = torch.tensor([3.7])
omega_j2 = torch.tensor([1.0])  # ratio 3.7, far from all ratios
mu2 = compute_consonance_degree(omega_i2, omega_j2, sigma=0.1)
print("mu (far ratio, expect ~0):", mu2.item())
assert mu2.item() < 0.01

# 2. harmonic_loss: single pair, exact-ratio match -> loss ~0; off-ratio -> loss > 0
# NOTE: recalibrated for the neuroscience-grounded CONSONANCE_RATIOS=[1,2,3,4]
# (the original 4-node/6-pair toy example was hand-picked against the old,
# denser musical-consonance ratio set and no longer demonstrates the intended
# contrast now that the ratio set is coarser - a single clear pair is less
# ambiguous than trying to hand-construct a 4-node example where every one
# of 6 pairwise ratios simultaneously lands near a widely-spaced target).
edges2 = all_pairs_edge_index(2)

omega_consonant = torch.tensor([2.0, 1.0])  # ratio 2.0, exact match
loss_mixed = harmonic_loss(omega_consonant, edges2)
print("harmonic_loss (exact ratio=2.0 match):", loss_mixed.item())

omega_dissonant = torch.tensor([2.7, 1.0])  # ratio 2.7, far from {1,2,3,4}
loss_dissonant = harmonic_loss(omega_dissonant, edges2)
print("harmonic_loss (off-ratio=2.7, expect higher):", loss_dissonant.item())
assert loss_dissonant.item() > loss_mixed.item()

# gradient flow check
omega_grad = torch.tensor([2.0, 1.0], requires_grad=True)
loss_g = harmonic_loss(omega_grad, edges2)
loss_g.backward()
print("harmonic_loss grad flows:", omega_grad.grad is not None, omega_grad.grad)

# empty edge_pairs raises
try:
    harmonic_loss(omega_consonant, torch.empty((0, 2), dtype=torch.long))
    print("FAIL: should have raised on empty edge_pairs")
except ValueError as e:
    print("OK, raised on empty edges:", e)

# 4. get_frontal_pairs
ch_names_toy = ["AF7", "Fpz", "F7", "Fz", "T7", "FC6"]
pairs = get_frontal_pairs(ch_names_toy)
print("frontal pairs (toy, 4 frontal chans present -> C(4,2)=6):", pairs.shape, pairs.tolist())
assert pairs.shape[0] == 6

# missing enough frontal channels -> raises
try:
    get_frontal_pairs(["T7", "FC6"])
    print("FAIL: should have raised, no frontal channels present")
except ValueError as e:
    print("OK, raised on missing frontal channels:", e)

# 5. determine_rule_direction on SYNTHETIC data with KNOWN correlation
torch.manual_seed(0)
labels_synth = torch.tensor([0, 0, 0, 1, 1, 1])
mu_direct = torch.tensor([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])  # clearly higher mu <-> label=1
direction, corr = determine_rule_direction(mu_direct, labels_synth)
print("determine_rule_direction (constructed positive corr):", direction, corr)
assert direction == "direct" and corr > 0

mu_inverse = torch.tensor([0.9, 0.85, 0.8, 0.2, 0.15, 0.1])  # higher mu <-> label=0
direction2, corr2 = determine_rule_direction(mu_inverse, labels_synth)
print("determine_rule_direction (constructed negative corr):", direction2, corr2)
assert direction2 == "inverse" and corr2 < 0

# 6. symbolic_implication_loss: perfect consonance + high confidence -> low loss (direct)
mu_high = torch.tensor([0.95, 0.9])
conf_high = torch.tensor([0.9, 0.95])
l_symb_low = symbolic_implication_loss(mu_high, conf_high, direction="direct")
print("symbolic_implication_loss (high mu, high conf, direct, expect low):", l_symb_low.item())

# high consonance but LOW confidence -> should give a higher loss (direct)
conf_low = torch.tensor([0.1, 0.05])
l_symb_high = symbolic_implication_loss(mu_high, conf_low, direction="direct")
print("symbolic_implication_loss (high mu, low conf, direct, expect higher):", l_symb_high.item())
assert l_symb_high.item() > l_symb_low.item()

# inverse direction should flip which confidence is "good"
l_symb_inv = symbolic_implication_loss(mu_high, conf_low, direction="inverse")
print("symbolic_implication_loss (high mu, low conf, INVERSE, expect low):", l_symb_inv.item())
assert l_symb_inv.item() < l_symb_high.item()

# 7. total_loss combiner
lt = torch.tensor(0.5)
lh = torch.tensor(0.2)
ls = torch.tensor(0.1)
ltot = total_loss(lt, lh, ls)
print("total_loss (triplet defaulted to 0):", ltot.item())
assert abs(ltot.item() - 0.8) < 1e-6

ltot_with_triplet = total_loss(lt, lh, ls, l_triplet=torch.tensor(0.3), lambda3=2.0)
print("total_loss (with triplet term, lambda3=2.0):", ltot_with_triplet.item())
assert abs(ltot_with_triplet.item() - (0.5 + 0.2 + 0.1 + 2.0*0.3)) < 1e-6

print("\n=== ALL TOY TESTS PASSED ===\n")

print("\n=== REAL DATA TEST ===")
print("Skipped in this checked-in version: the checkpoint this section")
print("depends on (real_embeddings.pt) is a local artifact from a prior")
print("session, not committed to the repo. See Week 4 Notion notes for")
print("an independently-reproduced real-data run (fresh untrained")
print("GRNEncoder on UB0136), which confirmed the same omega-collapse")
print("diagnosis this test file's docstring describes.")
