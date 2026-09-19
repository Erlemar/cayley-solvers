# --- Quick sanity eval: greedy P_B on exact-depth states ---
# BFS a few shells from solved and check greedy-P_B optimality. (Real solving uses
# the wide beam in the companion inference notebook; greedy only probes shallow.)
def bfs_shells(gens, solved, max_depth, per):
    rng = np.random.default_rng(0); seen = {solved.astype(np.uint8).tobytes()}
    frontier = solved[None].astype(np.uint8); out = {}
    for d in range(1, max_depth + 1):
        ch = np.unique(frontier[:, gens].reshape(-1, frontier.shape[1]), axis=0)
        keep = np.fromiter((c.tobytes() not in seen for c in ch), bool, len(ch))
        frontier = ch[keep]
        for c in frontier: seen.add(c.tobytes())
        idx = rng.choice(len(frontier), min(per, len(frontier)), replace=False)
        out[d] = frontier[idx].astype(np.int32)
    return out

ph = jax.tree_util.tree_map(lambda x: x[0], P)
mdl = eqx.combine(ph, static)
shells = bfs_shells(gens, solved, 4, 64)
for d, st in shells.items():
    done, ln = greedy(mdl, jnp.asarray(st), 4 * d + 8)
    opt = float(((ln == d) & done).mean())
    print(f"depth {d}: greedy solve {float(done.mean()):.2f}  optimal {opt:.2f}")
print("(shallow greedy is just a sanity check; wide-beam solving is the real metric)")
