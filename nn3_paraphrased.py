import sys; args = sys.argv[1:]
"""
Lab: train a small NN to decide whether a point falls inside a circle.
- input is an inequality like x*x+y*y<=0.9
- net outputs a float in [0,1]; we threshold at 0.5
- grader throws 100k random pts in [-1.5, 1.5]^2
- scoring:
    200 wrong -> 100%, 1000 -> 98%, 2000 -> 90%, 30000 -> 50%
"""

import math, random, time, re

eq_arg = args[0]   # the inequality string, e.g. "x*x+y*y<=0.9"


# ---- core math bits ----
def sig(z):
    return 1.0 / (1.0 + math.exp(-z))

def sig_prime(a):  # input is the activation a, not the net input
    return a * (1.0 - a)

def dotprod(u, v):
    return sum(ui * vi for ui, vi in zip(u, v))


def make_weights(sizes):
    """
    sizes = [n_in, h1, ..., n_out]
    - layer 0 gets the input bias (so fan_in = n_in + 1)
    - hidden layers in between are full matrices, no extra bias dim
    - last layer is a 1->1 elementwise mapping, stored as a flat list
    """
    W = []
    n = len(sizes)
    for li in range(n - 1):
        fan_in  = sizes[li] + 1 if li == 0 else sizes[li]
        fan_out = sizes[li + 1]
        if li == n - 2:                     # final 1-1 layer (one weight per unit)
            W.append([random.uniform(-2, 2) for _ in range(fan_in)])
        else:                               # full (fan_out x fan_in) matrix
            W.append([
                [random.uniform(-2, 2) for _ in range(fan_in)]
                for _ in range(fan_out)
            ])
    return W


def forward(x, W):
    """
    Runs a forward pass and stashes activations along the way.
    - activations[0] = input + bias
    - hidden activations all have a trailing bias 1.0
    - final activation = elementwise mult (no extra bias added)
    Returns (activations, final_output).
    """
    acts = []
    cur = x[:] + [1.0]                      # bias on the input
    acts.append(cur)

    for layer_w in W[:-1]:                  # every hidden layer
        nets = [dotprod(row, cur) for row in layer_w]
        cur  = [sig(n) for n in nets] + [1.0]   # bias on the hidden side too
        acts.append(cur)

    # final 1-1 elementwise mapping
    last = W[-1]
    out  = [cur[i] * last[i] for i in range(len(last))]
    acts.append(out)
    return acts, out


def err(out, t):
    # output is a list, target is a scalar — same as the original mse_loss
    return 0.5 * (t - out[0]) ** 2


def back(acts, W, t):
    """
    Backprop. Returns deltas D such that D[l] aligns with the rows of W[l].
    """
    L = len(W)
    D = [None] * L

    # output layer: scalar (target - prediction)
    D[-1] = [t - acts[-1][0]]

    for li in range(L - 2, -1, -1):
        if li == L - 2:
            # coming back through the 1-1 final layer is a special case
            lw = W[li + 1]
            D[li] = [
                sig_prime(acts[li + 1][i]) * lw[i] * D[li + 1][i]
                for i in range(len(lw))
            ]
        else:
            nw = W[li + 1]
            nd = D[li + 1]
            a_no_bias = acts[li + 1][:-1]   # strip bias so dims line up
            D[li] = [
                sig_prime(a_no_bias[i]) *
                sum(nw[k][i] * nd[k] for k in range(len(nd)))
                for i in range(len(a_no_bias))
            ]
    return D


def step(W, acts, D, lr):
    """
    Returns a NEW weight list updated by one gradient step (doesn't mutate in place).
    """
    nxt = []
    for li, layer_w in enumerate(W):
        a = acts[li]
        d = D[li]
        if li == len(W) - 1:    # flat (final) layer
            nxt.append([layer_w[i] + lr * d[i] * a[i] for i in range(len(layer_w))])
        else:                   # full matrix
            rows = []
            for j, row in enumerate(layer_w):
                rows.append([row[i] + lr * d[j] * a[i] for i in range(len(row))])
            nxt.append(rows)
    return nxt


# ---- target function (the inequality we're trying to learn) ----
def in_region(x, y, op, r2):
    s = x * x + y * y
    if op == '<':  return int(s <  r2)
    if op == '<=': return int(s <= r2)
    if op == '>=': return int(s >= r2)
    if op == '>':  return int(s >  r2)


def make_dataset(n, op, r2):
    xs, ys = [], []
    for _ in range(n):
        px = random.uniform(-1.5, 1.5)
        py = random.uniform(-1.5, 1.5)
        xs.append([px, py])
        ys.append(in_region(px, py, op, r2))
    return xs, ys


def dump_weights(layers, W):
    # what the grader actually wants to see
    show = layers.copy()
    show[0] += 1            # bias counted in the printout
    print("Layer counts:", *show)
    for layer in W:
        if isinstance(layer[0], list):
            for row in layer:
                print(" ".join(map(str, row)), end=' ')
            print()
        else:
            print(" ".join(map(str, layer)))


def main():
    t0 = time.time()

    # parse the inequality, e.g. "x*x+y*y<=0.9"
    m  = re.match(r"x\*x\+y\*y(?P<op>[<>=]+)(?P<r>[0-9.]+)", eq_arg)
    op = m.group('op')
    r2 = float(m.group('r'))

    inputs, targets = make_dataset(10000, op, r2)

    # layer plan. note: a couple of hidden sizes are defined but skipped in
    # the final list — leaving that quirk in place to keep behavior identical
    n_in, n_out = 2, 1
    h1, h2, h3 = 3, 5, 7
    h4, h5     = 9, 7         # unused — leftover from a wider shape
    h6, h7, h8 = 5, 3, 1
    layers = [n_in, h1, h2, h3, h6, h7, h8, n_out]

    # hyperparams
    alpha       = 0.3
    threshold   = 15
    max_epochs  = 30000

    W = make_weights(layers)

    # warm-up: get an initial error before we start updating
    errs = []
    for x, t in zip(inputs, targets):
        _, o = forward(x, W)
        errs.append(err(o, t))
    total = sum(errs)

    epoch = 0
    cut1 = True   # one-shot LR drops (each fires once)
    cut2 = True

    while total >= threshold and epoch < max_epochs:
        # adaptive alpha — knock it down as we get closer
        if cut1 and total < 100:
            alpha /= 10
            cut1 = False
        if cut2 and total < 25:
            alpha /= 3
            cut2 = False
        if epoch == 1000:
            alpha *= 0.9

        # one full pass through the data, updating after every example
        for i, x in enumerate(inputs):
            acts, _   = forward(x, W)
            D         = back(acts, W, targets[i])
            W         = step(W, acts, D, alpha)
            acts, o   = forward(x, W)
            errs[i]   = err(o, targets[i])

        total = sum(errs)
        epoch += 1

        dump_weights(layers, W)

    # ---- evaluate on a fresh 100k sample ----
    test_x, test_y = make_dataset(100000, op, r2)
    hits = misses = 0
    for i in range(100000):
        _, o = forward(test_x[i], W)
        if int(o[0] > 0.5) == test_y[i]:
            hits += 1
        else:
            misses += 1
    print(f"True: {hits} False: {misses} Error: {misses / 100000}")

    # original printed (start - end) so it ends up negative — kept as-is
    print(t0 - time.time())
    layers[0] += 1   # bias accounted for in the printout count


if __name__ == "__main__":
    main()


# michelle, p4, 2027
