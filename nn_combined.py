import sys; args = sys.argv[1:]
import math
import time
import random

# sigmoid / logistic — same thing, different vibes
def sigmoid(x):
    if x < -15: return 0  # clamp to avoid math range errors
    if x > 15: return 1
    return 1.0 / (1.0 + math.exp(-x))

def dsigmoid(y):  # this takes the activation y, NOT the net input. y(1-y) is the deriv
    if y < -15: return 0  # honestly this branch shouldnt happen but ehh better safe
    if y > 15: return 1
    return y * (1.0 - y)

def dot(u, v):  # plain dot product, zip handles the length
    return sum(a * b for a, b in zip(u, v))


def parse_file(path):
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    data = []
    for ln in lines:                          # parse each training pair
        left, right = ln.split('=>')
        x = [float(v) for v in left.split()]
        x.append(1.0)                         # dc offset / bias from the lab prompt
        y = [float(v) for v in right.split()]
        data.append((x, y))
    return data


def init_weights(layers):
    """
    layers: [n_in, hidden, n_out, n_out]
    weights stored as flat lists, ordered [j][i] -> j*n_in+i, same as the lab spec
    final layer is 1->1 elementwise so it's just one weight per output
    """
    W = []
    nlayers = len(layers)
    for l in range(nlayers - 1):
        n_in = layers[l]
        n_out = layers[l + 1]
        if l == nlayers - 2:  # final 1-1 mapping (no full matrix needed)
            W.append([random.uniform(-1, 1) for _ in range(n_in)])
        else:
            W.append([random.uniform(-1, 1) for _ in range(n_in * n_out)])
    return W


def feed_forward(x, W):
    """
    runs the net forward and stores ALL activations along the way for backprop
    returns (activations_list, final_output)
    """
    activations = [x[:]]
    current = x[:]
    for layer in W[:-1]:                       # all layers except the final 1-1
        n_in = len(current)
        n_out = len(layer) // n_in
        nxt = []
        for j in range(n_out):
            nodew = layer[j * n_in:(j + 1) * n_in]   # weights for the j-th node
            nxt.append(sigmoid(dot(current, nodew)))
        current = nxt
        activations.append(current)
    # final 1-1 elementwise multiply, no sigmoid
    final_w = W[-1]
    out = [c * w for c, w in zip(current, final_w)]
    activations.append(out)
    return activations, out


def backprop(A, W, target, hidden, n_out):
    """
    classic backprop — gradients per layer, returned as a list-of-lists matching W.
    not super pretty but it follows the chain rule from class
    """
    n_in = len(A[0])
    grad_w0 = [0.0] * len(W[0])
    grad_w1 = [0.0] * len(W[1])
    grad_w2 = [0.0] * len(W[2])

    # output layer error: (out - target)  -> from the 0.5(t-o)^2 formula
    delout = [A[-1][k] - target[k] for k in range(n_out)]

    # final 1-1 layer (no sigmoid, just elementwise mult)
    del2 = []
    for k in range(n_out):
        d_a2 = delout[k] * W[-1][k]
        del2.append(d_a2 * dsigmoid(A[2][k]))
        grad_w2[k] += delout[k] * A[2][k]

    # hidden -> output (full matrix W[1])
    del1 = [0.0] * hidden
    for j in range(hidden):
        d_a1 = 0.0
        for k in range(n_out):
            d_a1 += del2[k] * W[1][k * hidden + j]
        del1[j] = d_a1 * dsigmoid(A[1][j])
    for k in range(n_out):
        for j in range(hidden):
            grad_w1[k * hidden + j] += del2[k] * A[1][j]

    # input -> hidden (full matrix W[0])
    for j in range(hidden):
        for i in range(n_in):
            grad_w0[j * n_in + i] += del1[j] * A[0][i]

    return [grad_w0, grad_w1, grad_w2]


def mse(out, target):
    return 0.5 * sum((t - o) ** 2 for t, o in zip(target, out))


def dump_weights(layers, W):
    return "\n".join([
        f"Layer counts {' '.join(str(c) for c in layers)}",
        " ".join(str(w) for w in W[0]),
        " ".join(str(w) for w in W[1]),
        " ".join(str(w) for w in W[2]),
    ])


def main():
    training = parse_file(args[0])
    n_in = len(training[0][0])      # input dim already includes bias
    n_out = len(training[0][1])
    hidden = 2                       # following recommended layer counts
    layers = [n_in, hidden, n_out, n_out]

    # hyperparams — alpha is the learning rate, decays a bit after a while
    alpha = 0.1
    max_epochs = 30000
    threshold = 0.005

    W = init_weights(layers)
    best_err = float('inf')
    best_dump = ""
    rounds = 0
    start = time.time()
    deadline = 29       # time limit (seconds), give a buffer before the grader times out

    while time.time() - start < deadline and rounds < max_epochs:
        # learning rate schedule — drops a bit at 1k to fine tune
        if rounds == 1000:
            alpha *= 0.9

        total_err = 0.0
        # accumulate grads across the full training set (batch GD)
        grads_total = [[0.0] * len(w) for w in W]
        for x, y in training:
            A, out = feed_forward(x, W)
            total_err += mse(out, y)
            grads = backprop(A, W, y, hidden, n_out)
            for li in range(3):
                for i in range(len(grads_total[li])):
                    grads_total[li][i] += grads[li][i]

        # apply update — straight gradient descent step
        for li in range(3):
            for i in range(len(W[li])):
                W[li][i] -= alpha * grads_total[li][i]

        # track best run so far so we never report worse than what we found
        if total_err < best_err:
            best_err = total_err
            best_dump = dump_weights(layers, W)

        # early stop if we're basically done
        if total_err < 0.01:
            break

        rounds += 1
        # if we're stuck in a local min after 5k rounds and still bad, re-roll
        if rounds > 5000 and total_err > 0.1:
            W = init_weights(layers)
            rounds = 0

        # secondary threshold check (from the structured version)
        if total_err < threshold:
            break

    print(best_dump)


if __name__ == "__main__":
    main()
# michelle, p4, 2027
