import sys; args = sys.argv[1:]
import math
import re

DEBUG = False

# direction -> grid char for printing the layout
LOOKUPTABLE = {
    (): '.', ('N',): 'N', ('E',): 'E', ('S',): 'S', ('W',): 'W',
    ('E', 'N'): 'L', ('N', 'W'): 'J', ('S', 'W'): '7', ('E', 'S'): 'r',
    ('E', 'W'): '-', ('N', 'S'): '|',
    ('E', 'N', 'W'): '^', ('E', 'N', 'S'): '>', ('E', 'S', 'W'): 'v', ('N', 'S', 'W'): '<',
    ('E', 'N', 'S', 'W'): '+'
}

# policy direction strings -> chars (note: different from layout above bc it's policy not edges)
ADJ_POLICY_LOOKUPTABLE = {
    '': '.', 'E': 'R', 'N': 'U', 'S': 'D', 'W': 'L',
    'EN': 'V', 'ENS': 'W', 'ES': 'S', 'ESW': 'T', 'SW': 'E', 'NSW': 'F',
    'NW': 'M', 'ENW': 'N', 'NS': '|', 'EW': '-', 'ENSW': '+', '*': '*',
}


def toggle(edges, u, v):  # helper for the B / E~ directives — toggles an edge both ways
    if v in edges.get(u, set()):
        edges[u].remove(v)
        edges[v].discard(u)
    else:
        edges.setdefault(u, set()).add(v)
        edges.setdefault(v, set()).add(u)


def largest_divisor(n):  # for figuring out width when not given
    a = 1
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            a = i
    return a


def native_edges(size, width):
    """
    builds the default grid-world connections — every cell connects N/E/S/W
    to its neighbor unless it's on the boundary. same as in the slider labs.
    """
    if not width:
        return {i: set() for i in range(size)}
    dc = {}
    for i in range(size):
        nbrs = set()
        if i % width != 0:           nbrs.add(i - 1)         # can go W
        if i % width != width - 1:   nbrs.add(i + 1)         # can go E
        if i >= width:               nbrs.add(i - width)     # can go N
        if i < size - width:         nbrs.add(i + width)     # can go S
        dc[i] = nbrs
    return dc


def parse_vslices(s, size, width):
    """
    parses the V slicing syntax — numbers, ranges, slices, rectangles.
    examples: '4', '2:7', '1:10:2', '3#15' (rect from 3 to 15), '::' (all)
    """
    if not s:
        return []
    out = []
    parts = s.split(',')
    for part in parts:
        part = part.strip()
        if '#' in part:  # rectangle from corner to corner
            p1, p2 = part.split('#', 1)
            c1 = int(p1) if p1 else 0
            c2 = int(p2) if p2 else size - 1
            if c1 < 0: c1 += size
            if c2 < 0: c2 += size
            if not width: continue
            sr, er = sorted((c1 // width, c2 // width))
            sc, ec = sorted((c1 % width,  c2 % width))
            for r in range(sr, er + 1):
                for c in range(sc, ec + 1):
                    out.append(r * width + c)
        elif part.count(':') >= 1:  # slice form, w/ 1 or 2 colons
            bits = part.split(':')
            start = int(bits[0]) if bits[0] != '' else None
            stop  = int(bits[1]) if len(bits) > 1 and bits[1] != '' else None
            step  = int(bits[2]) if len(bits) > 2 and bits[2] != '' else None
            if step == 0:  # would be infinite, just bail
                continue
            sl = slice(start, stop, step)
            a, b, st = sl.indices(size)         # let python normalize for us
            for i in range(a, b, st):
                if 0 <= i < size:
                    out.append(i)
        else:  # just a number, possibly negative
            v = int(part)
            if v < 0: v += size
            out.append(v)
    return out


def grf_parse(lstArgs):
    """
    builds the graph tuple from cli args. supports a mix of styles:
       <size>            -> size (required, first non-flag int)
       <width>           -> optional width (second int)
       G0 / G1           -> policy mode (0 = max reward then min dist, 1 = reward/dist)
       R:<n>             -> default graph reward
       R<v>[:<n>]        -> vertex reward (uses default if no value given)
       B<v>[<dirs>]      -> toggle edges from v in given dirs (or all if none)
       V<slice>[R<n>]    -> apply reward n to a vslice (rw4 style)
       E<slice>~<slice>  -> toggle edges between two slice sets
    returned tuple: (gtyp, edges, width, v_props, e_props, default_rwd, size, g_mode)
    """
    # first pass: figure out graph type / size / width
    size = None
    width = None
    idx = 0
    gtyp = "G"
    if lstArgs and lstArgs[0].upper().startswith('N'):
        gtyp = "N"
        # if it's like 'N16' grab the size from after N
        rest = lstArgs[0][1:]
        if rest.isdigit():
            size = int(rest)
            idx = 1
    if size is None:
        # look for first plain integer
        while idx < len(lstArgs) and not lstArgs[idx].lstrip('-').isdigit():
            idx += 1
        if idx >= len(lstArgs):
            raise ValueError("need a size somewhere!!")
        size = int(lstArgs[idx])
        idx += 1
    # optional width
    if idx < len(lstArgs) and lstArgs[idx].isdigit():
        width = int(lstArgs[idx])
        idx += 1
    elif gtyp == "G":
        # if no width given, pick a reasonable one (square-ish)
        width = math.ceil(math.sqrt(size))
        while size % width != 0:
            width += 1
    else:
        width = 0  # network has no grid width

    # defaults
    default_rwd = 12
    v_props = {}
    e_props = {}
    g_mode = 1
    edges = native_edges(size, width)
    initial = native_edges(size, width)   # snapshot of native edges, for jump detection

    # second pass: directives
    for arg in lstArgs[idx:]:
        a = arg.upper()
        if a == 'G0':
            g_mode = 0
        elif a == 'G1':
            g_mode = 1
        elif a == 'GW2':      # rw4 alias
            g_mode = 0
        elif a == 'GW4':
            g_mode = 1
        elif a.startswith('R:'):
            default_rwd = int(a[2:])
        elif a.startswith('R') and len(a) > 1 and (a[1].isdigit() or a[1] == '-'):
            parts = a[1:].split(':')
            v = int(parts[0])
            if v < 0: v += size
            if len(parts) == 2:
                v_props[v] = {'rwd': int(parts[1])}
            else:
                v_props[v] = {'rwd': default_rwd}
        elif a.startswith('B'):
            # B<v><dirs> — toggle edges at vertex v
            m = re.match(r'B(-?\d+)([NSEW]*)', a)
            if not m: continue
            sq = int(m.group(1))
            if sq < 0: sq += size
            dires = m.group(2)
            if not dires:  # no direction = toggle ALL four
                r, c = sq // width, sq % width
                if r > 0:                    toggle(edges, sq, sq - width)
                if r < (size // width) - 1:  toggle(edges, sq, sq + width)
                if c > 0:                    toggle(edges, sq, sq - 1)
                if c < width - 1:            toggle(edges, sq, sq + 1)
            else:
                for d in dires:
                    if d == 'N' and sq >= width:                 toggle(edges, sq, sq - width)
                    if d == 'S' and sq < size - width:           toggle(edges, sq, sq + width)
                    if d == 'E' and sq % width != width - 1:     toggle(edges, sq, sq + 1)
                    if d == 'W' and sq % width != 0:             toggle(edges, sq, sq - 1)
        elif a.startswith('V'):
            # V<slice>[R<n>] — set reward on a slice of vertices
            m = re.match(r'V([#,0-9:\-]*)R?(\-?\d*)?', a)
            if not m: continue
            slc = m.group(1)
            rwd_str = m.group(2) if m.group(2) else None
            targets = parse_vslices(slc, size, width)
            if rwd_str is None or rwd_str == '':
                for t in targets:
                    v_props[t] = {'rwd': default_rwd}
            else:
                for t in targets:
                    v_props[t] = {'rwd': int(rwd_str)}
        elif a.startswith('E'):
            # E<slc>~<slc> — toggle edges between two slice sets (rw4 style, simplified)
            m = re.match(r'E([!+*~@])?([,0-9:\-]+)([=~])([,0-9:\-]+)', a)
            if m:
                op = m.group(1) or '~'
                s1 = parse_vslices(m.group(2), size, width)
                bidir = m.group(3) == '='
                s2 = parse_vslices(m.group(4), size, width)
                pairs = list(zip(s1, s2))
                if bidir:
                    pairs += [(b, a_) for a_, b in pairs]
                for u, v in pairs:
                    if op == '~':       # toggle
                        toggle(edges, u, v)
                    elif op == '!':     # remove
                        edges.get(u, set()).discard(v)
                        edges.get(v, set()).discard(u)
                    elif op in ('*', '+'):  # add if missing
                        edges.setdefault(u, set()).add(v)
                        edges.setdefault(v, set()).add(u)

    return (gtyp, edges, width, v_props, e_props, default_rwd, size, g_mode, initial)


def grf_size(graph):
    return graph[6]


def grf_nbrs(graph, v):
    return graph[1].get(v, set())


def grf_gprops(graph):
    if graph[0] == "N":
        return {"rwd": graph[5]}
    return {"width": graph[2], "rwd": graph[5]}


def grf_vprops(graph, n):
    return graph[3].get(n, {})


def grf_eprops(graph, a, b):
    return graph[4].get((a, b), {})


def grid_from_edges(graph):
    """builds the string representation of the layout, w/ jumps appended after if any."""
    _, current, width, _, _, _, size, _, native = graph
    if not width: return ""
    directors = {i: set() for i in range(size)}
    jumps = []   # edges that aren't part of the native grid -> "jumps"
    for n, nbrs in current.items():
        for nb in nbrs:
            if nb in native.get(n, set()):
                if   nb == n - width: directors[n].add("N")
                elif nb == n + width: directors[n].add("S")
                elif nb == n - 1:     directors[n].add("W")
                elif nb == n + 1:     directors[n].add("E")
            else:
                jumps.append((n, nb))
    chars = []
    for i in range(size):
        chars.append(LOOKUPTABLE[tuple(sorted(directors[i]))])
    final = "".join(chars)
    if jumps:
        jumps.sort()
        final += "\nJumps: " + ";".join(f"{u}~{v}" for u, v in jumps)
    return final


def grf_str_edges(graph):
    if not graph[2]:
        return ""
    return grid_from_edges(graph)


def grf_str_props(graph):
    typesr, edges, width, v_props, e_props, rwd, size, g_mode, _ = graph
    parts = []
    if typesr == "G":
        parts.append(f"rwd: {rwd}, width: {width}")
    else:
        parts.append(f"rwd: {rwd}")
    for v in sorted(v_props.keys()):
        if v_props[v]:
            parts.append(f"{v}: {v_props[v]}")
    for u, v in sorted(e_props.keys()):
        if e_props[(u, v)]:
            parts.append(f"({u}, {v}): {e_props[(u, v)]}")
    return "\n".join(parts)


def policy(graph):
    """
    computes the optimal policy for each cell using BFS from each reward source.
    g_mode 0 (== GW2): prefer highest reward, break ties by shortest distance
    g_mode 1 (== GW4): prefer best reward/distance ratio
    """
    gtype, edges, width, v_props, e_props, rwd, size, g_mode, _ = graph
    rwds = {k: v['rwd'] for k, v in v_props.items() if 'rwd' in v}

    # BFS from each reward cell to get distances (don't pass thru other rewards)
    dist_from = {}
    for rc in rwds:
        d = {i: float('inf') for i in range(size)}
        d[rc] = 0
        q = [rc]
        while q:
            cur = q.pop(0)
            for nb in edges.get(cur, set()):
                if nb in rwds and nb != rc:
                    continue   # don't path through other reward cells
                if d[nb] == float('inf'):
                    d[nb] = d[cur] + 1
                    q.append(nb)
        dist_from[rc] = d

    chars = ['.'] * size
    for i in range(size):
        if i in rwds:
            chars[i] = '*'
            continue
        best_score = None
        best_tgts = []
        for rc, r in rwds.items():
            dist = dist_from[rc][i]
            if dist == float('inf'): continue
            if g_mode == 0:                # max reward, then min distance
                score = (r, -dist)
            else:                          # reward / distance
                score = r / dist
            if best_score is None or score > best_score:
                best_score = score
                best_tgts = [(rc, dist)]
            elif score == best_score:
                best_tgts.append((rc, dist))
        if not best_tgts:
            continue
        # figure out which neighbor(s) actually advance us towards a best target
        bdirs = []
        for j in edges.get(i, set()):
            for rc, rd in best_tgts:
                if dist_from[rc][j] == rd - 1:
                    if width and width > 0:
                        if   j == i - width:                          bdirs.append('N')
                        elif j == i + width:                          bdirs.append('S')
                        elif j == i + 1 and (i + 1) % width != 0:     bdirs.append('E')
                        elif j == i - 1 and i % width != 0:           bdirs.append('W')
                    break
        key = "".join(sorted(set(bdirs)))
        chars[i] = ADJ_POLICY_LOOKUPTABLE.get(key, '.')
    return "".join(chars)


def main():
    graph = grf_parse(args)
    pol = policy(graph)
    width = graph[2]

    # split off jump section if any (none in current policy output but kept for fwd compat)
    parts = pol.split("\nJumps:")
    gpart = parts[0]

    print("Policy: ")
    if gpart and width and width > 0:
        for i in range(0, len(gpart), width):
            print(gpart[i:i + width])
    elif gpart:
        # network mode — just dump the chars
        print(gpart)
    if len(parts) > 1:
        print("Jumps:" + parts[1])

    if DEBUG:
        print(grf_str_edges(graph))
        print(grf_str_props(graph))


if __name__ == "__main__":
    main()
# combined graph/policy — pulls from two takes on the same grid-world lab
