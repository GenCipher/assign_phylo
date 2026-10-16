import numpy as np
from helper_functions import neighbor_joining, Node


def get_paths(node, depth=0.0, path=(), out=None):
    """For every leaf, record the chain of (node name, depth from root) down to it."""
    if out is None:
        out = {}
    path = path + ((node.name, depth),)
    if node.left is None and node.right is None:
        out[node.name] = path
        return out
    get_paths(node.left, depth + node.left_distance, path, out)
    get_paths(node.right, depth + node.right_distance, path, out)
    return out


def leaf_distance(paths, a, b):
    """Distance between leaves a and b = depth(a) + depth(b) - 2 * depth(LCA)."""
    pa, pb = paths[a], paths[b]
    k = 0
    while k < min(len(pa), len(pb)) and pa[k][0] == pb[k][0]:
        k += 1
    lca_depth = pa[k - 1][1]  # last node the two paths share
    return pa[-1][1] + pb[-1][1] - 2 * lca_depth


def check(D, labels):
    tree = neighbor_joining(D, labels)
    paths = get_paths(tree)
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            got = leaf_distance(paths, labels[i], labels[j])
            assert np.isclose(got, D[i, j]), (labels[i], labels[j], got, D[i, j])
    print("OK: all pairwise distances reproduced")


# Test 1: the textbook matrix from the unit tests
D1 = np.array([[0, 5, 4, 9, 8],
               [5, 0, 5, 10, 9],
               [4, 5, 0, 7, 6],
               [9, 10, 7, 0, 7],
               [8, 9, 6, 7, 0]], float)
check(D1, list("ABCDE"))


# Test 2: build a tree by hand, derive its distance matrix, see if NJ recovers it
def leaf(name):
    return Node(name, None, 0.0, None, 0.0)

hand_tree = Node(
    "ROOT",
    Node("X", leaf("A"), 6, leaf("B"), 8), 2,
    Node("Y", leaf("C"), 4, leaf("D"), 2), 2,
)
labels = list("ABCD")
paths = get_paths(hand_tree)
D2 = np.array([[leaf_distance(paths, a, b) for b in labels] for a in labels])
check(D2, labels)