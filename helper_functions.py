"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes


class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence


def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """
    # Work on a copy so the matrix that was passed in is never changed
    # (one of the unit tests checks this)
    D = np.array(distances, dtype=float)


    # Every label starts as its own leaf node with no children
    nodes = [Node(lab, None, 0.0, None, 0.0) for lab in labels]


    # Used to give each new internal node a unique name. reroot_tree finds
    # nodes by name, so the names must not repeat
    counter = 0


    # Keep joining pairs until only two nodes are left
    while len(nodes) > 2:
        n = len(nodes)


        # r[i] = sum of distances from node i to every other node
        r = D.sum(axis=1)


        # Q matrix: the pair with the smallest Q is joined next.
        # Q(i, j) = (n - 2) * d(i, j) - r[i] - r[j]
        Q = (n - 2) * D - r[:, None] - r[None, :]


        # A node can't be joined with itself, so make the diagonal huge
        np.fill_diagonal(Q, np.inf)


        # argmin gives the first smallest value, which also settles ties
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        if i > j:
            i, j = j, i


        # Branch lengths from the new node down to nodes i and j
        d_ij = D[i, j]
        d_i = 0.5 * d_ij + (r[i] - r[j]) / (2 * (n - 2))
        d_j = d_ij - d_i


        # The new internal node has i and j as its children
        new_node = Node(f"node_{counter}", nodes[i], d_i, nodes[j], d_j)
        counter += 1


        # Distance from the new node to every other node k:
        # d(new, k) = (d(i, k) + d(j, k) - d(i, j)) / 2
        d_new = 0.5 * (D[i] + D[j] - d_ij)


        # Build the smaller matrix: drop rows/columns i and j, and add the
        # new node at the END (the docstring asks for this)
        keep = [k for k in range(n) if k not in (i, j)]
        D_next = np.zeros((len(keep) + 1, len(keep) + 1))
        D_next[:-1, :-1] = D[np.ix_(keep, keep)]
        D_next[-1, :-1] = d_new[keep]
        D_next[:-1, -1] = d_new[keep]


        nodes = [nodes[k] for k in keep] + [new_node]
        D = D_next


    # Two nodes left. Join them under a root, putting the root halfway
    # along the branch between them
    d = D[0, 1]
    return Node("root", nodes[0], d / 2, nodes[1], d / 2)


def plot_nj_tree(tree: Node, ax: Axes = None, leaf_colors: dict = None,
                 fontsize: int = 10, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")

    """
    import matplotlib.pyplot as plt

    # If no axes were given, make a new one
    if ax is None:
        _, ax = plt.subplots()
    leaf_colors = leaf_colors or {}


    # These are one-item lists so the inner function can change them
    next_leaf_y = [0]   # the y position the next leaf will get
    max_x = [0.0]       # the furthest right x we have drawn to


    def draw(node: Node, x: float) -> float:
        """Draw the subtree starting at `node`, which sits at position x.
        Returns the y position of the node so its parent can use it."""
        max_x[0] = max(max_x[0], x)


        # Leaf: give it the next free y position and write its name
        if node.left is None and node.right is None:
            y = next_leaf_y[0]
            next_leaf_y[0] += 1
            ax.text(x, y, f"  {node.name}", va="center", fontsize=fontsize,
                    color=leaf_colors.get(node.name, "k"))
            return y


        # Internal node: a child sits at (this x + its branch length)
        x_left = x + node.left_distance
        x_right = x + node.right_distance


        # Draw the children first, because this node's y position
        # depends on where they ended up
        y_left = draw(node.left, x_left)
        y_right = draw(node.right, x_right)


        # Vertical line joining the two children
        ax.plot([x, x], [y_left, y_right], color="k", lw=1.5)
        # Horizontal branches out to each child
        ax.plot([x, x_left], [y_left, y_left], color="k", lw=1.5)
        ax.plot([x, x_right], [y_right, y_right], color="k", lw=1.5)


        # This node sits halfway between its two children
        return (y_left + y_right) / 2


    # The root is at x = 0
    root_y = draw(tree, 0.0)

    # Short line to the left of the root, like in the example figure
    stub = 0.03 * max_x[0] if max_x[0] > 0 else 0.2
    ax.plot([-stub, 0], [root_y, root_y], color="k", lw=1.5)


    # Leave room on the right for the leaf names, and tidy the axes
    ax.margins(x=0.3)
    ax.set_xlabel("Genetic distance")
    ax.set_yticks([])
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)

    return ax



def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root

def _count_leaves(node: Node) -> int:
    """Count the leaves in the subtree below `node` (helper for sorting)."""
    if node is None:
        return 0
    # A leaf has no children
    if node.left is None and node.right is None:
        return 1
    return _count_leaves(node.left) + _count_leaves(node.right)

def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Parameters
    ----------
    tree: Node
        The root node of the tree.

    """
    # Nothing to sort at a leaf
    if tree is None or (tree.left is None and tree.right is None):
        return


    # Sort the subtrees first, then this node
    sort_children_by_leaves(tree.left)
    sort_children_by_leaves(tree.right)


    # The child with fewer leaves should be on the left. When swapping the
    # children, swap their branch lengths too, because a distance belongs
    # to its branch
    if _count_leaves(tree.left) > _count_leaves(tree.right):
        tree.left, tree.right = tree.right, tree.left
        tree.left_distance, tree.right_distance = tree.right_distance, tree.left_distance



def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()

def global_alignment(seq1, seq2, scoring_function):
    """Global sequence alignment using the Needleman–Wunsch algorithm.
    Parameters
    ----------
    seq1: str
        First sequence to be aligned.
    seq2: str
        Second sequence to be aligned.
    scoring_function: Callable
        Takes two characters (either can be "-" for a gap) and returns
        their score.

    Returns
    -------
    (str, str, float)
        The two aligned sequences and the final alignment score.

    Examples
    --------
    >>> global_alignment("abracadabra", "dabarakadara", lambda x, y: [-1, 1][x == y])
    ('-ab-racadabra', 'dabarakada-ra', 5.0)
    """
    n_rows = len(seq1)
    n_cols = len(seq2)


    # dp_table[r, c] = best score for aligning the first r letters of seq1
    #                  with the first c letters of seq2
    # directions[r, c] = which move gave that best score, used later to
    #                    walk back and rebuild the alignment
    dp_table = np.zeros((n_rows + 1, n_cols + 1), dtype=float)
    directions = np.zeros((n_rows + 1, n_cols + 1), dtype=int)
    DIAG, UP, LEFT = 1, 2, 3


    # First column: seq1's letters aligned against nothing, so every one
    # of them is paired with a gap
    for r in range(1, n_rows + 1):
        dp_table[r, 0] = dp_table[r - 1, 0] + scoring_function(seq1[r - 1], "-")
        directions[r, 0] = UP
    # First row: same thing for seq2's letters
    for c in range(1, n_cols + 1):
        dp_table[0, c] = dp_table[0, c - 1] + scoring_function("-", seq2[c - 1])
        directions[0, c] = LEFT


    # Fill in the rest of the table. Row r is the letter seq1[r - 1] and
    # column c is seq2[c - 1] (the table has an extra row and column)
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            # Pair the two letters: come from the diagonal cell
            diag_val = dp_table[r - 1, c - 1] + scoring_function(seq1[r - 1], seq2[c - 1])
            # seq1's letter against a gap: come from the cell above
            up_val = dp_table[r - 1, c] + scoring_function(seq1[r - 1], "-")
            # seq2's letter against a gap: come from the cell on the left
            left_val = dp_table[r, c - 1] + scoring_function("-", seq2[c - 1])


            best_val = max(diag_val, up_val, left_val)
            dp_table[r, c] = best_val


            # Remember which move won. Ties go diagonal, then up, then left
            if best_val == diag_val:
                directions[r, c] = DIAG
            elif best_val == up_val:
                directions[r, c] = UP
            else:
                directions[r, c] = LEFT


    # Traceback: start in the bottom-right corner and follow the stored
    # directions back to (0, 0). This builds the alignment backwards
    aligned_a = []
    aligned_b = []
    r, c = n_rows, n_cols
    while r > 0 or c > 0:
        step = directions[r, c]
        if step == DIAG:
            # Both letters are paired with each other
            aligned_a.append(seq1[r - 1])
            aligned_b.append(seq2[c - 1])
            r -= 1
            c -= 1
        elif step == UP:
            # seq1's letter is paired with a gap in seq2
            aligned_a.append(seq1[r - 1])
            aligned_b.append("-")
            r -= 1
        else:
            # seq2's letter is paired with a gap in seq1
            aligned_a.append("-")
            aligned_b.append(seq2[c - 1])
            c -= 1


    # Reverse, because we built the strings from the end to the start
    final_seq1 = "".join(reversed(aligned_a))
    final_seq2 = "".join(reversed(aligned_b))
    final_score = float(dp_table[n_rows, n_cols])
    return final_seq1, final_seq2, final_score