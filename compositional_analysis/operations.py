from nltk import Tree
import NLI_hyponomy_analysis.data_pipeline.matrix_operations.hyponymy_library as hl

import conditions as cond
from typing import Callable, Iterable, Tuple


def pairwise_product_with_ignored_labels(tree1: Tree, tree2: Tree,
                                         bivariate_operator: Callable[[Tree, Tree], Tree]=hl.mult,
                                         ignore_labels: Iterable[str]=('ls', 'pos', '.', 'dt', ',')) -> Tree:
    if cond.is_ignored(tree1, ignore_labels):
        if cond.is_ignored(tree2, ignore_labels):
            return Tree(None, [None])
        return tree2
    if cond.is_ignored(tree2, ignore_labels):
        return tree1

    return bivariate_operator(tree1, tree2)


def l2r_pairwise(*trees: Tuple[Tree],
                 bivariate_operator: Callable[[Tree, Tree], Tree]=pairwise_product_with_ignored_labels) -> Tree:
    product = trees[0]
    if len(trees) > 1:
        for tree in trees[1:]:
            if tree[0] is None:
                if product is None:
                    product = Tree(None, [None])
            if product is None:
                product = tree
            product = bivariate_operator(tree, product)
    return product


def r2l_pairwise(*trees: Tuple[Tree],
                 bivariate_operator: Callable[[Tree, Tree], Tree]=pairwise_product_with_ignored_labels) -> Tree:
    product = trees[-1]

    if len(trees) > 1:
        for tree in reversed(trees[:-1]):
            if tree[0] is None:
                if product is None:
                    product = Tree(None, [None])
            if product is None:
                product = tree
            product = bivariate_operator(tree, product)

    return product


def right_only(tree1: Tree, tree2: Tree) -> Tree:
    return tree2


def left_only(tree1: Tree, tree2: Tree) -> Tree:
    return tree1


def mult(tree1: Tree, tree2: Tree) -> Tree:
    return Tree(None, hl.mult(tree1[0], tree2))
