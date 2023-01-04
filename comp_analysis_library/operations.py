from nltk import Tree

import NLI_hyponomy_analysis.data_pipeline.matrix_operations.hyponymy_library as hl

import NLI_hyponomy_analysis.comp_analysis_library.conditions as cond
from typing import Callable, Iterable, Tuple


def add(tree1: Tree, tree2: Tree) -> Tree:
    return Tree(None, [tree1[0] + tree2[0]])


def mult(tree1: Tree, tree2: Tree) -> Tree:
    return Tree(None, [hl.mult(tree1[0], tree2[0])])


def mmult1(tree1: Tree, tree2: Tree) -> Tree:
    return Tree(None, [hl.mmult1(tree1[0], tree2[0])])


def mmult2(tree1: Tree, tree2: Tree) -> Tree:
    return Tree(None, [hl.mmult2(tree1[0], tree2[0])])


def mmult_o(tree1: Tree, tree2: Tree, tree3: Tree) -> Tree:
    return mmult1(tree1, mmult1(tree3, tree2))


def mmult_s(tree1: Tree, tree2: Tree, tree3: Tree) -> Tree:
    return mmult1(tree3, mmult1(tree1, tree2))


def pairwise_product_with_ignored_labels(tree1: Tree, tree2: Tree,
                                         bivariate_operator: Callable[[Tree, Tree], Tree]=mult,
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
                if product[0] is None:
                    product = Tree(None, [None])
                    continue
                continue
            if product[0] is None:
                product = tree
                continue

            product = bivariate_operator(tree, product)
    return product


def r2l_pairwise(*trees: Tuple[Tree],
                 bivariate_operator: Callable[[Tree, Tree], Tree]=pairwise_product_with_ignored_labels) -> Tree:

    if len(trees) == 1:
        return trees[0]

    product = trees[-1]

    for tree in reversed(trees[:-1]):
        if tree[0] is None:
            if product[0] is None:
                product = Tree(None, [None])
                continue
            continue
        if product[0] is None:
            product = tree
            continue

        product = bivariate_operator(tree, product)
    return product


def right_only(tree1: Tree, tree2: Tree) -> Tree:
    return tree2


def left_only(tree1: Tree, tree2: Tree) -> Tree:
    return tree1
