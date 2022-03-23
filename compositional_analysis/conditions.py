from nltk import Tree
from typing import Tuple


default_ignored_labels = ('ls', 'pos', '.', 'dt', ',')


def universal_true(*args) -> bool:
    return True


def does_not_contain_none_tree(*trees: Tuple[Tree]) -> bool:
    for tree in trees:
        if tree[0] is not None:
            return True
    return False


def is_ignored(tree: Tree, ignore_labels=default_ignored_labels) -> bool:
    return tree[0] in ignore_labels


def left_tree_is_ignored(tree1: Tree, tree2: Tree, ignore_labels=default_ignored_labels) -> bool:
    if is_ignored(tree1, ignore_labels) and tree2[0] is not None:
        return True
    return False


def right_tree_is_ignored(tree1: Tree, tree2: Tree, ignore_labels=default_ignored_labels) -> bool:
    return is_ignored(tree2, ignore_labels) and tree1[0] is not None


def is_verb_noun(tree1: Tree, tree2: Tree) -> bool:
    return is_verb(tree1) and is_noun(tree2)


def is_verb(tree: Tree) -> bool:
    label = tree.label()
    if label == '':
        return False
    if label[0].lower() == 'v':
        return True
    return False


def is_noun(tree: Tree) -> bool:
    label = tree.label()
    if label == '':
        return False
    label = label.lower()
    if label[0] == 'n':
        return True
    if label == 'wp':
        return True
    return False
