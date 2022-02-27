from __future__ import annotations

import copy
import NLI_hyponomy_analysis.data_pipeline.matrix_operations.hyponymy_library as hl

from nltk import Tree
from functools import reduce
from typing import Callable, List
import numpy as np


class ChildAlreadyExistsError(Exception):
    def __init__(self, child, new_child):
        self.message = f"The node already has child {child}. Cannot insert {new_child}"
        super(ChildAlreadyExistsError, self).__init__(self.message)


class BinaryParseTree:
    ignore_labels = ['ls', 'pos', '.']
    hadamard_labels = ['dt', 'ex', 'cd', 'md', 'pdt', 'prp', 'prp$', 'rp', 'uh', 'to']

    def __init__(self, binary_parse_string, word_vectors):
        self.data = self.pos_string_to_binary_tree(binary_parse_string)

        self.word_vectors = word_vectors

    def __repr__(self):
        return self.data.__repr__()

    def __binary_operation(self, label1: str, label2: str) -> Callable[[np.array, np.array], np.array]:
        if label1 in self.ignore_labels:
            if label2 in self.ignore_labels:
                return None
            return lambda x, y: y
        if label2 in self.ignore_labels:
            return lambda x, y: x

        if self.__is_noun_label(label1) and self.__is_verb_label(label2):
            return hl.mmult1
        if self.__is_verb_label(label1) and self.__is_noun_label(label2):
            return hl.mmult2

        return hl.mult

    @staticmethod
    def __is_verb_label(label: str):
        if label[0] == 'v':
            return True
        return False

    @staticmethod
    def __is_noun_label(label: str):
        if label[0] == 'n':
            return True
        if label == 'wp':
            return True
        return False

    def __evaluate_2(self, tree1: Tree, tree2: Tree, parent_label: str):
        label1 = tree1.label()
        label2 = tree2.label()

        vector1 = tree1[0]
        if isinstance(vector1, str):
            vector1 = self.word_vectors.safe_lookup(tree1[0])

        vector2 = tree2[0]
        if isinstance(vector2, str):
            vector2 = self.word_vectors.safe_lookup(tree2[0])

        operation = self.__binary_operation(label1, label2)

        if vector1 is None:
            if vector2 is None:
                return Tree(parent_label, [None])
            return Tree(parent_label, [vector2])
        if vector2 is None:
            return Tree(parent_label, [vector1])

        return Tree(parent_label, [operation(vector1, vector2)])

    def __evaluate_greater_than_2(self, tree_list: List[Tree], parent_label: str):
        assert len(tree_list) > 2, ValueError

        current_tree = tree_list[0]

        for tree in tree_list[1:]:
            current_tree = self.__evaluate_2(current_tree, tree, current_tree.label())

        return Tree(parent_label, [current_tree[0]])

    @staticmethod
    def __tree_is_leaf(tree: Tree):
        if len(tree) == 1:
            if not isinstance(tree[0], Tree):
                return True
        return False

    def evaluate(self) -> None:
        self.data = self.__evaluate(self.data)
        return None

    def __evaluate(self, tree: Tree):
        """
        We maintain a stack of parents.
        If all the children in the current node are leaves, we operate on tree1, tree2, tree3, ...
        """

        tree_list = []
        for child in tree:
            if self.__tree_is_leaf(child):
                tree_list.append(child)
            else:
                tree_list.append(self.__evaluate(child))

        if len(tree_list) == 1:
            vector1 = tree_list[0]
            if isinstance(vector1, str):
                vector1 = self.word_vectors.safe_lookup(vector1)
            return Tree(tree.label(), vector1)

        if len(tree_list) == 2:
            return self.__evaluate_2(tree_list[0], tree_list[1], tree.label())

        return self.__evaluate_greater_than_2(tree_list, tree.label())

    def tree_to_binary(self, tree):
        """
            Recursively turn a tree into a binary tree.
            """
        if isinstance(tree, str):
            return tree
        elif len(tree) == 1:
            return self.tree_to_binary(tree[0])
        else:
            label = tree.label()
            return reduce(lambda x, y: Tree(label, (self.tree_to_binary(x), self.tree_to_binary(y))), tree)

    @staticmethod
    def pos_string_to_binary_tree(pos_string: str):
        parsed_tree = Tree.fromstring(pos_string, remove_empty_top_bracketing=False)
        return parsed_tree


class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

    def __repr__(self):
        output_str = "-"*10 + f"Data: {self.data}, \n Left: {self.left}\n Right: {self.right}"
        return output_str

    def __copy__(self):
        new_node = Node(self.data)
        new_node.left = self.left
        new_node.right = self.right
        return new_node

    def __deepcopy__(self, memodict={}):
        new_node = Node(copy.deepcopy(self.data))
        new_node.left = copy.deepcopy(self.left)
        new_node.right = copy.deepcopy(self.right)
        return new_node

    def insert_left(self, node: Node) -> None:
        if self.left is not None:
            raise ChildAlreadyExistsError(self.left, node)
        self.left = node
        return None

    def insert_right(self, node: Node) -> None:
        if self.right is not None:
            raise ChildAlreadyExistsError(self.right, node)
        self.right = node
        return None

    def truncate(self) -> None:
        self.left = None
        self.right = None
        return None


def main():
    node1 = Node(0)
    node1.insert_left(Node(1))
    node1.insert_right(Node(2))
    print(node1)


if __name__ == "__main__":
    pass
