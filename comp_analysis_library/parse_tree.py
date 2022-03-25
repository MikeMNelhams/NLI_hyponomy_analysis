from __future__ import annotations

from nltk import Tree
from functools import reduce
from NLI_hyponomy_analysis.comp_analysis_library.policies import Policy


class ChildAlreadyExistsError(Exception):
    def __init__(self, child, new_child):
        self.message = f"The node already has child {child}. Cannot insert {new_child}"
        super(ChildAlreadyExistsError, self).__init__(self.message)


class ParseTree:
    def __init__(self, binary_parse_string, word_vectors, policy: Policy):
        self.data = self.pos_string_to_binary_tree(binary_parse_string)
        self.word_vectors = word_vectors
        self.policy = policy

    def __repr__(self):
        return self.data.__repr__()

    def evaluate(self) -> None:
        self.data = self.__evaluate(self.data)
        return None

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

    @classmethod
    def from_sentence(cls, sentence: str, word_vectors, policy: Policy, delimiter=' ', tags=None):
        if '(' not in sentence:
            output = ParseTree('()', word_vectors, policy)
            leaves = sentence.split(delimiter)
            if tags is None:
                leaves = [Tree('', [leaf]) for leaf in leaves]
            else:
                assert len(tags) == len(leaves), TypeError
                leaves = [Tree(tag, [leaf]) for tag, leaf in zip(tags, leaves)]
            output.data = Tree("Root", leaves)
            return output
        return ParseTree(sentence, word_vectors, policy)

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

        for i, subtree in enumerate(tree_list):
            vector = subtree[0]
            if isinstance(vector, str):
                vector = self.word_vectors.safe_lookup(vector)
            tree_list[i] = Tree(subtree.label(), [vector])

        return self.policy.apply(tree.label(), *tree_list)

    @staticmethod
    def __tree_is_leaf(tree: Tree):
        if len(tree) == 1:
            if not isinstance(tree[0], Tree):
                return True
        return False

    @staticmethod
    def pos_string_to_binary_tree(pos_string: str):
        parsed_tree = Tree.fromstring(pos_string, remove_empty_top_bracketing=False)
        return parsed_tree


def main():
    pass
    # p = ParseTree.pos_string_to_binary_tree("(Root p{she,lifted}•heavy•weights)")
    # p.draw()
    # p = ParseTree.pos_string_to_binary_tree("(Root (S (PP I) (VBD ran) (RB very • quickly)))")
    # p.draw()
    # p = ParseTree.pos_string_to_binary_tree("(Root (S I • ran) (RB very • quickly))")
    # p.draw()
    # p = ParseTree.pos_string_to_binary_tree("(Root I • ran • RB very • quickly)")
    # p.draw()


if __name__ == "__main__":
    main()
