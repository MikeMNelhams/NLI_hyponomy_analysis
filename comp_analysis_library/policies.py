import copy

from NLI_hyponomy_analysis.comp_analysis_library.rules import Rule, DefaultRule, Normalisation

import NLI_hyponomy_analysis.comp_analysis_library.conditions as cond
import NLI_hyponomy_analysis.comp_analysis_library.operations as op

from nltk import Tree


class InvalidDefaultRule:
    pass


class Policy:
    """ Policy class to be used for ParseTree
        0. If only one tree, return that.
        1. Goes through the first match of self.rules
        2. If no matches, then checks if all Tree leaves are None, if not, then executes default_rule
        3. Return None if the above failed
    """
    def __init__(self, list_of_rules=None, default_rule: DefaultRule=DefaultRule.default_l2r):
        """
        :param list_of_rules: List[Rule] executed left to right in order, first match is executed
        :param default_rule: if there are no matches, tries the DefaultRule
        """
        if list_of_rules is None:
            list_of_rules = []
        assert isinstance(default_rule, DefaultRule), InvalidDefaultRule
        self.rules = list_of_rules
        self.default_rule = default_rule
        self.product_scaling = 1

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, copy.deepcopy(value, memodict))
        return result

    def __call__(self, parent_label: str, *trees) -> Tree:
        number_of_trees = len(trees)
        if isinstance(trees, Tree):
            resulting_tree = trees
            resulting_tree.set_label(parent_label)
            return resulting_tree

        if self.has_non_default_rules:
            for rule in self.rules:
                if len(rule) != number_of_trees:
                    continue
                if rule(*trees):
                    resulting_tree = rule.function(*trees)
                    resulting_tree.set_label = parent_label
                    self.product_scaling *= rule.normalisation.previous_product_scaling
                    return resulting_tree

        if self.default_rule(*trees):
            resulting_tree = self.default_rule.function(*trees)
            resulting_tree.set_label(parent_label)
            self.product_scaling *= self.default_rule.normalisation.previous_product_scaling
            return resulting_tree
        return Tree(parent_label, [None])

    def apply(self, parent_label: str, *trees) -> Tree:
        return self(parent_label, *trees)

    @property
    def has_non_default_rules(self):
        return bool(self.rules)


def example_policy() -> Policy:
    default_rule = DefaultRule.default_l2r()
    rule1 = Rule.right_only_if()
    rule2 = Rule(condition=cond.is_verb_noun, function=op.mmult2)
    return Policy([rule1, rule2], default_rule)


def only_mult() -> Policy:
    default_rule = DefaultRule.default_l2r()
    return Policy([], default_rule)


def only_mult_trace() -> Policy:
    default_rule = DefaultRule.default_r2l()
    default_rule.normalisation = Normalisation("trace")
    return Policy([], default_rule)


def only_mult_maxeig() -> Policy:
    default_rule = DefaultRule.default_r2l()
    default_rule.normalisation = Normalisation("maxeig")
    return Policy([], default_rule)


def only_projection() -> Policy:
    default_rule = DefaultRule.default_r2l(bivariate_operator=op.mmult1)
    return Policy([], default_rule)


def only_addition() -> Policy:
    default_rule = DefaultRule.default_r2l(bivariate_operator=op.add)
    return Policy([], default_rule)


def verbs_mmult2() -> Policy:
    default_rule = DefaultRule.default_r2l(bivariate_operator=op.mmult2)
    rule1 = Rule.right_only_if()
    rule2 = Rule(condition=cond.is_verb_noun, function=op.mmult1)
    rule3 = Rule(condition=cond.is_adj_noun, function=op.mmult1)
    rule4 = Rule(condition=cond.is_noun_verb, function=op.mmult2)
    return Policy([rule1, rule2, rule3, rule4], default_rule)


def verbs_switch() -> Policy:
    default_rule = DefaultRule.default_r2l(bivariate_operator=op.mmult2)
    rule1 = Rule.right_only_if()
    rule2 = Rule(condition=cond.is_verb_noun, function=op.mmult1)
    rule3 = Rule(condition=cond.is_adj_noun, function=op.mmult1)
    rule4 = Rule(condition=cond.is_noun_verb, function=op.mmult1)
    return Policy([rule1, rule2, rule3, rule4], default_rule)


