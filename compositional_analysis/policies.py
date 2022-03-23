from typing import Tuple, Iterable

from rules import Rule, DefaultRule

import conditions as cond
import operations as op

from nltk import Tree


class InvalidDefaultRule:
    pass


class Policy:
    """ Policy class to be used for ParseTree

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

    def __call__(self, *trees: Tuple[Tree], parent_label: str) -> Tree:
        number_of_trees = len(trees)
        if self.has_non_default_rules:
            for rule in self.rules:
                if len(rule) != number_of_trees:
                    continue
                if rule(trees):
                    return Tree(parent_label, [rule.function(trees)])

        if self.default_rule(trees):
            return Tree(parent_label, [self.default_rule.function(trees)])

        return Tree(parent_label, [None])

    @property
    def has_non_default_rules(self):
        return bool(self.rules)


def example_policy() -> Policy:
    default_rule = DefaultRule.default_l2r()
    rule1 = Rule.right_only_if()
    rule2 = Rule(condition=cond.is_verb_noun, function=op.mult)
    return Policy([rule1, rule2], default_rule)


def only_mult() -> Policy:
    default_rule = DefaultRule.default_l2r()
    return Policy([], default_rule)
