from __future__ import annotations

from typing import Callable, Tuple
from nltk import Tree
from inspect import signature, Parameter
import numpy as np

import NLI_hyponomy_analysis.comp_analysis_library.conditions as cond
import NLI_hyponomy_analysis.comp_analysis_library.operations as op

BinaryOperation = Callable[[np.array, np.array], np.array]


def number_of_pos_args(func: callable):
    params = signature(func).parameters

    total = 0
    for param in params.values():
        if param.default is Parameter.empty:
            total += 1

    return total


class RuleConditionSigError(Exception):
    def __init__(self, condition, function):
        self.message = f"The condition takes {condition} args. This is not the same number of args as the function: {function}"
        super(RuleConditionSigError, self).__init__(self.message)


class Rule:
    """ If True when rule() is called, returns bool"""

    def __init__(self, condition: Callable[[Tuple[Tree]], bool], function: Callable[[Tuple[Tree]], Tree]):
        self.condition = condition
        self.__function = function
        self._cond_param_num = number_of_pos_args(condition)
        self.__func_param_num = number_of_pos_args(function)
        self.__norm_product = 1
        assert self._cond_param_num == self.__func_param_num, RuleConditionSigError(self._cond_param_num, self.__func_param_num)

    def __call__(self, *trees: Tuple[Tree]) -> bool:
        return self.condition(*trees)

    def __len__(self):
        return self._cond_param_num

    def function(self, *trees: Tuple[Tree], scaling=True) -> (float, Tree):
        result = self.__function(*trees)
        divisor = 1
        if scaling:
            numerator = np.max(result[0])
            denominator = np.max([tree[0] for tree in trees if tree[0] is not None])
            divisor: float = numerator / denominator
            self.__norm_product *= divisor
            result[0] /= divisor
        return divisor, result

    @staticmethod
    def right_only_if(condition: Callable[[Tree, Tree], bool]=cond.left_tree_is_ignored) -> Rule:
        return Rule(condition, op.right_only)

    @staticmethod
    def left_only_if(condition: Callable[[Tree, Tree], bool]=cond.right_tree_is_ignored) -> Rule:
        return Rule(condition, op.left_only)


class DefaultRule(Rule):
    """ Default Rules are when no conditions pass and will perform a default operation."""
    def __init__(self, function: callable):
        super(DefaultRule, self).__init__(cond.does_not_contain_none_tree, function=function)
        self._cond_param_num = -1  # Default rules should have infinite args amount

    @staticmethod
    def default_l2r(bivariate_operator: BinaryOperation=op.mult) -> DefaultRule:
        def decorated_l2r_pairwise(*args):
            return op.l2r_pairwise(*args, bivariate_operator=bivariate_operator)
        return DefaultRule(decorated_l2r_pairwise)

    @staticmethod
    def default_r2l(bivariate_operator: BinaryOperation=op.mult) -> DefaultRule:
        def decorated_r2l_pairwise(*args):
            return op.r2l_pairwise(*args, bivariate_operator=bivariate_operator)
        return DefaultRule(decorated_r2l_pairwise)
