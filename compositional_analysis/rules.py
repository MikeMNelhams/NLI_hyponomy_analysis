from __future__ import annotations
import NLI_hyponomy_analysis.data_pipeline.matrix_operations.hyponymy_library as hl

from typing import Callable, Tuple
from nltk import Tree
from inspect import signature
import numpy as np

import conditions as cond
import operations as op


class RuleConditionSigError(Exception):
    def __init__(self, condition, function):
        self.message = f"The condition {condition} does take the same number of args as {function}"
        super(RuleConditionSigError, self).__init__(self.message)


class Rule:
    """ If True when rule() is called, returns function"""

    def __init__(self, condition: Callable[[Tuple[Tree]], bool], function: Callable[[Tuple[Tree]], Tree]):
        self.condition = condition
        self.function = function
        self._cond_param_num = len(signature(condition).parameters)
        self.__func_param_num = len(signature(function).parameters)
        assert self._cond_param_num == self.__func_param_num, RuleConditionSigError

    def __call__(self, *trees: Tuple[Tree]) -> bool:
        return self.condition(trees)

    def __len__(self):
        return self._cond_param_num

    @staticmethod
    def right_only_if(condition: Callable[[Tree, Tree], bool]=cond.left_tree_is_ignored) -> Rule:
        return Rule(condition, op.right_only)

    @staticmethod
    def left_only_if(condition: Callable[[Tree, Tree], bool]=cond.right_tree_is_ignored) -> Rule:
        return Rule(condition, op.left_only)


class DefaultRule(Rule):
    def __init__(self, function: callable):
        super(DefaultRule, self).__init__(cond.does_not_contain_none_tree, function=function)
        self._cond_param_num = -1  # Default rules should have infinite args amount

    @staticmethod
    def default_l2r(bivariate_operator: Callable[[np.array, np.array], np.array]=hl.mult) -> DefaultRule:
        def decorated_l2r_pairwise(*args):
            return op.l2r_pairwise(*args, operator=bivariate_operator)
        return DefaultRule(decorated_l2r_pairwise)

    @staticmethod
    def default_r2l(bivariate_operator: Callable[[np.array, np.array], np.array]=hl.mult) -> DefaultRule:
        def decorated_r2l_pairwise(*args):
            return op.r2l_pairwise(*args, operator=bivariate_operator)
        return DefaultRule(decorated_r2l_pairwise())



