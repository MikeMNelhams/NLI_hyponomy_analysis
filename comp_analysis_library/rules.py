from __future__ import annotations

from typing import Callable, Tuple
from nltk import Tree
from inspect import signature, Parameter
import numpy as np

import NLI_hyponomy_analysis.comp_analysis_library.conditions as cond
import NLI_hyponomy_analysis.comp_analysis_library.operations as op


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


class Normalisation:
    __names = (False, None, "none", "__max_eig", "maxeig", "trace", "tr")

    def __init__(self, norm_mode: str = "none"):
        correct_norm_mode = norm_mode.lower()
        self.__assert_valid_norm_mode(correct_norm_mode)
        self.norm_mode = correct_norm_mode
        self.__normalise_func = self.__mapping(correct_norm_mode)
        self.product = 1
        self.__previous_product = 1

    def __call__(self, x: Tree) -> Tree:
        return Tree(x.label(), [self.__normalise_func(x[0])])

    def __mapping(self, norm_mode: str) -> callable:
        maps = {False: self.__none, None: self.__none, "none": self.__none, "__max_eig": self.__max_eig,
                "maxeig": self.__max_eig, "trace": self.__trace, "tr": self.__trace}
        return maps[norm_mode]

    @staticmethod
    def __none(x):
        return x

    def __max_eig(self, x):
        maxeig = np.max(np.linalg.eigvalsh(x))
        self.product *= maxeig
        self.__previous_product = maxeig
        x = x / maxeig
        return x

    def __trace(self, x):
        divisor = np.trace(x)
        self.product *= divisor
        self.__previous_product = divisor
        x = x / divisor
        return x

    def __assert_valid_norm_mode(self, norm_mode: str) -> None:
        if norm_mode not in self.__names:
            raise TypeError
        return None

    @property
    def previous_product_scaling(self):
        return self.__previous_product


class Rule:
    """ If True when rule() is called, returns bool"""

    def __init__(self, condition: Callable[[Tuple[Tree]], bool], function: Callable[[Tuple[Tree]], Tree],
                 normalisation: Normalisation=Normalisation()):
        self.condition = condition
        self.normalisation = normalisation
        self.__function = function
        self._cond_param_num = number_of_pos_args(condition)
        self.__func_param_num = number_of_pos_args(function)
        assert self._cond_param_num == self.__func_param_num, RuleConditionSigError(self._cond_param_num, self.__func_param_num)

    def __call__(self, *trees: Tuple[Tree]) -> bool:
        return self.condition(*trees)

    def __len__(self):
        return self._cond_param_num

    def function(self, *trees: Tuple[Tree]) -> Tree:
        return self.normalisation(self.__function(*trees))

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
    def default_l2r(bivariate_operator: Callable[[np.array, np.array], np.array]=op.mult) -> DefaultRule:
        def decorated_l2r_pairwise(*args):
            return op.l2r_pairwise(*args, bivariate_operator=bivariate_operator)
        return DefaultRule(decorated_l2r_pairwise)

    @staticmethod
    def default_r2l(bivariate_operator: Callable[[np.array, np.array], np.array]=op.mult) -> DefaultRule:
        def decorated_r2l_pairwise(*args):
            return op.r2l_pairwise(*args, bivariate_operator=bivariate_operator)
        return DefaultRule(decorated_r2l_pairwise)
