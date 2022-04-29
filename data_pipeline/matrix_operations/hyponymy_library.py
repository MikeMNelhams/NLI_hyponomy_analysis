from enum import Enum

import random
import numpy as np
import scipy.special
import sklearn.metrics as metrics

from NLI_hyponomy_analysis.data_pipeline.matrix_operations import matrix_library as matrix


class InvalidPhraseType(Exception):
    """ When the user passes an invalid phrase type into an argument """
    def __init__(self, phrase_type: str):
        self.message = "Unexpected phrase_type: should be 'SV', 'VO', 'SVO'. You have {0}".format(phrase_type)
        super().__init__(self.message)


class InvalidPhraseListLength(Exception):
    """ Raised when the matrix list length would not correspond to any possible phrase type"""
    def __init__(self, length: int):
        self.message = "unexpected list length: should be 2 for SV and VO, 3 for SVO. You have {0}".format(length)


class Phrases(Enum):
    phrase_types = ('SV', 'VO', 'SVO')
    valid_numbers_of_matrices = (2, 3)


def assert_valid_phrase_type(phrase_type: str) -> None:
    assert phrase_type in Phrases.phrase_types, InvalidPhraseType(phrase_type)
    return None


def is_symb(a_word, b_word, hyp_dict) -> bool:
    if a_word in hyp_dict[b_word]:
        return True
    return False


def tad(list_of_matrices):
    dim = np.shape(list_of_matrices[0])[0]
    number_of_matrices = len(list_of_matrices)
    assert number_of_matrices in Phrases.valid_numbers_of_matrices, InvalidPhraseListLength(number_of_matrices)
    if len(list_of_matrices) == 2:
        mat = (np.trace(list_of_matrices[0])*list_of_matrices[1] +
               np.trace(list_of_matrices[1])*list_of_matrices[0]) / dim + matrix.diag(list_of_matrices)
        mat = mat / 3
        return mat
    if len(list_of_matrices) == 3:
        mat = (np.trace(list_of_matrices[1])*list_of_matrices[2] +
               np.trace(list_of_matrices[2])*list_of_matrices[1]) / dim + matrix.diag(list_of_matrices[1:3])
        mat = mat/3
        mat2 = (np.trace(mat)*list_of_matrices[0] + np.trace(list_of_matrices[0])*mat) / dim + \
               matrix.diag([list_of_matrices[0], mat])
        mat2 = mat2/3
        return mat2


def project(list_of_matrices, verb_operator=True, phrase_type='SV', tol=1e-8):
    number_of_matrices = len(list_of_matrices)

    assert number_of_matrices in Phrases.valid_numbers_of_matrices, InvalidPhraseListLength(number_of_matrices)
    assert_valid_phrase_type(phrase_type)

    mat = None
    if phrase_type == 'SV':
        noun, verb = list_of_matrices
        if verb_operator:
            mat = matrix.projection(verb, noun, tol=tol)
        else:
            mat = matrix.projection(noun, verb, tol=tol)
    elif phrase_type == 'VO':
        verb, noun = list_of_matrices
        if verb_operator:
            mat = matrix.projection(verb, noun, tol=tol)
        else:
            mat = matrix.projection(noun, verb, tol=tol)
    elif phrase_type == "SVO":
        noun1, verb, noun2 = list_of_matrices[0], list_of_matrices[1], list_of_matrices[2]
        if verb_operator:
            mat = matrix.projection(verb, noun2, tol=tol)
            mat = matrix.projection(mat, noun1, tol=tol)
        else:
            noun2_sqrt = matrix.matrix_sqrt(noun2)
            noun1_sqrt = matrix.matrix_sqrt(noun1)
            mat = noun2_sqrt.dot(verb).dot(noun2_sqrt)
            mat = noun1_sqrt.dot(mat).dot(noun1_sqrt)
    return mat


def k_e(a, b, tol=1e-8) -> float:
    matrix.assert_non_zero_matrix(a, b)

    eigenvalues = np.linalg.eigvalsh(b - a)

    matrix.assert_real_eigenvalues(eigenvalues, tolerance=tol)

    eigenvalues_real = eigenvalues.real
    e = np.diag([e if e < 0 else 0 for e in eigenvalues_real])
    metric = 1-np.linalg.norm(e)/np.linalg.norm(a)
    return metric


def k_ba(a, b, tol=1e-8) -> float:
    """ A metric for measuring hyponymy """
    matrix.assert_non_zero_matrix(a, b)

    eigenvalues = np.linalg.eigvalsh(b - a)

    matrix.assert_real_eigenvalues(eigenvalues, tolerance=tol)

    eigenvalues = eigenvalues.real

    if np.all(eigenvalues == 0):
        return 1
    
    return sum(eigenvalues)/sum(np.abs(eigenvalues))


def k_mult(a, b, tol=1e-8) -> np.array:
    """ helper function for Kraus  """
    matrix.assert_real_eigenvalues(a, b, tolerance=tol)

    a = np.real(a)
    b = np.real(b)
    assert matrix.is_symmetric(a), matrix.MatrixNotSymmetricError

    values, vectors = np.linalg.eigh(a)

    matrix.assert_real_eigenvalues(values, tolerance=tol)

    values = np.real(values)
    vectors = np.array([vec for val, vec in zip(values, vectors) if np.abs(val) > tol])
    vals = values[np.abs(values) > tol]

    assert len(vectors) == len(vals)
    assert np.all(np.abs(vectors.imag) < tol), matrix.ContainsComplexEigenvaluesError(vectors)

    vectors = np.real(vectors)
    ab = np.zeros(np.shape(a))

    for vector, val in zip(vectors, vals):
        p = np.outer(vector, vector)
        ab += val * p.dot(b.dot(p))
    return ab


def k_similar(x, y):
    entropy_x = scipy.special.xlogy(x, x)
    entropy_xy = scipy.special.xlogy(x, y)
    return 1 / (1 + np.trace(entropy_x - entropy_xy))


def kraus(list_of_matrices, verb_operator=True, phrase_type='SV'):
    number_of_matrices = len(list_of_matrices)

    assert number_of_matrices in Phrases.valid_numbers_of_matrices, InvalidPhraseListLength(number_of_matrices)
    assert_valid_phrase_type(phrase_type)

    mat = None
    if phrase_type == 'SV':
        noun = list_of_matrices[0]
        verb = list_of_matrices[1]
        if verb_operator:
            mat = k_mult(verb, noun)
        else:
            mat = k_mult(noun, verb)
    elif phrase_type == 'VO':
        noun = list_of_matrices[1]
        verb = list_of_matrices[0]
        if verb_operator:
            mat = k_mult(verb, noun)
        else:
            mat = k_mult(noun, verb)
    elif phrase_type == 'SVO':
        noun1 = list_of_matrices[0]
        verb = list_of_matrices[1]
        noun2 = list_of_matrices[2]
        if verb_operator:
            mat = k_mult(verb, noun2)
            mat = k_mult(mat, noun1)
        else:
            mat = k_mult(noun2, verb)
            mat = k_mult(noun1, mat)
    return mat


def traced_addition(list_of_matrices):
    dim = np.shape(list_of_matrices[0])[0]
    number_of_matrices = len(list_of_matrices)
    assert number_of_matrices in Phrases.valid_numbers_of_matrices, InvalidPhraseListLength
    if len(list_of_matrices) == 2:
        mat = (np.trace(list_of_matrices[0])*list_of_matrices[1] + np.trace(list_of_matrices[1])*list_of_matrices[0])/2
        mat = mat/dim
        return mat
    if len(list_of_matrices) == 3:
        mat = (np.trace(list_of_matrices[1])*list_of_matrices[2] + np.trace(list_of_matrices[2])*list_of_matrices[1])/2
        mat = mat/dim
        mat2 = (np.trace(mat)*list_of_matrices[0] + np.trace(list_of_matrices[0])*mat)/2
        mat2 = mat2/dim
        return mat2


def sum_addition(list_of_matrices):
    dim = np.shape(list_of_matrices[0])[0]
    number_of_matrices = len(list_of_matrices)
    assert number_of_matrices in Phrases.valid_numbers_of_matrices, InvalidPhraseListLength
    if len(list_of_matrices) == 2:
        mat = (np.sum(list_of_matrices[0])*list_of_matrices[1] + np.sum(list_of_matrices[1])*list_of_matrices[0])/2
        mat = mat/(dim**2)
        return mat
    if len(list_of_matrices) == 3:
        mat = (np.sum(list_of_matrices[1])*list_of_matrices[2] + np.sum(list_of_matrices[2])*list_of_matrices[1])/2
        mat = mat/(dim**2)
        mat2 = (np.sum(mat)*list_of_matrices[0] + np.sum(list_of_matrices[0])*mat)/2
        mat2 = mat2/(dim**2)
        return mat2
    raise ValueError


def sum_n_diag_v(list_of_matrices, phrase_type='SV', normalisation='maxeig1'):
    if normalisation == 'maxeig1':
        dim = np.shape(list_of_matrices[0])[0]
    else:
        dim = 1.
    assert_valid_phrase_type(phrase_type)
    if phrase_type == 'SV':
        mat = np.diag(np.diag(list_of_matrices[1]))*np.sum(list_of_matrices[0])/(dim**2)
    elif phrase_type == 'SVO':
        # We end up summing over the diagonal, which is the trace
        mat = np.diag(np.diag(list_of_matrices[1]))*np.sum(list_of_matrices[0])*np.trace(list_of_matrices[2])/(dim**3)
    else:
        mat = np.diag(np.diag(list_of_matrices[0]))*np.sum(list_of_matrices[1])/(dim**2)
    return mat


def sum_v_diag_n(list_of_matrices, phrase_type='SV', normalisation='maxeig1'):
    if normalisation == 'maxeig1':
        dim = np.shape(list_of_matrices[0])[0]
    else:
        dim = 1.
    assert_valid_phrase_type(phrase_type)
    if phrase_type == 'SV':
        mat = np.diag(np.diag(list_of_matrices[0]))*np.sum(list_of_matrices[1])/(dim**2)
    elif phrase_type == 'SVO':
        mat = np.diag(np.diag(list_of_matrices[0]))*np.sum(list_of_matrices[1])*np.trace([list_of_matrices[2]])/(dim**3)
    else:
        mat = np.diag(np.diag(list_of_matrices[1]))*np.sum(list_of_matrices[0])/(dim**2)
    return mat


def verb_only(list_of_matrices, phrase_type='SV'):
    assert_valid_phrase_type(phrase_type)
    if phrase_type == 'SV' or phrase_type == 'SVO':
        place = 1
    else:
        place = 0
    mat = list_of_matrices[place]
    return mat


def noun_only(list_of_matrices, phrase_type='SV'):
    assert_valid_phrase_type(phrase_type)
    if phrase_type == 'SV' or phrase_type == 'SVO':
        place = 0
    else:
        place = 1
    mat = list_of_matrices[place]
    return mat


def traced_verb_only(list_of_matrices, phrase_type='SV', normalisation='maxeig1'):
    if normalisation == 'maxeig1':
        dim = np.shape(list_of_matrices[0])[0]
    else:
        dim = 1.
    assert_valid_phrase_type(phrase_type)
    if phrase_type == 'SV':
        mat = list_of_matrices[1]*np.trace(list_of_matrices[0])/dim
    elif phrase_type == 'SVO':
        mat = list_of_matrices[1]*np.trace(list_of_matrices[0])*np.trace(list_of_matrices[2])/(dim*dim)
    else:
        mat = list_of_matrices[0]*np.trace(list_of_matrices[1])/dim
    return mat


def traced_noun_only(list_of_matrices, phrase_type='SV', normalisation='maxeig1'):
    if normalisation == 'maxeig1':
        dim = np.shape(list_of_matrices[0])[0]
    else:
        dim = 1.
    assert_valid_phrase_type(phrase_type)
    if phrase_type == 'SV':
        mat = list_of_matrices[0]*np.trace(list_of_matrices[1])/dim
    elif phrase_type == 'SVO':
        mat = list_of_matrices[0]*np.trace(list_of_matrices[1])*np.trace(list_of_matrices[2])/(dim*dim)
    else:
        mat = list_of_matrices[1]*np.trace(list_of_matrices[0])/dim
    return mat


def sum_verb_only(list_of_matrices, phrase_type='SV', normalisation='maxeig1'):
    if normalisation == 'maxeig1':
        dim = np.shape(list_of_matrices[0])[0]
    else:
        dim = 1.
    assert_valid_phrase_type(phrase_type)
    if phrase_type == 'SV':
        mat = list_of_matrices[1] * np.sum(list_of_matrices[0]) / (dim**2)
    elif phrase_type == 'SVO':
        mat = list_of_matrices[1] * np.sum(list_of_matrices[0]) * np.sum(list_of_matrices[2]) / (dim**4)
    else:
        mat = list_of_matrices[0] * np.sum(list_of_matrices[1])/dim
    return mat


def sum_noun_only(list_of_matrices, phrase_type='SV', normalisation='maxeig1'):
    if normalisation == 'maxeig1':
        dim = np.shape(list_of_matrices[0])[0]
    else:
        dim = 1.
    assert_valid_phrase_type(phrase_type)
    if phrase_type == 'SV':
        mat = list_of_matrices[0] * np.sum(list_of_matrices[1]) / (dim**2)
    elif phrase_type == 'SVO':
        mat = list_of_matrices[0] * np.sum(list_of_matrices[1]) * np.sum(list_of_matrices[2]) / (dim**4)
    else:
        mat = list_of_matrices[1] * np.sum(list_of_matrices[0]) / (dim**2)
    return mat


def entailments(phrase_pairs, density_matrices, func: callable, pt,
                norm_type='maxeig1', entailment_metric: callable = k_ba, verb_operator=True, tol=1e-8) -> dict:
    combined_phrases = {}
    for phrase_pair in phrase_pairs:
        wv_list = [density_matrices[p] for p in phrase_pair]
        middle = int(len(wv_list) / 2)

        w_list = wv_list[:middle]
        v_list = wv_list[middle:]

        w = matrix.compose(w_list, func, pt=pt, verb_operator=verb_operator, norm_type=norm_type)
        v = matrix.compose(v_list, func, pt=pt, verb_operator=verb_operator, norm_type=norm_type)
        combined_phrases[phrase_pair] = entailment_metric(w, v, tol)
    return combined_phrases


def negative_entailments(phrase_pairs, i, d, density_matrices, func: callable, pt,
                         norm_type='maxeig1', entailment_func=k_ba, verb_operator=True, tol=1e-8) -> dict:
    combined_phrases = {}
    for phrase_pair in phrase_pairs:
        if i == 'plain':
            wv_list = [density_matrices[p] for p in phrase_pair]
        elif i == 'neg_verb':
            wv_list = [density_matrices[phrase_pair[0]],
                       np.eye(d) - density_matrices[phrase_pair[1]],
                       density_matrices[phrase_pair[2]],
                       np.eye(d) - density_matrices[phrase_pair[3]]]
        elif i == 'neg_noun':
            wv_list = [np.eye(d) - density_matrices[phrase_pair[0]],
                       density_matrices[phrase_pair[1]],
                       np.eye(d) - density_matrices[phrase_pair[2]],
                       density_matrices[phrase_pair[3]]]
        else:
            wv_list = [np.eye(d) - density_matrices[phrase_pair[0]],
                       np.eye(d) - density_matrices[phrase_pair[1]],
                       np.eye(d) - density_matrices[phrase_pair[2]],
                       np.eye(d) - density_matrices[phrase_pair[3]]]

        middle = int(len(wv_list) / 2)

        w_list = wv_list[:middle]
        v_list = wv_list[middle:]

        w = matrix.compose(w_list, func, pt=pt, verb_operator=verb_operator, norm_type=norm_type)
        v = matrix.compose(v_list, func, pt=pt, verb_operator=verb_operator, norm_type=norm_type)

        combined_phrases[phrase_pair] = entailment_func(w, v, tol)
    return combined_phrases


def calculate_roc(phrase_pairs, combined_phrases, phrase_type, roc_dict, func_name):
    sorted_true = [phrase_pairs[key] for key in sorted(phrase_pairs.keys())]
    sorted_calculated = [combined_phrases[key] for key in sorted(phrase_pairs.keys())]
    sorted_true = [1 if val == 't' else 0 for val in sorted_true]
    roc_dict[(phrase_type, func_name)] = metrics.roc_auc_score(sorted_true, sorted_calculated)
    return roc_dict


def calculate_roc_bootstraps(phrase_pairs, combined_phrases, phrase_type, roc_dict, func_name,
                             number_of_iterations=100):
    for i in range(number_of_iterations):
        bs_keys = [random.choice(list(phrase_pairs.keys())) for _ in range(len(phrase_pairs))]
        sorted_true = [phrase_pairs[key] for key in sorted(bs_keys)]
        sorted_calculated = [combined_phrases[key] for key in sorted(bs_keys)]
        sorted_true = [1 if val == 't' else 0 for val in sorted_true]
        if (phrase_type, func_name) in roc_dict:
            roc_dict[(phrase_type, func_name)].append(metrics.roc_auc_score(sorted_true, sorted_calculated))
        else:
            roc_dict[(phrase_type, func_name)] = [metrics.roc_auc_score(sorted_true, sorted_calculated)]
    return roc_dict


def mult(a: np.array, b: np.array) -> np.array:
    try:
        result = matrix.hadamard_product(a, b)
    except Exception as e:
        print(a.shape)
        print(b.shape)
        raise e
    return result


def mmult1(a: np.array, b: np.array) -> np.array:
    result = matrix.projection(a, b)
    return result


def mmult2(a: np.array, b: np.array) -> np.array:
    return matrix.projection(b, a)
