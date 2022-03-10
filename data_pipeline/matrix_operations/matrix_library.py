import math
import numpy as np
from scipy.linalg import sqrtm


class InvalidNormalizationTypeError(Exception):
    """ When the given normalization method is invalid """
    def __init__(self, normalization: str):
        self.message = 'Possible arguments to normalize are "trace1", "maxeig1" or None (default). ' \
                  'You entered "{0}".'.format(normalization)
        super().__init__(self.message)


class ContainsComplexEigenvaluesError(Exception):
    """ When a matrix contains non-real eigenvalues """
    def __init__(self, complex_eigenvalues: np.array = None):
        error_causing_eigenvalues = ''
        if complex_eigenvalues is not None:
            error_causing_eigenvalues = ": {0}".format(complex_eigenvalues)
        self.message = "Matrix contains non-real eigenvalues" + error_causing_eigenvalues
        super().__init__(self.message)


class MatrixAllZeroError(Exception):
    """ When a matrix only contains 0 values """
    def __init__(self):
        self.message = "Matrix is empty and only contains 0 values"
        super().__init__(self.message)


class MatrixNotSymmetricError(Exception):
    """ Raised when a matrix is not symmetric """
    def __init__(self):
        self.message = "Matrix is not symmetric"
        super().__init__(self.message)


def assert_non_zero_matrix(*matrices) -> None:
    for matrix in matrices:
        if np.all(matrix == 0):
            raise MatrixAllZeroError
    return None


def assert_real_eigenvalues(*matrices, tolerance=1e-8) -> None:
    for matrix in matrices:
        assert np.all(np.abs(matrix.imag) < tolerance), ContainsComplexEigenvaluesError
    return None


def is_symmetric(a, rtol=1e-05, atol=1e-08) -> bool:
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def compose(list_of_matrices, func: callable, norm_type=None, **kwargs) -> np.array:
    matrix = func(list_of_matrices, **kwargs)
    matrix = normalize(matrix, norm_type)
    return matrix


def projection(a: np.array, b: np.array, tol=1e-8) -> np.array:
    """ a is outer, b is middle"""
    # verb_sqrt = matrix_sqrt(a)
    verb_sqrt = sqrtm(a)
    # assert_real_eigenvalues(verb_sqrt, tolerance=tol)
    mat = verb_sqrt.dot(b).dot(verb_sqrt)
    return mat


def normalize(a, normalization: str) -> np.array:
    """ matrix normalization """
    normalizations = ['trace1', 'maxeig1', None]

    assert normalization in normalizations, InvalidNormalizationTypeError(normalization)

    if normalization == 'trace1':
        assert np.trace(a) != 0, TypeError('Trace of a is 0, cannot normalize')
        a = a/np.trace(a)
    elif normalization == 'maxeig1':
        maxeig = np.max(np.linalg.eigvalsh(a))
        if maxeig >= 1:
            a = a/maxeig
    return a


def matrix_sqrt(a, tol=1e-4):
    assert_real_eigenvalues(a, tolerance=tol)

    a = np.real(a)

    assert is_symmetric(a), MatrixNotSymmetricError

    values, vectors = np.linalg.eigh(a)
    assert_real_eigenvalues(values, tolerance=tol)

    values = np.real(values)
    vectors = np.array([vec for val, vec in zip(values, vectors) if np.abs(val) > tol])
    values = values[np.abs(values) > tol]

    assert np.all(values >= 0), values
    values_sqrt = [math.sqrt(v) for v in values]

    assert len(vectors) == len(values_sqrt), "different number of eigenvectors and values"
    assert np.all(np.abs(vectors.imag) < tol), "Some eigenvectors complex: {0}".format(vectors)

    vectors = np.real(vectors)
    a_sqrt = vectors.T.dot(np.diag(values_sqrt)).dot(vectors)
    return a_sqrt


def multiply(list_of_matrices):
    mat = list_of_matrices[0]
    for m in list_of_matrices[1:]:
        mat = np.multiply(mat, m)
    return mat


def mean(list_of_matrices):
    mat = sum(list_of_matrices) / len(list_of_matrices)
    return mat


def sum_matrices(list_of_matrices):
    mat = sum(list_of_matrices)
    return mat


def diag(list_of_matrices):
    mat = np.diag(np.diag(list_of_matrices[0]))
    for m in list_of_matrices[1:]:
        mat = mat.dot(np.diag(np.diag(m)))
    return mat


def hadamard_product(a: np.array, b: np.array) -> np.array:
    return np.multiply(a, b)


def main():
    pass


if __name__ == "__main__":
    main()
