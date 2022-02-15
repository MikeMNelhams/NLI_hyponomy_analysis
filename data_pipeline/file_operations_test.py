import os
import unittest
import file_operations as file_op


class TestTeardown:
    def __init__(self, dir_path: str):
        self.__dir_path = dir_path

    def __delete(self) -> None:
        if file_op.is_file(self.__dir_path):
            os.remove(self.__dir_path)
        return None

    def __enter__(self):
        self.__delete()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__delete()


def make_blank_file(file_path: str, num_lines_non_blank: int, num_lines_blank: int) -> None:
    if num_lines_non_blank == 0:
        lines = ['\n' for _ in range(num_lines_blank - 1)]

        with open(file_path, 'w') as out_file:
            out_file.writelines(lines)

        return None

    lines = ["test\n" for _ in range(num_lines_non_blank - 1)]
    if num_lines_non_blank > 1:
        lines += "test"

    if num_lines_blank > 0:
        lines += ['\n' for _ in range(num_lines_blank)]

    with open(file_path, 'w') as out_file:
        out_file.writelines(lines)

    return None


class TestFilePathOfExtension(unittest.TestCase):
    def test_txt_is_txt_returnTrue(self):
        self.assertTrue(file_op.file_path_is_of_extension("test.txt", ".txt"))

    def test_txt_is_csv_returnFalse(self):
        self.assertFalse(file_op.file_path_is_of_extension("test.txt", '.csv'))

    def test_csv_is_csv_returnTrue(self):
        self.assertTrue(file_op.file_path_is_of_extension("test.csv", '.csv'))

    def test_csv_is_txt_returnFalse(self):
        self.assertFalse(file_op.file_path_is_of_extension("test.csv", '.txt'))

    def test_empty_but_correct_extension_False(self):
        self.assertFalse(file_op.file_path_is_of_extension(".csv", ".csv"))

    def test_empty_and_incorrect_extension_False(self):
        self.assertFalse(file_op.file_path_is_of_extension(".txt", ".csv"))

    def test_has_csv_but_no_dot(self):
        self.assertFalse(file_op.file_path_is_of_extension("test_csv", ".csv"))

    def test_extension_to_check_has_no_dot(self):
        with self.assertRaises(file_op.InvalidPathError):
            file_op.file_path_is_of_extension("test.csv", "csv")


class TestCountLines(unittest.TestCase):
    def test_blank1(self):
        test_path = "test.txt"

        with TestTeardown(test_path):
            make_blank_file(test_path, 10, 2)
            self.assertEqual(12, file_op.count_file_lines(test_path))

    def test_blank2(self):
        test_path = "test.txt"

        with TestTeardown(test_path):
            make_blank_file(test_path, 10, 1)
            self.assertEqual(11, file_op.count_file_lines(test_path))

    def test_blank3(self):
        test_path = "test.txt"

        with TestTeardown(test_path):
            make_blank_file(test_path, 10, 0)
            file_op.trim_end_of_file_blank_line(test_path)
            self.assertEqual(10, file_op.count_file_lines(test_path))

    def test_blank4(self):
        test_path = "test.txt"

        with TestTeardown(test_path):
            make_blank_file(test_path, 1, 2)
            self.assertEqual(3, file_op.count_file_lines(test_path))

    def test_blank5(self):
        test_path = "test.txt"

        with TestTeardown(test_path):
            make_blank_file(test_path, 1, 1)
            self.assertEqual(2, file_op.count_file_lines(test_path))

    def test_blank6(self):
        test_path = "test.txt"

        # with TestTeardown(test_path):
        make_blank_file(test_path, 0, 1)
        self.assertEqual(1, file_op.count_file_lines(test_path))

    def test_blank7(self):
        test_path = "test.txt"

        with TestTeardown(test_path):
            make_blank_file(test_path, 0, 0)
            self.assertEqual(1, file_op.count_file_lines(test_path))


class TestTrimEmptyLines(unittest.TestCase):
    """ Relies on TestCountLines to pass all tests."""

    def test_blank1(self):
        test_path = "test.txt"

        with TestTeardown(test_path):
            make_blank_file(test_path, 10, 2)
            file_op.trim_end_of_file_blank_line(test_path)
            self.assertEqual(10, file_op.count_file_lines(test_path))

    def test_blank2(self):
        test_path = "test.txt"

        with TestTeardown(test_path):
            make_blank_file(test_path, 10, 1)
            file_op.trim_end_of_file_blank_line(test_path)
            self.assertEqual(10, file_op.count_file_lines(test_path))

    def test_blank3(self):
        test_path = "test.txt"

        with TestTeardown(test_path):
            make_blank_file(test_path, 10, 0)
            file_op.trim_end_of_file_blank_line(test_path)
            self.assertEqual(10, file_op.count_file_lines(test_path))

    def test_blank4(self):
        test_path = "test.txt"

        with TestTeardown(test_path):
            make_blank_file(test_path, 1, 2)
            file_op.trim_end_of_file_blank_line(test_path)
            self.assertEqual(1, file_op.count_file_lines(test_path))

    def test_blank5(self):
        test_path = "test.txt"

        with TestTeardown(test_path):
            make_blank_file(test_path, 1, 1)
            file_op.trim_end_of_file_blank_line(test_path)
            self.assertEqual(1, file_op.count_file_lines(test_path))

    def test_blank6(self):
        test_path = "test.txt"

        with TestTeardown(test_path):
            make_blank_file(test_path, 1, 0)
            file_op.trim_end_of_file_blank_line(test_path)
            self.assertEqual(1, file_op.count_file_lines(test_path))

    def test_blank7(self):
        test_path = "test.txt"

        with TestTeardown(test_path):
            make_blank_file(test_path, 0, 2)
            file_op.trim_end_of_file_blank_line(test_path)
            self.assertEqual(1, file_op.count_file_lines(test_path))

    def test_blank8(self):
        test_path = "test.txt"

        with TestTeardown(test_path):
            make_blank_file(test_path, 0, 1)
            file_op.trim_end_of_file_blank_line(test_path)
            self.assertEqual(1, file_op.count_file_lines(test_path))

    def test_blank9(self):
        test_path = "test.txt"

        with TestTeardown(test_path):
            make_blank_file(test_path, 0, 0)
            file_op.trim_end_of_file_blank_line(test_path)
            self.assertEqual(1, file_op.count_file_lines(test_path))

    def test_double_trim(self):
        test_path = "test.txt"

        with TestTeardown(test_path):
            make_blank_file(test_path, 10, 5)
            file_op.trim_end_of_file_blank_line(test_path)
            file_op.trim_end_of_file_blank_line(test_path)
            self.assertEqual(10, file_op.count_file_lines(test_path))


if __name__ == '__main__':
    unittest.main()
