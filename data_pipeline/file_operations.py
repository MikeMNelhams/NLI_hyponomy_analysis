import json
import os
import pickle

from typing import List


class InvalidPathError(Exception):
    """ When a path could never point to a file, e.g missing .extension"""
    pass


def list_files_in_directory(directory_path: str, extension_type: str = '.txt') -> list:
    child_file_paths = os.listdir(directory_path)
    return [file_path for file_path in child_file_paths
            if file_path_is_of_extension(file_path, extension=extension_type)]


def file_path_is_of_extension(file_path: str, extension: str= '.txt') -> bool:
    """ Only checks the string, not that file exists"""
    if len(file_path) <= len(extension):
        return False

    if '.' not in extension:
        raise InvalidPathError

    if file_path[-len(extension):] != extension:
        return False
    return True


def file_path_extension(file_path: str) -> str:
    dot_index = -1
    reversed_index_iterator = range(len(file_path) - 1, -1, -1)
    for index in reversed_index_iterator:
        if file_path[index] == '.':
            dot_index = index
            break
    if dot_index != -1:
        return file_path[dot_index:]
    raise InvalidPathError


def count_file_lines(file_path: str) -> int:
    with open(file_path, "r") as file:
        data = file.read()

    if len(data) == 0:
        return 1

    number_of_lines = sum(1 for line in data if line[-1] == '\n') + 1

    return number_of_lines


def is_file(file_path: str) -> bool:
    """ Assumes that the file exists """
    if len(file_path) == 0:
        return False
    if not os.path.isfile(file_path):
        return False
    if len(file_path) < 2:
        return False
    return True


def is_dir(file_path: str) -> bool:
    if len(file_path) == 0:
        return False
    return os.path.isdir(file_path)


def make_dir(file_path: str) -> None:
    if file_path[-1] != '/':
        raise InvalidPathError
    os.mkdir(file_path)
    return None


def dirname(file_path: str) -> str:
    return os.path.dirname(file_path)


def file_path_without_extension(file_path: str) -> str:
    """ Does not assume the file exists """

    if '.' not in file_path:
        raise InvalidPathError

    stop_index = 0
    for i, character in enumerate(reversed(file_path)):
        if character == '.':
            stop_index = i + 1
            break

    return file_path[:-stop_index]


def trim_end_of_file_blank_line(file_path: str) -> None:
    with open(file_path, 'r') as in_file:
        data = in_file.read()

    with open(file_path, 'w') as out_file:
        out_file.write(data.rstrip('\n'))

    return None


def make_empty_file(file_path: str) -> None:
    """Use make_empty_file_safe if you don't want to overwrite data"""

    with open(file_path, "w") as outfile:
        outfile.write('')
    return None


def make_empty_file_safe(file_path: str) -> None:
    if os.path.isfile(file_path):
        raise FileExistsError
    make_empty_file(file_path)
    return None


def load_print_decorator(func: callable) -> callable:
    def wrapper(*args, **kwargs):
        file_path = args[0].file_path
        print(f"Loading file: {file_path}")
        data = func(*args, **kwargs)
        print(f"Finished loading file: {file_path}")
        return data
    return wrapper


def save_print_decorator(func: callable) -> callable:
    def wrapper(*args, **kwargs):
        file_path = args[0].file_path
        print(f"Saving to file: {file_path}")
        data = func(*args, **kwargs)
        print(f"Finished saving to file: {file_path}")
        return data
    return wrapper


class DictWriter:
    supported_file_extensions = ('.p', '.json')

    def __init__(self, file_path: str):
        assert file_path_extension(file_path) in DictWriter.supported_file_extensions, InvalidPathError
        self.file_path = file_path

    @property
    def file_exists(self):
        return is_file(self.file_path)

    @property
    def file_empty(self):
        return self.file_exists and os.stat(self.file_path).st_size == 0

    @property
    def keys(self) -> list:
        if not self.file_exists:
            raise KeyError
        return list(self.load().keys())

    @load_print_decorator
    def load(self) -> dict:
        file_extension = file_path_extension(self.file_path)
        if file_extension == ".p":
            return self.__load_pickle()
        if file_extension == ".json":
            return self.__load_json()
        raise InvalidPathError

    @save_print_decorator
    def save(self, data: dict) -> None:
        assert type(data) == dict, TypeError
        file_extension = file_path_extension(self.file_path)
        if file_extension == ".p":
            return self.__save_pickle(data)
        if file_extension == ".json":
            return self.__save_json(data)
        print(f"Finished saving to file: {self.file_path}")
        return None

    def __load_pickle(self) -> dict:
        with open(self.file_path, "rb") as pickle_file:
            data = pickle.load(pickle_file)
        return data

    def __save_pickle(self, data: dict) -> None:
        with open(self.file_path, "wb") as pickle_file:
            pickle.dump(data, pickle_file)
        return None

    def __load_json(self) -> dict:
        with open(self.file_path, 'r') as json_file:
            data = json.load(json_file)
        return data

    def __save_json(self, data: dict) -> None:
        with open(self.file_path, 'w') as json_file:
            json.dump(data, json_file)
        return None


class TextWriterSingleLine:
    def __init__(self, file_path: str):
        self.file_path = file_path

    @property
    def file_exists(self):
        return is_file(self.file_path)

    @property
    def file_empty(self):
        return self.file_exists and os.stat(self.file_path).st_size == 0

    @load_print_decorator
    def load(self) -> str:
        file_extension = file_path_extension(self.file_path)
        if file_extension == ".txt":
            return self.__load_text()
        raise InvalidPathError

    @load_print_decorator
    def load_safe(self):
        if not self.file_exists:
            return None

        return self.load()

    @save_print_decorator
    def save(self, data: str) -> None:
        assert hasattr(data, "__repr__")
        file_extension = file_path_extension(self.file_path)
        if file_extension == ".txt":
            self.__save_text(str(data))
        return None

    def __load_text(self) -> str:
        with open(self.file_path, "r") as text_file:
            data = text_file.read()
        return data

    def __save_text(self, data) -> None:
        with open(self.file_path, "w") as text_file:
            text_file.write(data)
        return None


class TextLogger:
    def __init__(self, file_path: str):
        assert file_path_extension(file_path) == ".txt", InvalidPathError
        self.file_path = file_path

    @property
    def file_exists(self):
        return is_file(self.file_path)

    @property
    def file_empty(self):
        return self.file_exists and os.stat(self.file_path).st_size == 0

    @load_print_decorator
    def load(self) -> List[str]:
        file_extension = file_path_extension(self.file_path)
        if file_extension == ".txt":
            return self.__load_text()
        raise InvalidPathError

    @load_print_decorator
    def load_safe(self):
        if not self.file_exists:
            return None

        return self.load()

    @save_print_decorator
    def save(self, data: list) -> None:
        assert type(data) == list, TypeError
        file_extension = file_path_extension(self.file_path)
        if file_extension == ".txt":
            self.__save_text(data)
        return None

    def append_lines(self, lines: list) -> None:
        if not self.file_exists:
            self.__save_text(lines)
        else:
            with open(self.file_path, 'a') as text_file:
                text_file.writelines('\n'.join(lines))
                text_file.write('\n')
        return None

    def remove_last_line(self) -> None:
        with open(self.file_path, 'r') as text_file:
            content = text_file.readlines()

        with open(self.file_path, 'w') as text_file:
            text_file.writelines(content[:-1])

    def __load_text(self) -> list:
        with open(self.file_path, "r") as text_file:
            data = text_file.readlines()
        return data

    def __save_text(self, lines: list) -> None:
        with open(self.file_path, "w") as text_file:
            text_file.writelines('\n'.join(lines))
            text_file.write('\n')
        return None


if __name__ == "__main__":
    pass
