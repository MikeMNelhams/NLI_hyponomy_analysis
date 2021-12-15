import json
import os
import pickle


class InvalidPathError(Exception):
    """ When a path could never point to a file, e.g missing .extension"""
    pass


def list_files_in_directory(directory_path: str, extension_type: str = '.txt') -> list:
    child_file_paths = os.listdir(directory_path)
    return [file_path for file_path in child_file_paths
            if file_path_is_of_extension(file_path, extension=extension_type)]


def file_path_is_of_extension(file_path: str, extension: str= '.txt') -> bool:
    """ Does not assume that the file exists"""
    if len(file_path) <= len(extension):
        return False
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


def is_file(file_path: str, extension: str) -> bool:
    """ Assumes that the file exists """
    if not os.path.isfile(file_path):
        return False
    if len(file_path) < 3:
        return False
    if len(file_path) <= len(extension):
        return False
    if file_path[-len(extension):] != extension:
        return False
    return True


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
        return is_file(self.file_path, '.json')

    @property
    def file_empty(self):
        return self.file_exists and os.stat(self.file_path).st_size == 0

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


if __name__ == "__main__":
    pass
