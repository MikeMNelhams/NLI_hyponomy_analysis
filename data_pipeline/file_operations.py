import json
import os
import random


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


def is_file(file_path: str, extension: str) -> bool:
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


class JSON_writer:
    def __init__(self, file_path: str):
        assert file_path_is_of_extension(file_path, '.json')
        self.file_path = file_path
        self.data = {}

        if self.file_exists:
            self.load()

    @property
    def file_exists(self):
        return is_file(self.file_path, '.json')

    def load(self) -> None:
        assert self.file_exists, FileNotFoundError
        with open(self.file_path, 'r') as json_file:
            self.data = json.load(json_file)
        return None

    def save(self) -> None:
        with open(self.file_path, 'w') as json_file:
            json.dump(self.data, json_file)
        return None
