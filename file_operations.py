import os


def list_files_in_directory(directory_path: str, extension_type: str = '.txt') -> list:
    child_file_paths = os.listdir(directory_path)
    return [file_path for file_path in child_file_paths if file_path_is_extension(file_path, extension=extension_type)]


def file_path_is_extension(file_path: str, extension: str= '.txt') -> bool:
    """ Does not assume that the file exists"""
    if len(file_path) <= len(extension):
        return False
    if file_path[-len(extension):] != extension:
        return False
    return True