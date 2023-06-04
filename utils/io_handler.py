import numpy as np
import os
import shutil
import re
import cv2
import settings

from prettytable import PrettyTable
from tqdm import tqdm


class IOHandler:

    @staticmethod
    def get_file_paths(root_path: str = None) -> np.array:
        filepaths = []
        for classname in os.listdir(root_path):
            class_path = os.path.join(root_path, classname)
            for filename in os.listdir(class_path):
                full_filepath = os.path.join(class_path, filename)
                filepaths.append([full_filepath, '1' if classname == 'malignant' else '0'])
        return np.array(filepaths)

    @staticmethod
    def create_folder(folder_path: str, delete_if_exists: bool = False,  multiple: bool = False) -> str:
        folder_exists = os.path.exists(path=folder_path)
        routine = os.makedirs if multiple else os.mkdir
        if not folder_exists:
            routine(folder_path)
        elif folder_exists and delete_if_exists:
            shutil.rmtree(folder_path)
            routine(folder_path)
        return folder_path

    @staticmethod
    def copy_file(source_filepath: str, dest_filepath: str) -> any:
        if not os.path.exists(dest_filepath):
            IOHandler.create_folder(dest_filepath)
        shutil.copy(source_filepath, dest_filepath)

    @staticmethod
    def file_exists(filepath: str) -> bool:
        return os.path.exists(filepath)

    @staticmethod
    def get_file_size(filepath: str) -> int:
        if IOHandler.file_exists(filepath):
            return os.path.getsize(filepath)

    @staticmethod
    def save_as_csv(data: np.array,  filepath: str = None, filename: str = None) -> str:
        if filepath:
            if not os.path.exists(filepath):
                IOHandler.create_folder(filepath, delete_if_exists=False, multiple=True)
        else:
            filepath = os.path.join(IOHandler.__get_application_root_path(), settings.DEFAULT_DATA_FOLDER)
        filepath = os.path.join(filepath, f'{filename}.csv')
        np.savetxt(filepath, data, delimiter=',')
        return filepath

    @staticmethod
    def split_path(path: str, exclude_file: bool = True) -> list:
        parts = path.split('\\')
        # Idk about that... hahahaha
        return parts if not exclude_file else parts[:-1] if '.' in parts[-1] else parts

    @staticmethod
    def join_path_parts(parts: list) -> str:
        return '\\'.join(parts)
        # return os.path.join(*parts)

    @staticmethod
    def read_csv(filepath: str, filename: str) -> np.array:

        if not filename.endswith('csv'):
            filename = f'{filename}.csv'
        full_filepath = os.path.join(filepath, filename)
        return np.loadtxt(full_filepath, delimiter=',')

    @staticmethod
    def print_results_table(extractor_name: str, results: list) -> None:
        table = PrettyTable()
        table.field_names = ['Round', 'Extractor', 'Accuracy', 'Kappa Score']
        for _round, result in enumerate(results, start=1):
            table.add_row([_round, extractor_name, result[0], result[1]])
        print(table)
        table.clear()

    @staticmethod
    def resize_image(
            original_image_path: str,
            original_image_name: str,
            new_image_path: str,
            new_image_name: str,
            new_size: tuple,
            save_image: bool = True) -> str:
        img = cv2.imread(os.path.join(original_image_path, original_image_name))
        img = cv2.resize(img, new_size)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if save_image:
            cv2.imwrite(os.path.join(new_image_path, new_image_name), img)
        return img

    @staticmethod
    def prepare_breakhis_dataset(
            source_breakhis_path: str,
            dest_folder_name: str = 'datasets',
            dest_folder_path: str = None,
            verbose: str = True) -> str:

        output_dataset_path = dest_folder_path

        if not output_dataset_path:
            output_dataset_path = IOHandler.__get_application_root_path()
            output_dataset_path = os.path.join(output_dataset_path, settings.DEFAULT_DATA_FOLDER, dest_folder_name)

        source_filepaths = []
        endpoints = [(root, files) for (root, _, files) in os.walk(source_breakhis_path) if root.endswith('0X')]
        [source_filepaths.extend(list(map(lambda x: os.path.join(root, x), files))) for root, files in endpoints]

        for source_filepath in (tqdm(source_filepaths, ncols=100, colour='white') if verbose else source_filepaths):
            filename = source_filepath.split('\\')[-1]
            dest_filepath = IOHandler.__map_folder(str(filename))
            dest_filepath = os.path.join(str(output_dataset_path), dest_filepath)
            dest_folder_path = IOHandler.join_path_parts(IOHandler.split_path(dest_filepath))

            if not os.path.exists(dest_folder_path):
                IOHandler.create_folder(dest_folder_path, multiple=True)
            IOHandler.copy_file(source_filepath, dest_folder_path)

        return output_dataset_path

    @staticmethod
    def __map_folder(filename: str) -> str:
        _class = settings.DATASET_CLASSES[re.search(r'(?<=[A-Z]_)(.*?)(?=_)', filename).group()]
        magnification = filename.split('-')[-2] + 'X'
        return os.path.join(magnification, _class, filename)

    @staticmethod
    def __get_application_root_path() -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
