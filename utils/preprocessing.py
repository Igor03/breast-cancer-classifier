import numpy as np
import os
import settings

from utils.io_handler import IOHandler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from keras.utils import load_img, img_to_array
from typing import Callable
from tqdm import tqdm


class Preprocessing:

    @staticmethod
    def image_to_array(
            image_path: np.array,
            target_image_size: tuple,
            preprocess_data: bool,
            preprocess_routine: Callable) -> np.array:

        image_data = img_to_array(load_img(image_path, target_size=target_image_size))
        image_data = image_data.reshape((1, image_data.shape[0], image_data.shape[1], image_data.shape[2]))
        image_data = preprocess_routine(image_data) if preprocess_data else image_data

        return image_data

    @staticmethod
    def apply_smote(dataset: np.array):
        classes = Preprocessing.isolate_classes(dataset)
        features_only = Preprocessing.isolate_features(dataset)
        smote = SMOTE()
        resampled_features, resampled_classes = smote.fit_resample(features_only, classes)
        return Preprocessing.concat_classes(resampled_features, resampled_classes)

    @staticmethod
    def min_max_normalization(dataset: np.array, consider_classes: bool = True):
        if consider_classes:
            labels = Preprocessing.isolate_classes(dataset)
            features = Preprocessing.isolate_features(dataset)
            _min = features.min()
            _max = features.max()
            normalized_data = np.vectorize(lambda x: (x - _min) / (_max - _min))(features)
            return Preprocessing.concat_classes(normalized_data, labels)
        _min = dataset.min()
        _max = dataset.max()
        return np.vectorize(lambda x: (x - _min) / (_max - _min))(dataset)

    @staticmethod
    def train_test_split(dataset: np.array, test_size=settings.TEST_RATIO) -> tuple:
        y = Preprocessing.isolate_classes(dataset)
        x = Preprocessing.isolate_features(dataset)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        train = Preprocessing.concat_classes(x_train, y_train)
        test = Preprocessing.concat_classes(x_test, y_test)
        return train, test

    @staticmethod
    def isolate_features(dataset: np.array) -> np.array:
        return dataset[:, :dataset.shape[1] - 1]

    @staticmethod
    def isolate_classes(dataset: np.array) -> np.array:
        return dataset[:, dataset.shape[1] - 1:dataset.shape[1]]

    @staticmethod
    def concat_classes(dataset: np.array, classes: np.array) -> np.array:
        return np.concatenate((dataset, classes.reshape(len(classes), 1)), axis=1)

    @staticmethod
    def concat_features(dataset_1: np.array, dataset_2: np.array) -> np.array:
        isolated_bit_features = Preprocessing.isolate_features(dataset_1)
        isolated_bit_classes = Preprocessing.isolate_classes(dataset_1)
        isolated_cnn_features = Preprocessing.isolate_features(dataset_2)
        isolated_cnn_classes = Preprocessing.isolate_classes(dataset_2)
        if not np.array_equal(isolated_bit_classes, isolated_cnn_classes):
            return np.empty()
        concatenated_features = np.concatenate((isolated_bit_features, isolated_cnn_features), axis=1)
        return Preprocessing.concat_classes(concatenated_features, isolated_bit_classes)

    @staticmethod
    def resize_dataset(
            original_dataset_path: str,
            new_dataset_path: str,
            classes: tuple,
            new_size: tuple,
            save_files: bool) -> tuple:
        resized_images = []
        created_folder_path = IOHandler.create_folder(new_dataset_path, delete_if_exists=True, multiple=True)
        class1_path = IOHandler.create_folder(os.path.join(created_folder_path, classes[0]), delete_if_exists=True, multiple=True)
        class2_path = IOHandler.create_folder(os.path.join(created_folder_path, classes[1]), delete_if_exists=True, multiple=True)
        all_file_paths = IOHandler.get_file_paths(original_dataset_path)
        for path in tqdm(all_file_paths, ncols=100, colour='white'):
            _class = int(path[-1])
            _path_parts = IOHandler.split_path(path[0], exclude_file=False)
            resized_image = IOHandler.resize_image(
                    original_image_path=IOHandler.join_path_parts(_path_parts[0:2]),
                    original_image_name=_path_parts[-1],
                    new_image_path=class1_path if _class == 0 else class2_path,
                    new_image_name=_path_parts[-1],
                    new_size=new_size,
                    save_image=save_files)
            resized_images.append(resized_image)
        return created_folder_path, np.array(resized_images)

