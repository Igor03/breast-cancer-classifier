import os
import settings
import numpy as np
import shutil

from utils.preprocessing import Preprocessing
from utils.io_handler import IOHandler
from classifiers.svm import SVM
from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_score, recall_score, f1_score
from feature_extractors.resnet101_extractor import Resnet101Extractor as Extractor
from prettytable import PrettyTable


def prepare_dataset(
        original_dataset_path: str = settings.SOURCE_BREAKHIS_FOLDER,
        prepared_dataset_path: str = settings.DEFAULT_DATA_FOLDER) -> str:

    organized_dataset_path = IOHandler.prepare_breakhis_dataset(
        original_dataset_path,
        dest_folder_path=os.path.join(prepared_dataset_path, 'breakhis'))

    # Todo: Fix mixed slashes bug
    resized_dataset_path = os.path.abspath(
        os.path.join(prepared_dataset_path, 'resized_breakhis')).replace('\\', '/')
    organized_dataset_path = os.path.abspath(organized_dataset_path).replace('\\', '/')

    for magnification in settings.MAGNIFICATIONS:
        Preprocessing.resize_dataset(
            original_dataset_path=os.path.join(f'{organized_dataset_path}/', f'{magnification}X'),
            new_dataset_path=os.path.join(f'{resized_dataset_path}/', f'{magnification}X'),
            classes=('benign', 'malignant'),
            new_size=settings.DEFAULT_IMAGE_SIZE,
            save_files=True)

    shutil.rmtree(organized_dataset_path)

    return resized_dataset_path


def get_classification_results(
        rounds: int,
        dataset: np.array,
        apply_smote: bool) -> tuple:

    # Arrays to store the predictions metrics
    _accuracy = []
    _kappa = []
    _f1 = []
    _precision = []
    _recall = []

    for _round in range(rounds):

        # Splitting data into train and test randomly
        train, test = Preprocessing.train_test_split(dataset)

        # Generating synthetic data to equalize the amount of data belonging to both classes
        train = Preprocessing.apply_smote(train) if apply_smote else train

        # Isolating classes and features for the training and test steps
        x_train = Preprocessing.isolate_features(train)
        y_train = Preprocessing.isolate_classes(train)
        x_test = Preprocessing.isolate_features(test)
        y_test = Preprocessing.isolate_classes(test)

        # Fitting the classifier with pre-defined hyperparams
        classifier = SVM(x_train, y_train.flatten()).train()

        # Getting classification results for the test data
        predictions = classifier.predict(x_test).flatten()

        _accuracy.append(accuracy_score(y_test.flatten(), predictions))
        _kappa.append(cohen_kappa_score(y_test.flatten(), predictions))
        _precision.append(precision_score(y_test.flatten(), predictions))
        _recall.append(recall_score(y_test.flatten(), predictions))
        _f1.append(f1_score(y_test.flatten(), predictions))

    return (
        np.average(_accuracy),
        np.average(_kappa),
        np.average(_precision),
        np.average(_recall),
        np.average(_f1))


if __name__ == '__main__':

    # ---------------------------------------------------------------------------------------------------
    # Preparing the BreakHis dataset. Here, we expect to receive the dataset in its original form
    # beginning in the folder 'breast'.
    # It's important to set all the paths in the settings.py file or else the preparation might not work
    # as expected.
    # Also, the output folder will contain all the files in the breakhis dataset organized by
    # magnification and class as follows:
    # output_path/
    #   40X/
    #     malignant/
    #     benign/
    #   100X
    #     malignant/
    #     benign/
    #   ...
    # ---------------------------------------------------------------------------------------------------
    # prepare_dataset()

    # Defining the dataset path
    magnification = settings.MAGNIFICATIONS[2]
    images_dataset_path = f'{settings.DEFAULT_DATA_FOLDER}/resized_breakhis/{magnification}X'

    # ---------------------------------------------------------------------------------------------
    # Extracting the deep learning features from the images
    # Alternatively, you can save the deep learning features into a .csv file and then load it
    # to memory to avoid the need to extract the same features again.
    # ---------------------------------------------------------------------------------------------

    feature_extractor = Extractor(settings.DEFAULT_IMAGE_SIZE, verbose=True)
    deep_learning_features = feature_extractor.batch_extract(dataset_path=images_dataset_path)

    IOHandler.save_as_csv(
        data=deep_learning_features,
        filepath=f'{settings.DEFAULT_DATA_FOLDER}/features',
        filename=f'{Extractor.__name__}_{magnification}X')

    # deep_learning_features = IOHandler.read_csv(
    #     filepath=f'{settings.DEFAULT_DATA_FOLDER}/feature_files',
    #     filename='my_features')

    deep_learning_features = Preprocessing.min_max_normalization(deep_learning_features)

    (acc, kappa, precision, recall, f1) = get_classification_results(
                                                settings.ROUNDS_OF_TEST,
                                                deep_learning_features,
                                                settings.APPLY_SMOTE)

    table = PrettyTable()
    table.field_names = ['Metric', f'Average result for {settings.ROUNDS_OF_TEST} iterations']
    table.add_row(['Accuracy', acc])
    table.add_row(['Kappa score', kappa])
    table.add_row(['Precision', precision])
    table.add_row(['Recall', recall])
    table.add_row(['F1 score', f1])

    print(table)

