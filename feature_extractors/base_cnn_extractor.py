from keras.models import Model
import numpy as np
from utils.preprocessing import Preprocessing
from utils.io_handler import IOHandler
from tqdm import tqdm


class BaseCNNExtractor:

    def __init__(
            self,
            model: Model,
            preprocess_routine: callable,
            target_image_size: tuple,
            verbose: bool = True):
        self.model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        self.preprocess_routine = preprocess_routine
        self.target_image_size = target_image_size
        self.verbose = verbose

    def extract(self, image_path: str, preprocess_data: bool = True) -> np.array:

        data = Preprocessing.image_to_array(
            image_path=image_path,
            preprocess_data=preprocess_data,
            target_image_size=self.target_image_size,
            preprocess_routine=self.preprocess_routine)
        features = self.model.predict(data, verbose=0)

        return features

    def batch_extract(self, dataset_path: str, preprocess_data: bool = True) -> np.array:

        features = []

        files = IOHandler.get_file_paths(dataset_path)
        filepaths = Preprocessing.isolate_features(files).flatten()
        classes = Preprocessing.isolate_classes(files).flatten()

        for layer in self.model.layers:
            layer.trainable = False

        iterator = zip(filepaths, classes)

        for x, y in tqdm(iterator, total=len(classes), ncols=100) if self.verbose else iterator:
            _features = self.extract(x, preprocess_data)
            features.append(np.append(_features.flatten(), int(y)))

        return np.array(features)
