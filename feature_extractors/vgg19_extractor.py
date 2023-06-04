from keras.applications.vgg19 import VGG19, preprocess_input
from numpy import array
from feature_extractors import base_cnn_extractor


class VGG19Extractor(base_cnn_extractor.BaseCNNExtractor):

    def __init__(self, image_target_size: tuple, verbose: bool = True):
        model = VGG19(weights='imagenet', include_top=True)
        super().__init__(model, preprocess_input, image_target_size, verbose)

    def batch_extract(self, dataset_path: str, preprocess_data: bool = True) -> array:
        return super().batch_extract(dataset_path, preprocess_data=True)

    def extract(self, image_path: str, preprocess_data: bool = True):
        return super().extract(image_path, preprocess_data)
