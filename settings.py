# Pre-processing
NORMALIZE = True
APPLY_SMOTE = True
USE_CONCAT_FEATURES = True
VERBOSE = True
DEFAULT_IMAGE_SIZE = (224, 224)
FEATURE_EXTRACTORS = ['Densenet169',
                      'Densenet201',
                      'Mobilenet',
                      'MobilenetV2',
                      'Resnet50',
                      'Resnet101',
                      'Resnet152',
                      'VGG16',
                      'VGG19']

# Breakhis initial configuration
DEFAULT_DATA_FOLDER = 'data'
SOURCE_BREAKHIS_FOLDER = r'D:\TCC\breast'
DATASET_CLASSES = {'M': 'malignant', 'B': 'benign'}
MAGNIFICATIONS = [40, 100, 200, 400]

# Testing
ROUNDS_OF_TEST = 10
TRAIN_RATIO = .8
TEST_RATIO = .2
