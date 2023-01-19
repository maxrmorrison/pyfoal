import pyfoal


###############################################################################
# Preprocess
###############################################################################


def datasets(datasets):
    """Preprocess a dataset
    Arguments
        name - string
            The name of the dataset to preprocess
    """
    for dataset in datasets:
        input_directory = pyfoal.DATA_DIR / dataset
        output_directory = pyfoal.CACHE_DIR / dataset

        # TODO - Perform preprocessing
        raise NotImplementedError
