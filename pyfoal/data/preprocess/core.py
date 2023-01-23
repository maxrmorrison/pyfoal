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
        directory = pyfoal.CACHE_DIR / dataset

        # Get text files
        text_files = directory.rglob('*.txt')

        # Get output phoneme files
        phoneme_files = [file.with_suffix('.pt') for file in text_files]

        # Grapheme-to-phoneme
        pyfoal.g2p.from_files_to_files(text_files, phoneme_files)
