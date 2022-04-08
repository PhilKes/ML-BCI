from sys import stdout

from tqdm import tqdm


class ProgressWrapper(tqdm):
    file = stdout

    def __init__(self, iterable, **kwargs):
        super().__init__(iterable, **kwargs, file=ProgressWrapper.file)
