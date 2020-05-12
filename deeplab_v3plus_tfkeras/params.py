# import numpy as np
# import PIL.Image as Image

class params():
    def __init__(self,
                 n_categories=None,
                 image_size=(None, None),
                 n_epochs=None):

        self.n_categories = n_categories
        self.image_size = image_size

    def check(self):
        if self.n_categories is None:
            raise Exception("n_categories is not set")
        if self.image_size == (None, None):
            raise Exception("image_size is not set")
        if self.n_epochs is None:
            raise Exception("n_epochs is not set")
