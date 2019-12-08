import numpy as np
from keras.utils import Sequence
from keras_preprocessing import image
from keras.applications.resnet50 import preprocess_input

from PIL import Image

class ImageGenerator(Sequence):

    def __init__(self, list_IDs, labels, batch_size=32, dim=(224, 224), n_channels=3,
                 n_classes=10, shuffle=True):

        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, * self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        for i, image_path in enumerate(list_IDs_temp):
            img = image.load_img("Market-1501/" + image_path, target_size=(224, 224, 3))
            
            img_array = image.img_to_array(img)
            img_array = preprocess_input(img_array)

            img_index = image_path[:4]

            labels = np.delete(self.labels[img_index], 0) - 1
            
            X[i, ] = img_array
            y[i, ] = labels
            
        return X, y