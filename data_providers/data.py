import os
# import cv2
import random
import tempfile
import numpy as np
from Queue import Queue
from threading import Thread
import nipy
from data_providers.base_provider import VideosDataset, DataProvider
from datasets.hdf5_loader import get_data


class Data(VideosDataset):
    def __init__(self, name, train_ids, normalization, sequence_length,
                 crop_size, num_classes, queue_size, all_hdf5_data):
        """
    Args:
      name: str, name of the data (train, test or validation)
      paths: list, list of string that have the video path and label
        information
      sequence_length: video clip length
      crop_size: `tuple`, image resize size (width, height)
      normalization: `str` or None
        None: no any normalization
        divide_255: divide all pixels by 255
        divide_256: divide all pixels by 256
      num_classes: `integer`, number of classes that the dataset has
      queue_size: `integer`, data queue size
    """
        self.name = name
        self.train_ids = train_ids
        self.normalization = normalization
        self.sequence_length = sequence_length  # slices
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.queue = DataQueue(name, queue_size)
        self.examples = None
        self.all_hdf5_data = all_hdf5_data
        self._start_data_thread()

    def get_frames_data(self, filename, num_frames_per_clip=16):
        ''' Given a directory containing extracted frames, return a video clip of
    (num_frames_per_clip) consecutive frames as a list of np arrays

    Args
      num_frames_per_clip: sequence_length of the video clip(slices)

    Returns
      mri: numpy, video clip with shape
        [sequence_length, height, width, channels]
    -------i'm splitters------------------------------
    video = []
    s_index = 0
    for parent, dirnames, files in os.walk(filename):
      filenames = [ fi for fi in files if fi.endswith((".nii", ".jpg", "jpeg")) ]
      if(len(filenames) < num_frames_per_clip):
        return None
      suffix = filenames[0].split('.', 1)[1]
      filenames_int = [i.split('.', 1)[0] for i in filenames]
      filenames_int = sorted(filenames_int)
      s_index = random.randint(0, len(filenames) - num_frames_per_clip)
      for i in range(s_index, s_index + num_frames_per_clip):
        image_name = str(filename) + '/' + str(filenames_int[i]) + '.' + suffix
        # print image_name
        img = cv2.imread(image_name)
        img = cv2.resize(img, self.crop_size)
        if self.normalization:
          img_data = self.normalize_image(img, self.normalization)
        video.append(img_data)
    return video
    '''
        mri = nipy.load_image(filename).get_data().transpose((1, 0, 2))
        data = np.zeros((num_frames_per_clip, mri.shape[1], mri.shape[2], 1), dtype='float32')
        mx = mri.max(axis=0).max(axis=0).max(axis=0)
        mri = np.array(mri) / mx
        if mri.shape[0] < num_frames_per_clip:
            return None
        if mri.shape[0] > num_frames_per_clip:
            mri = mri[:num_frames_per_clip, :, :]
        data[:, :, :, 0] = mri
        return data

    def extract_video_data(self):
        ''' Single tread to extract video and label information from the dataset
    '''
        # Generate one randome index and
        while True:
            # print 'enter for...'*n
            index = random.randint(0, len(self.train_ids) - 1)
            mri, label = get_data(self.train_ids[index], self.all_hdf5_data)
            self.queue.put((mri, label))

    def _start_data_thread(self):
        print("Start thread: %s data preparation ..." % self.name)
        self.worker = Thread(target=self.extract_video_data)
        self.worker.setDaemon(True)
        self.worker.start()

    @property
    def num_examples(self):

        return len(self.train_ids)

    def next_batch(self, batch_size):
        ''' Get the next batches of the dataset
    Args
      batch_size: video batch size

    Returns
      videos: numpy, shape
        [batch_size, sequence_length, height, width, channels]
      labels: numpy
        [batch_size, num_classes]
    '''
        # print "begin a batch"
        mris, labels = self.queue.get(batch_size)
        mris = np.array(mris)
        labels = np.array(labels)
        # labels = self.labels_to_one_hot(labels, self.num_classes)
        return mris, labels


class DataQueue():
    def __init__(self, name, maximum_item, block=True):
        """
    Args
      name: str, data type name (train, validation or test)
      maximum_item: integer, maximum item that this queue can store
      block: boolean, block the put or get information if the queue is
        full or empty
    """
        self.name = name
        self.block = block
        self.maximum_item = maximum_item
        self.queue = Queue(maximum_item)

    @property
    def queue(self):
        return self.queue

    @property
    def name(self):
        return self.name

    def put(self, data):
        self.queue.put(data, self.block)

    def get(self, batch_size):
        '''
    Args:
      batch_size: integer, the number of the item you want to get from the queue

    Returns:
      videos: list, list of numpy data with shape
        [sequence_length, height, width, channels]
      labels: list, list of integer number
    '''
        videos = []
        labels = []
        for i in range(batch_size):
            video, label = self.queue.get(self.block)
            videos.append(video)
            labels.append(label)
        return videos, labels


class DataProvider(DataProvider):
    def __init__(self, config, dataset_train, dataset_test, all_hdf5_data_train, all_hdf5_data_test, whichFoldData):
        """
    Args:
      num_classes: the number of the classes
      validation_set: `bool`.
      validation_split: `int` or None
          float: chunk of `train set` will be marked as `validation set`.
          None: if 'validation set' == True, `validation set` will be
              copy of `test set`
      normalization: `str` or None
          None: no any normalization
          divide_255: divide all pixels by 255
          divide_256: divide all pixels by 256
      sequence_length: `integer`, video clip length
      crop_size: `tuple`, the size that you want to reshape the images, (width, height)
      train: `boolean`, whether we need the training queue or not
      test: `test`, whether we need the testing queue or not
      queue_size: `integer`, data queue size , default is 300
    """

        self._sequence_length = 109
        self._crop_size = (91, 91)

        img, label = get_data(dataset_train[0], all_hdf5_data_train)

        print("step3")
        config.h = img.shape[0]
        config.w = img.shape[1]
        config.c = img.shape[2]

        # if len(img.shape) == 3:
        #     config.c = img.shape[2]
        # else:
        #     config.c = 1

        config.num_class = label.shape[0]
        self._num_classes = label.shape[0]

        if config.train:
            self.train = Data('train', dataset_train,
                              'std', self._sequence_length,
                              self._crop_size, config.num_class, config.queue_size, all_hdf5_data_train)
        if config.test:
            self.test = Data('test', dataset_test,
                             'std', self._sequence_length,
                             self._crop_size, config.num_class, config.queue_size, all_hdf5_data_test)

    def get_videos_labels_lines(self, path):
        # Open the file according to the filename
        lines = open(path, 'r')
        lines = list(lines)
        new_lines = [os.path.join(self._path, line) for line in lines]
        return new_lines

    @property
    def data_shape(self):
        return (self._sequence_length, self._crop_size[1], self._crop_size[0], 1)

    @property
    def n_classes(self):
        return self._num_classes
