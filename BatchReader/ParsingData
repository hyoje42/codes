import numpy as np
import os
import random
from six.moves import cPickle as pickle
import glob

def create_image_lists(image_dir):
    if not os.path.isdir(image_dir):
        print("There is no directoriy '{}'".format(image_dir))
    directories = ['train']
    image_list = {}
    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = os.path.join(image_dir, '*.' + 'tif')
        file_list.extend(glob.glob(file_glob))
        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]
                record = {'image': f, 'filename': filename}
                image_list[directory].append(record)
        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))
    return image_list


def read_dataset(image_dir):
    pickle_filename = "train_list.pickle"
    pickle_filepath = os.path.join(pickle_filename)
    if not os.path.exists(pickle_filepath):
        result = create_image_lists(image_dir)
        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['train']
        del result
    print("Make pickle files.")
    
    return training_records
