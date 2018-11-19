import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = r'C:\Users\Hyoje\Desktop\py3.6\datasets\horse2zebra'

def load_imgs(folder, want_size, is_to_RGB=True):
    path = os.path.join(DATASET_PATH, folder)
    imgs = [cv2.imread(os.path.join(path, file)) for file in os.listdir(path)]
    imgs = np.array([cv2.resize(im, (want_size, want_size), cv2.INTER_LINEAR) for im in imgs])
    
    if is_to_RGB:
        # BGR to RGB
        b, g, r = imgs[:, :, :, 0], imgs[:, :, :, 1], imgs[:, :, :, 2]
        expands = [np.expand_dims(chanel, axis=-1) for chanel in [r, g, b]]
        imgs = np.concatenate(expands, axis=-1)
    
    return imgs
    
    
def merge(images, shape, want_size=128):
    h, w = want_size, want_size
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * shape[0], w * shape[1], c))
        for idx, image in enumerate(images):
            img_resize = cv2.resize(image, (want_size, want_size), cv2.INTER_LINEAR)
            i = idx % shape[1]
            j = idx // shape[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = img_resize
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * shape[0], w * shape[1]))
        for idx, image in enumerate(images):
            img_resize = cv2.resize(image, (want_size, want_size), cv2.INTER_LINEAR)
            i = idx % shape[1]
            j = idx // shape[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')
                         
                         
def imsave(imgs, _iter, dir_folder, name='imgs.png', shape=(5, 5), want_size=128, is_rgb=True):
    # RGB to BGR
    if is_rgb:
        r, g, b = imgs[:, :, :, 0], imgs[:, :, :, 1], imgs[:, :, :, 2]
        expands = [np.expand_dims(chanel, axis=-1) for chanel in [b, g, r]]
        imgs = np.concatenate(expands, axis=-1)
    # range [-1, 1]
    if imgs.mean() < 1:
        # convert to [0, 255]
        imgs = (255*(imgs + 1.0) / 2.0).astype(np.uint8)
        
    cv2.imwrite(os.path.join(dir_folder, '{}_iter_{}'.format(_iter, name)), merge(imgs, shape=shape, want_size=want_size))

def convert2int_tf(img):
    return tf.image.convert_image_dtype(255*(img + 1.0) / 2.0, dtype=tf.uint8)    