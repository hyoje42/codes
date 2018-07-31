import numpy as np
import time

def split_tr_val(img_files, labels, val_rate=0.1):
    np.random.seed(777)
    
    class_num = {}
    for cls in labels:
        if cls not in class_num.keys():
            class_num[cls] = 0
        class_num[cls] += 1
    
    _shape = img_files.shape[1:]
    tr_files = np.array([]).reshape([0] + [i for i in _shape])
    val_files = np.array([]).reshape([0] + [i for i in _shape])
    tr_labels = np.array([]).reshape(0)
    val_labels = np.array([]).reshape(0)

    for key in class_num.keys():
        one_cls_idxs = np.where(labels == key)[0]

        one_cls_files = img_files[one_cls_idxs]
        one_cls_labeles = labels[one_cls_idxs]

        val_num = np.floor(class_num[key] * val_rate).astype(np.int)
        
        if val_num == 0:
            if len(one_cls_files) > 1:
                val_num = 1
            else:
                val_num = 0
            
        tr_num = len(one_cls_files) - val_num

        rd_idxs = np.random.permutation(range(len(one_cls_files)))

        val_files = np.concatenate((val_files, one_cls_files[rd_idxs][:val_num]))
        tr_files = np.concatenate((tr_files, one_cls_files[rd_idxs][val_num:]))

        tr_labels = np.concatenate((tr_labels, 
                                    np.full(shape=tr_num, fill_value=key))).astype(np.int)
        val_labels = np.concatenate((val_labels, 
                                     np.full(shape=val_num, fill_value=key))).astype(np.int)
        
        print('{} : {} --> Tr : {}, Val : {}'.format(key, class_num[key], tr_num, val_num))
    
    print('Tr : {}, {} / Val : {}, {}'.format(tr_files.shape, tr_labels.shape, 
                                             val_files.shape, val_labels.shape))
    np.random.seed(int(time.time()))
    
    return tr_files, tr_labels, val_files, val_labels
