import os
import cv2
import numpy as np
import random

def load_image(roots):
    dataset = []
    for root in roots:
        img = cv2.imread(root)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1)
        dataset.append(img)
    return (np.array(dataset).astype('float32')) / 127.5 - 1


def load_ck(train=True, emo='natural'):
    dataset = []
    path = '/home/pomelo96/Desktop/datasets/classifier_alignment_CK'
    if train:
        path = path + '/train'
    else:
        path = path + '/test'

    if emo == 'natural':
        path = path + '/Natural image'
    elif emo == 'expression':
        path = path + '/Expression image'

    id_file = os.listdir(path)
    for id_ in id_file:
        file_path = path + '/' + id_
        file_path = file_path + '/' + os.listdir(file_path)[0]
        img_name = os.listdir(file_path)
        for name in img_name:
            dataset.append(file_path + '/' + name)
    random.shuffle(dataset)
    return dataset

def load_ck_by_id(i, train, emo):
    if train:
        path = '/home/pomelo96/Desktop/datasets/classifier_alignment_CK/train'
    else:
        path = '/home/pomelo96/Desktop/datasets/classifier_alignment_CK/test'

    if emo == 'natural':
        path = path + '/Natural image'
    elif emo == 'expression':
        path = path + '/Expression image'

    path_dataset = []
    id_file = os.listdir(path)
    id_file.sort()
    id_ = id_file[i]
    id_img_file = path + '/' + id_
    id_img_file = id_img_file + '/' + os.listdir(id_img_file)[0]
    id_img_name = os.listdir(id_img_file)
    for name in id_img_name:
        img_path = id_img_file + '/' + name
        path_dataset.append(img_path)
    img_dataset = load_image(path_dataset)
    return img_dataset, path_dataset, id_

def load_celebA():
    path = '/home/pomelo96/Desktop/datasets/celebA'
    dataset = []
    img_roots = os.listdir(path)
    img_roots.sort()
    for roots in img_roots:
        dataset.append(path + '/' + roots)
    train_dataset = dataset[:32000]
    test_dataset = dataset[-3200:]
    return train_dataset, test_dataset


def get_batch_data(data,batch_idx,batch_size):
    range_min = batch_idx * batch_size
    range_max = (batch_idx + 1) * batch_size
    if range_max > len(data):
        range_max = len(data)
    index = list(range(range_min, range_max))
    temp_data = [data[idx] for idx in index]
    return temp_data


def build_pretrain_CK_data(train=True, direction='N2E'):
    if train: id_total = 46
    else: id_total = 33
    source = []
    target = []

    for i in range(id_total):
        if direction == 'N2E':
            _, source_root, _ = load_ck_by_id(i, train=train, emo='natural')
            _, target_root, _ = load_ck_by_id(i, train=train, emo='expression')
        elif direction =='E2N':
            _, source_root, _ = load_ck_by_id(i, train=train, emo='expression')
            _, target_root, _ = load_ck_by_id(i, train=train, emo='natural')

        for s in source_root:
            for t in target_root:
                source.append(s)
                target.append(t)
    temp = list(zip(source, target))
    random.shuffle(temp)
    source, target = zip(*temp)

    return source, target


def build_train_CK_data(train=True):
    natural_roots = load_ck(train=train, emo='natural')
    expression_roots = load_ck(train=train, emo='expression')

    x = len(expression_roots) // len(natural_roots) + 1
    natural_roots = natural_roots * x
    natural_roots = random.sample(natural_roots, len(expression_roots))
    natural_label = [0] * len(natural_roots)
    expression_label = [1] * len(expression_roots)
    return natural_roots, natural_label, expression_roots, expression_label


if __name__ == '__main__':
    source, target = build_pretrain_CK_data(train=True, direction='E2N')
    for i in range(10):
        print(source[i])
        print(target[i])
        print(' ')
    print(len(source), len(target))

