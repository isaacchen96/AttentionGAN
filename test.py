import os

from load_data import *
from build_model import *
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os

print(tf.__version__)
print(tf.test.is_gpu_available())
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
config = ConfigProto()
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

generator_E2N = build_generator()
generator_N2E = build_generator()
generator_E2N.load_weights('weight/gen_E2N_2')
generator_N2E.load_weights('weight/gen_N2E_36')

att_E2N_model = Model(generator_E2N.input, [generator_E2N.get_layer('conv2d_transpose_2').output,
                                            generator_E2N.get_layer('conv2d_transpose_3').output,
                                            generator_E2N.get_layer('conv2d_transpose_4').output])

att_N2E_model = Model(generator_N2E.input, [generator_N2E.get_layer('conv2d_transpose_11').output,
                                            generator_N2E.get_layer('conv2d_transpose_12').output,
                                            generator_N2E.get_layer('conv2d_transpose_13').output])

context_E2N_model = Model(generator_E2N.input, [generator_E2N.get_layer('conv2d_transpose_7').output,
                                                generator_E2N.get_layer('conv2d_transpose_8').output])

context_N2E_model = Model(generator_N2E.input, [generator_N2E.get_layer('conv2d_transpose_16').output,
                                                generator_N2E.get_layer('conv2d_transpose_17').output])
condition = 5
print('Attention GAN')
print('Now running test condition {}'.format(condition))

if condition == 0:


    # source_root = load_ck(train=True, emo='expression')
    # source = load_image(source_root[:10])
    path = '/home/pomelo96/Desktop/datasets/harry'
    img_name = os.listdir(path)
    roots = []
    [roots.append(path + '/' + img_name[i]) for i in range(len(img_name))]
    source = load_image(roots)

    att1, att2, att3 = att_E2N_model.predict(source)
    print(np.unique(att1))
    print(np.unique(att2))
    print(np.unique(att3))
    row, col = 3, source.shape[0]
    fig, axs = plt.subplots(row, col, sharex='col', sharey='row', figsize=(25, 25))
    plt.subplots_adjust(hspace=0.2)
    cnt = 0
    for j in range(col):
        axs[0, j].imshow(att1[cnt], cmap='gray')
        axs[0, j].axis('off')
        axs[1, j].imshow(att2[cnt], cmap='gray')
        axs[1, j].axis('off')
        axs[2, j].imshow(att3[cnt], cmap='gray')
        axs[2, j].axis('off')
        cnt += 1
    fig.savefig('att_E2N')

    att1, att2, att3 = att_N2E_model.predict(source)
    print(np.unique(att1))
    print(np.unique(att2))
    print(np.unique(att3))

    row, col = 3, source.shape[0]
    fig, axs = plt.subplots(row, col, sharex='col', sharey='row', figsize=(25, 25))
    plt.subplots_adjust(hspace=0.2)
    cnt = 0
    for j in range(col):
        axs[0, j].imshow(att1[cnt], cmap='gray')
        axs[0, j].axis('off')
        axs[1, j].imshow(att2[cnt], cmap='gray')
        axs[1, j].axis('off')
        axs[2, j].imshow(att3[cnt], cmap='gray')
        axs[2, j].axis('off')
        cnt += 1
    fig.savefig('att_N2E')

elif condition == 1:
    train = False
    if train: total_id = 46
    else: total_id = 33
    for emo in ['natural', 'expression']:
        if emo == 'natural':
            for i in range(total_id):
                source, _, id_ = load_ck_by_id(i, train, emo=emo)
                gen_img = generator_N2E.predict(source)
                att1, att2, att3 = att_N2E_model.predict(source)
                context1, context2 = context_N2E_model.predict(source)

                row, col = 7, source.shape[0]
                fig, axs = plt.subplots(row, col, sharex='col', sharey='row', figsize=(25, 25))
                plt.subplots_adjust(hspace=0.2)
                cnt = 0
                for j in range(col):
                    axs[0, j].imshow(source[cnt], cmap='gray')
                    axs[0, j].axis('off')
                    axs[1, j].imshow(att3[cnt], cmap='gray')
                    axs[1, j].axis('off')
                    axs[2, j].imshow(context1[cnt], cmap='gray')
                    axs[2, j].axis('off')
                    axs[3, j].imshow(att1[cnt], cmap='gray')
                    axs[3, j].axis('off')
                    axs[4, j].imshow(context2[cnt], cmap='gray')
                    axs[4, j].axis('off')
                    axs[5, j].imshow(att2[cnt], cmap='gray')
                    axs[5, j].axis('off')
                    axs[6, j].imshow(gen_img[cnt], cmap='gray')
                    axs[6, j].axis('off')
                fig.savefig('picture/cond1/N2E/{}'.format(id_))
                plt.close(fig)

        elif emo == 'expression':
            for i in range(total_id):
                source, _, id_ = load_ck_by_id(i, train, emo=emo)
                gen_img = generator_E2N.predict(source)
                att1, att2, att3 = att_E2N_model.predict(source)
                context1, context2 = context_E2N_model.predict(source)

                row, col = 7, source.shape[0]
                fig, axs = plt.subplots(row, col, sharex='col', sharey='row', figsize=(25, 25))
                plt.subplots_adjust(hspace=0.2)
                cnt = 0
                for j in range(col):
                    axs[0, j].imshow(source[cnt], cmap='gray')
                    axs[0, j].axis('off')
                    axs[1, j].imshow(att3[cnt], cmap='gray')
                    axs[1, j].axis('off')
                    axs[2, j].imshow(context1[cnt], cmap='gray')
                    axs[2, j].axis('off')
                    axs[3, j].imshow(att1[cnt], cmap='gray')
                    axs[3, j].axis('off')
                    axs[4, j].imshow(context2[cnt], cmap='gray')
                    axs[4, j].axis('off')
                    axs[5, j].imshow(att2[cnt], cmap='gray')
                    axs[5, j].axis('off')
                    axs[6, j].imshow(gen_img[cnt], cmap='gray')
                    axs[6, j].axis('off')
                fig.savefig('picture/cond1/E2N/{}'.format(id_))
                plt.close(fig)

elif condition == 2:
    bool_ = [True, False]
    path = 'picture/cond2/'
    for train in bool_:
        if train:
            total_id = 46
        else:
            total_id = 33
        for emo in ['natural', 'expression']:
            if emo == 'natural':
                for i in range(total_id):
                    source, img_name, id_ = load_ck_by_id(i, train, emo=emo)
                    gen_img = generator_N2E.predict(source)
                    cycle_img = generator_E2N.predict(gen_img)

                    gen_img = (255 * (0.5 * (gen_img + 1))).astype('uint8')
                    cycle_img = (255 * (0.5 * (cycle_img + 1))).astype('uint8')
                    gen_path = path + 'Gen expression'
                    cycle_path = path + 'Cycle natural'
                    if not os.path.exists(gen_path): os.makedirs(gen_path)
                    if not os.path.exists(cycle_path): os.makedirs(cycle_path)
                    gen_img_save_path = gen_path + '/' + id_
                    cycle_img_save_path = cycle_path + '/' + id_
                    if not os.path.exists(gen_img_save_path): os.makedirs(gen_img_save_path)
                    if not os.path.exists(cycle_img_save_path): os.makedirs(cycle_img_save_path)

                    for j in range(source.shape[0]):
                        img_save_name = img_name[j].split('/')[-1]
                        gen_img_save_path_ = gen_img_save_path + '/' + img_save_name
                        cycle_img_save_path_ = cycle_img_save_path + '/' + img_save_name
                        cv2.imwrite(gen_img_save_path_, gen_img[j])
                        cv2.imwrite(cycle_img_save_path_, cycle_img[j])

            elif emo == 'expression':
                for i in range(total_id):
                    source, img_name, id_ = load_ck_by_id(i, train, emo=emo)
                    gen_img = generator_E2N.predict(source)
                    cycle_img = generator_N2E.predict(gen_img)

                    gen_img = (255 * (0.5 * (gen_img + 1))).astype('uint8')
                    cycle_img = (255 * (0.5 * (cycle_img + 1))).astype('uint8')
                    gen_path = path + 'Gen natural'
                    cycle_path = path + 'Cycle expression'
                    if not os.path.exists(gen_path): os.makedirs(gen_path)
                    if not os.path.exists(cycle_path): os.makedirs(cycle_path)
                    gen_img_save_path = gen_path + '/' + id_
                    cycle_img_save_path = cycle_path + '/' + id_
                    if not os.path.exists(gen_img_save_path): os.makedirs(gen_img_save_path)
                    if not os.path.exists(cycle_img_save_path): os.makedirs(cycle_img_save_path)

                    for j in range(source.shape[0]):
                        img_save_name = img_name[j].split('/')[-1]
                        gen_img_save_path_ = gen_img_save_path + '/' + img_save_name
                        cycle_img_save_path_ = cycle_img_save_path + '/' + img_save_name
                        cv2.imwrite(gen_img_save_path_, gen_img[j])
                        cv2.imwrite(cycle_img_save_path_, cycle_img[j])

elif condition == 3:
    path = '/home/pomelo96/Desktop/datasets/harry'
    img_path = os.listdir(path)

    for name in img_path:
        source = cv2.imread(path + '/' + name)
        source = cv2.resize(source, (128, 128))
        source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        source = np.expand_dims(source, axis=-1)
        source = tf.reshape(source, (1, source.shape[0], source.shape[1], source.shape[2]))
        source = (source / 255) * 2 - 1

        gen_img = generator_N2E.predict(source)
        for i in range(gen_img.shape[0]):
            img = gen_img[i]
            img = (255 * (0.5 * (img + 1))).astype('uint8')
            save_path = 'picture/cond3/harry_AttentionGAN/' + name
            cv2.imwrite(save_path, img)

elif condition == 4: #CMU dataset
    path = '/home/pomelo96/Desktop/datasets/CMU/'
    emo_type = os.listdir(path)
    emo_type.sort()
    for emo in emo_type:
        emo_path = path + emo
        for t in ['train', 'test']:
            t_path = emo_path + '/' + t
            img_name_list = os.listdir(t_path)
            img_name_list.sort()
            img_path_list = []
            [img_path_list.append(t_path + '/' + img_name) for img_name in img_name_list]
            source = load_image(img_path_list)

            if emo == 'e':
                gen_img = generator_E2N.predict(source)
            elif emo == 'n':
                gen_img = generator_N2E.predict(source)

            for i in range(gen_img.shape[0]):
                img = gen_img[i]
                img = (255 * (0.5 * (img + 1))).astype('uint8')
                save_path = 'picture/cond4/source_' + emo + '/' + t + '/' + img_name_list[i]
                cv2.imwrite(save_path, img)

elif condition == 5:
    path = '/home/pomelo96/Desktop/datasets/Natural image'
    id_list = os.listdir(path)
    id_list.sort()
    for id_ in id_list:
        id_path = path + '/' + id_
        id_path = id_path + '/' + os.listdir(id_path)[0]
        img_name_list = os.listdir(id_path)
        img_name_list.sort()

        img_path_list = []
        [img_path_list.append(id_path + '/' + img_name) for img_name in img_name_list]

        source = load_image((img_path_list))
        gen_img = generator_N2E.predict(source)

        for i in range(gen_img.shape[0]):
            img = gen_img[i]
            img = (255 * (0.5 * (img + 1))).astype('uint8')
            save_path = 'picture/cond5/' + id_
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_img_path = save_path + '/' + img_name_list[i]
            cv2.imwrite(save_img_path, img)

elif condition == 6:
    path = '/home/pomelo96/Desktop/datasets/Expression image'
    id_list = os.listdir(path)
    id_list.sort()
    for id_ in id_list:
        id_path = path + '/' + id_
        id_path = id_path + '/' + os.listdir(id_path)[0]
        img_name_list = os.listdir(id_path)
        img_name_list.sort()

        img_path_list = []
        [img_path_list.append(id_path + '/' + img_name) for img_name in img_name_list]

        source = load_image((img_path_list))
        gen_img = generator_E2N.predict(source)
        for i in range(gen_img.shape[0]):
            img = gen_img[i]
            img = (255 * (0.5 * (img + 1))).astype('uint8')
            save_path = 'picture/cond6/' + id_
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_img_path = save_path + '/' + img_name_list[i]
            cv2.imwrite(save_img_path, img)



