from load_data import *
from build_model import *
from loss_function import *
from tensorflow.keras.optimizers import *
import time
import matplotlib.pyplot as plt


class AttentionGAN_ck:
    def __init__(self):
        self.generator = build_generator()
        self.discriminator = build_discriminator()
        self.g_opt = Adam(1e-4)
        self.d_opt = Adam(1e-4)
        self.source_roots, self.target_roots = build_pretrain_CK_data(train=True, direction='N2E')
        self.test_source_roots, self.test_target_roots = build_pretrain_CK_data(train=False, direction='N2E')

    def gen_train_step(self, source, target, train=True):
        source = tf.cast(source, dtype='float32')
        target = tf.cast(target, dtype='float32')
        label = [1] * source.shape[0]
        label = tf.one_hot(label, depth=2)
        with tf.GradientTape() as tape:
            gen_img = self.generator.call(source)
            v_gen, c_gen = self.discriminator.call(gen_img)
            loss_img = reconstruction_loss(target, gen_img)
            loss_adv = adversarial_loss(v_gen, True)
            loss_cls = classify_loss(label, c_gen)
            loss_g = 10 * loss_img + loss_adv + 10 * loss_cls
        if train:
            grads = tape.gradient(loss_g, self.generator.trainable_weights)
            self.g_opt.apply_gradients(zip(grads, self.generator.trainable_weights))
            return loss_g
        else:
            return loss_g, 10 * loss_img, loss_adv, 10 * loss_cls

    def dis_train_step(self, source, target, train=True):
        source = tf.cast(source, dtype='float32')
        target = tf.cast(target, dtype='float32')
        natural_label = [0] * source.shape[0]
        natural_label = tf.one_hot(natural_label, depth=2)
        expression_label = [1] * source.shape[0]
        expression_label = tf.one_hot(expression_label, depth=2)
        with tf.GradientTape() as tape:
            gen_img = self.generator.call(source)
            v_gen, c_gen = self.discriminator.call(gen_img)
            v_real_N, c_real_N = self.discriminator.call(source)
            v_real_E, c_real_E = self.discriminator.call(target)
            loss_adv = 0.5 * (adversarial_loss(v_gen, False) + adversarial_loss(v_real_E, True))
            loss_cls = 0.5 * (classify_loss(natural_label, c_real_N) + classify_loss(expression_label, c_real_E))
            loss_d = loss_adv + 10 * loss_cls
        if train:
            grads = tape.gradient(loss_d, self.discriminator.trainable_weights)
            self.d_opt.apply_gradients(zip(grads, self.discriminator.trainable_weights))
            return loss_d
        else:
            return loss_d, loss_adv, 10 * loss_cls

    def train(self, epochs=20, interval=1, batch_size=16, batch_num=209):
        tr_L_G_avg = []
        tr_L_D_avg = []
        te_L_G_avg = []
        te_L_G_img_avg = []
        te_L_G_adv_avg = []
        te_L_G_cls_avg = []
        te_L_D_avg = []
        te_L_D_adv_avg = []
        te_L_D_cls_avg = []

        start = time.time()
        for epoch in range(epochs):
            ep_start = time.time()
            tr_L_G = []
            tr_L_D = []
            te_L_G = []
            te_L_G_img = []
            te_L_G_adv = []
            te_L_G_cls = []
            te_L_D = []
            te_L_D_adv = []
            te_L_D_cls = []
            for b in range(batch_num):
                source = load_image(get_batch_data(self.source_roots, b, batch_size))
                target = load_image(get_batch_data(self.target_roots, b, batch_size))
                b_test = random.randint(0, 95)
                source_test = load_image(get_batch_data(self.test_source_roots, b_test, batch_size))
                target_test = load_image(get_batch_data(self.test_target_roots, b_test, batch_size))
                for i in range(2):
                    loss_d = self.dis_train_step(source, target)
                loss_d_test, loss_adv_d, loss_cls_d = self.dis_train_step(source_test, target_test, train=False)

                loss_g = self.gen_train_step(source, target)
                loss_g_test, loss_img, loss_adv_g, loss_cls_g = self.gen_train_step(source_test, target_test, train=False)
                tr_L_G.append(loss_g)
                te_L_G_img.append(loss_img)
                te_L_G_adv.append(loss_adv_g)
                te_L_G_cls.append(loss_cls_g)
                tr_L_D.append(loss_d)
                te_L_D_adv.append(loss_adv_d)
                te_L_D_cls.append(loss_cls_d)
                te_L_G.append(loss_g_test)
                te_L_D.append(loss_d_test)

            tr_L_G_avg.append(np.mean(tr_L_G))
            te_L_G_img_avg.append(np.mean(te_L_G_img))
            te_L_G_adv_avg.append(np.mean(te_L_G_adv))
            te_L_G_cls_avg.append(np.mean(te_L_G_cls))
            tr_L_D_avg.append(np.mean(tr_L_D))
            te_L_D_adv_avg.append(np.mean(te_L_D_adv))
            te_L_D_cls_avg.append(np.mean(te_L_D_cls))
            te_L_G_avg.append(np.mean(te_L_G))
            te_L_D_avg.append(np.mean(te_L_D))

            t_pass = time.time() - start
            m_pass, s_pass = divmod(t_pass, 60)
            h_pass, m_pass = divmod(m_pass, 60)
            print('\nTime for pass  {:<4d}: {:<2d} hour {:<3d} min {:<4.3f} sec'.format(epoch + 1, int(h_pass),
                                                                                        int(m_pass), s_pass))
            print('Time for epoch {:<4d}: {:6.3f} sec'.format(epoch + 1, time.time() - ep_start))
            print('Train Loss Generator     :  {:8.5f}'.format(tr_L_G_avg[-1]))
            print('Test Loss Gen_image     :  {:8.5f}'.format(te_L_G_img_avg[-1]))
            print('Test Loss Gen_adv       :  {:8.5f}'.format(te_L_G_adv_avg[-1]))
            print('Test Loss Gen_cls       :  {:8.5f}'.format(te_L_G_cls_avg[-1]))
            print('Train Loss Discriminator :  {:8.5f}'.format(tr_L_D_avg[-1]))
            print('Test Loss Dis_adv       :  {:8.5f}'.format(te_L_D_adv_avg[-1]))
            print('Test Loss Dis_cls       :  {:8.5f}'.format(te_L_D_cls_avg[-1]))
            print('Test Loss Generator      :  {:8.5f}'.format(te_L_G_avg[-1]))
            print('Test Loss Discriminator  :  {:8.5f}'.format(te_L_D_avg[-1]))
            self.sample_image_pretrain(epoch)
            if (epoch % interval == 0 or epoch + 1 == epochs) and (te_L_G_avg[-1] <= np.mean(te_L_G_avg)):
                self.generator.save_weights('3pretrain_weight/ck_generator_N2E_weights_{}'.format(epoch + 1))
                self.discriminator.save_weights('3pretrain_weight/ck_discriminator_N2E_weights_{}'.format(epoch + 1))

        return [te_L_G_avg, te_L_G_img_avg, te_L_G_adv_avg, te_L_G_cls_avg], [te_L_D_avg, te_L_D_adv_avg, te_L_D_cls_avg], \
               [tr_L_G_avg, tr_L_D_avg]

    def sample_image_pretrain(self, epoch, path='3pretrain_picture/ck_pretrain_N2E'):
        source_train = load_image(get_batch_data(self.source_roots, 0, 5))
        source_test = load_image(get_batch_data(self.test_source_roots, 0, 5))
        source_sampling = tf.concat([source_train, source_test], axis=0)

        gen_img = self.generator.call(source_sampling)
        source_sampling = 0.5 * (source_sampling + 1)
        gen_img = 0.5 * (gen_img + 1)

        r, c = 2, 10
        fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(25, 25))
        plt.subplots_adjust(hspace=0.2)
        cnt = 0
        for j in range(c):
            axs[0, j].imshow(source_sampling[cnt], cmap='gray')
            axs[0, j].axis('off')
            axs[1, j].imshow(gen_img[cnt], cmap='gray')
            axs[1, j].axis('off')
            cnt += 1
        fig.savefig(path + '_{}.png'.format(epoch + 1))
        plt.close()


if __name__ == '__main__':
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

    attention_gan = AttentionGAN_ck()
    attention_gan.generator.load_weights('Xpretrain_weight/ck_generator_N2E_weights_3')
    attention_gan.discriminator.load_weights('Xpretrain_weight/ck_discriminator_N2E_weights_2')

    [te_L_G_avg, te_L_G_img_avg, te_L_G_adv_avg, te_L_G_cls_avg], [te_L_D_avg, te_L_D_adv_avg, te_L_D_cls_avg], \
    [tr_L_G_avg, tr_L_D_avg] = attention_gan.train(epochs=10, interval=1)

    plt.plot(tr_L_G_avg)
    plt.plot(te_L_G_avg)
    plt.legend(['Train', 'Test'])
    plt.title('CK Generator N2E pretrain loss')
    plt.savefig('3pretrain_picture/ck_Generator N2E pretrain loss.jpg')
    plt.close()

    plt.plot(tr_L_D_avg)
    plt.plot(te_L_D_avg)
    plt.legend(['Train', 'Test'])
    plt.title('CK Discriminator N2E pretrain loss')
    plt.savefig('3pretrain_picture/ck_Discriminator N2E pretrain loss.jpg')
    plt.close()

    plt.plot(te_L_G_adv_avg)
    plt.plot(te_L_D_adv_avg)
    plt.legend(['Generator', 'Discriminator'])
    plt.title('CK Adversarial N2E pretrain loss')
    plt.savefig('3pretrain_picture/ck_Adversarial N2E loss.jpg')
    plt.close()

    plt.plot(te_L_G_cls_avg)
    plt.title('CK Generator N2E pretrain classify loss')
    plt.savefig('3pretrain_picture/ck_Generator N2E Image loss.jpg')
    plt.close()

    plt.plot(te_L_D_cls_avg)
    plt.title('CK Discriminator N2E pretrain classify loss')
    plt.savefig('3pretrain_picture/ck_Discriminator N2E classify loss.jpg')
    plt.close()

    plt.plot(te_L_G_img_avg)
    plt.title('CK Generator N2E pretrain image loss')
    plt.savefig('3pretrain_picture/ck_Generator N2E Image loss.jpg')
    plt.close()
