from load_data import *
from build_model import *
from loss_function import *
from tensorflow.keras.optimizers import *
import time
import matplotlib.pyplot as plt


class AttentionGAN_celeA:
    def __init__(self):
        self.generator_N2E = build_generator()
        self.discriminator_E = build_discriminator()
        self.generator_E2N = build_generator()
        self.discriminator_N = build_discriminator()

        self.g_opt = Adam(2e-4)
        self.d_opt = Adam(1e-4)
        self.natural_train_roots, self.natural_train_label, \
        self.expression_train_roots, self.expression_train_label = build_train_CK_data(train=True)
        self.natural_test_roots, self.natural_test_label, \
        self.expression_test_roots, self.expression_test_label = build_train_CK_data(train=False)

    def cycle(self, natural_source, expression_source):
        gen_natural = self.generator_E2N.call(expression_source)
        gen_expression = self.generator_N2E.call(natural_source)
        loss = reconstruction_loss(natural_source, self.generator_E2N.call(gen_expression)) + \
            reconstruction_loss(expression_source, self.generator_N2E.call(gen_natural))
        return loss, gen_natural, gen_expression

    def gen_train_step(self, natural_source, natural_label, expression_source, expression_label, train=True):
        natural_source = tf.cast(natural_source, dtype='float32')
        expression_source = tf.cast(expression_source, dtype='float32')
        natural_label = tf.one_hot(natural_label, depth=2)
        expression_label = tf.one_hot(expression_label, depth=2)
        with tf.GradientTape() as tape:
            loss_cycle, gen_natural, gen_expression = self.cycle(natural_source, expression_source)
            v_gen_N, c_gen_N = self.discriminator_N.call(gen_natural)
            v_gen_E, c_gen_E = self.discriminator_E.call(gen_expression)
            loss_adv_N = adversarial_loss(v_gen_N, True)
            loss_adv_E = adversarial_loss(v_gen_E, True)
            loss_cls_N = classify_loss(natural_label, c_gen_N)
            loss_cls_E = classify_loss(expression_label, c_gen_E)
            loss_g = 10 * loss_cycle + loss_adv_N + loss_adv_E + 10 * loss_cls_N + 10 * loss_cls_E
        if train:
            trainable_weights = self.generator_N2E.trainable_weights + self.generator_E2N.trainable_weights
            grads = tape.gradient(loss_g, trainable_weights)
            self.g_opt.apply_gradients(zip(grads, trainable_weights))
        return 10 * loss_cycle, [loss_adv_N, 10 * loss_cls_N], [loss_adv_E, 10 * loss_cls_E]

    def dis_train_step(self, natural_source, natural_label, expression_source, expression_label, train=True):
        natural_source = tf.cast(natural_source, dtype='float32')
        expression_source = tf.cast(expression_source, dtype='float32')
        natural_label = tf.one_hot(natural_label, depth=2)
        expression_label = tf.one_hot(expression_label, depth=2)
        with tf.GradientTape() as tape:
            gen_natural = self.generator_E2N.call(expression_source)
            gen_expression = self.generator_N2E.call(natural_source)
            v_gen_N, c_gen_N = self.discriminator_N.call(gen_natural)
            v_gen_E, c_gen_E = self.discriminator_E.call(gen_expression)
            v_real_N, c_real_N = self.discriminator_N.call(natural_source)
            v_real_E, c_real_E = self.discriminator_E.call(expression_source)
            loss_adv_N = 0.5 * (adversarial_loss(v_gen_N, False) + adversarial_loss(v_real_N, True))
            loss_adv_E = 0.5 * (adversarial_loss(v_gen_E, False) + adversarial_loss(v_real_E, True))
            loss_cls_N = classify_loss(natural_label, c_real_N)
            loss_cls_E = classify_loss(expression_label, c_real_E)
            loss_d = loss_adv_N + loss_adv_E + 10 * loss_cls_N + 10 * loss_cls_E
        if train:
            trainable_weights = self.discriminator_N.trainable_weights + self.discriminator_E.trainable_weights
            grads = tape.gradient(loss_d, trainable_weights)
            self.d_opt.apply_gradients(zip(grads, trainable_weights))
        return [loss_adv_N, 10 * loss_cls_N], [loss_adv_E, 10 * loss_cls_E]

    def train(self, epochs=20, interval=1, batch_size=16, batch_num=36):
        tr_L_G_cycle_avg = []
        tr_L_G_N_avg, tr_L_G_adv_N_avg, tr_L_G_cls_N_avg = [], [], []
        tr_L_G_E_avg, tr_L_G_adv_E_avg, tr_L_G_cls_E_avg = [], [], []
        tr_L_D_N_avg, tr_L_D_adv_N_avg, tr_L_D_cls_N_avg = [], [], []
        tr_L_D_E_avg, tr_L_D_adv_E_avg, tr_L_D_cls_E_avg = [], [], []
        te_L_G_cycle_avg = []
        te_L_G_N_avg, te_L_G_E_avg = [], []
        te_L_D_N_avg, te_L_D_E_avg = [], []

        start = time.time()
        for epoch in range(epochs):
            ep_start = time.time()
            tr_L_G_cycle = []
            tr_L_G_N, tr_L_G_adv_N, tr_L_G_cls_N = [], [], []
            tr_L_G_E, tr_L_G_adv_E, tr_L_G_cls_E = [], [], []
            tr_L_D_N, tr_L_D_adv_N, tr_L_D_cls_N = [], [], []
            tr_L_D_E, tr_L_D_adv_E, tr_L_D_cls_E = [], [], []
            te_L_G_cycle = []
            te_L_G_N, te_L_G_E = [], []
            te_L_D_N, te_L_D_E = [], []
            for b in range(batch_num):
                natural_source = load_image(get_batch_data(self.natural_train_roots, b, batch_size))
                expression_source = load_image(get_batch_data(self.expression_train_roots, b, batch_size))
                natural_label = get_batch_data(self.natural_train_label, b, batch_size)
                expression_label = get_batch_data(self.expression_train_label, b, batch_size)
                b_test = random.randint(0, 15)
                natural_source_test= load_image(get_batch_data(self.natural_test_roots, b_test, batch_size))
                expression_source_test = load_image(get_batch_data(self.expression_test_roots, b_test, batch_size))

                natural_label_test = get_batch_data(self.natural_test_label, b_test, batch_size)
                expression_label_test = get_batch_data(self.expression_test_label, b_test, batch_size)
                for i in range(3):
                    loss_cycle, [loss_adv_g_N, loss_cls_g_N], [loss_adv_g_E, loss_cls_g_E] =  \
                        self.gen_train_step(natural_source, natural_label, expression_source, expression_label, train=True)
                tr_L_G_cycle.append(loss_cycle)
                tr_L_G_adv_N.append(loss_adv_g_N)
                tr_L_G_cls_N.append(loss_cls_g_N)
                tr_L_G_adv_E.append(loss_adv_g_E)
                tr_L_G_cls_E.append(loss_cls_g_E)
                tr_L_G_N.append(loss_cycle + loss_adv_g_N + loss_cls_g_N)
                tr_L_G_E.append(loss_cycle + loss_adv_g_E + loss_cls_g_E)
                loss_cycle, [loss_adv_g_N, loss_cls_g_N], [loss_adv_g_E, loss_cls_g_E] = \
                    self.gen_train_step(natural_source_test, natural_label_test,
                                        expression_source_test, expression_label_test, train=False)
                te_L_G_cycle.append(loss_cycle)
                te_L_G_N.append(loss_cycle + loss_adv_g_N + loss_cls_g_N)
                te_L_G_E.append(loss_cycle + loss_adv_g_E + loss_cls_g_E)
                [loss_adv_d_N, loss_cls_d_N], [loss_adv_d_E, loss_cls_d_E] = \
                    self.dis_train_step(natural_source, natural_label, expression_source, expression_label, train=True)
                tr_L_D_adv_N.append(loss_adv_d_N)
                tr_L_D_cls_N.append(loss_cls_d_N)
                tr_L_D_adv_E.append(loss_adv_d_E)
                tr_L_D_cls_E.append(loss_cls_d_E)
                tr_L_D_N.append(loss_adv_d_N + loss_cls_d_N)
                tr_L_D_E.append(loss_adv_d_E + loss_cls_d_E)
                [loss_adv_d_N, loss_cls_d_N], [loss_adv_d_E, loss_cls_d_E] = \
                    self.dis_train_step(natural_source_test, natural_label_test,
                                        expression_source_test, expression_label_test, train=False)
                te_L_D_N.append(loss_adv_d_N + loss_cls_d_N)
                te_L_D_E.append(loss_adv_d_E + loss_cls_d_E)

            tr_L_G_cycle_avg.append(np.mean(tr_L_G_cycle))
            tr_L_G_N_avg.append(np.mean(tr_L_G_N))
            tr_L_G_adv_N_avg.append(np.mean(tr_L_G_adv_N))
            tr_L_G_cls_N_avg.append(np.mean(tr_L_G_cls_N))
            tr_L_G_E_avg.append(np.mean(tr_L_G_E))
            tr_L_G_adv_E_avg.append(np.mean(tr_L_G_adv_E))
            tr_L_G_cls_E_avg.append(np.mean(tr_L_G_cls_E))
            tr_L_D_N_avg.append(np.mean(tr_L_D_N))
            tr_L_D_adv_N_avg.append(np.mean(tr_L_D_adv_N))
            tr_L_D_cls_N_avg.append(np.mean(tr_L_D_cls_N))
            tr_L_D_E_avg.append(np.mean(tr_L_D_E))
            tr_L_D_adv_E_avg.append(np.mean(tr_L_D_adv_E))
            tr_L_D_cls_E_avg.append(np.mean(tr_L_D_cls_E))
            te_L_G_cycle_avg.append(np.mean(te_L_G_cycle))
            te_L_G_N_avg.append(np.mean(te_L_G_N))
            te_L_G_E_avg.append(np.mean(te_L_G_E))
            te_L_D_N_avg.append(np.mean(te_L_D_N))
            te_L_D_E_avg.append(np.mean(te_L_D_E))

            t_pass = time.time() - start
            m_pass, s_pass = divmod(t_pass, 60)
            h_pass, m_pass = divmod(m_pass, 60)
            print('\nTime for pass  {:<4d}: {:<2d} hour {:<3d} min {:<4.3f} sec'.format(epoch + 1, int(h_pass),
                                                                                        int(m_pass), s_pass))
            print('Time for epoch {:<4d}: {:6.3f} sec'.format(epoch + 1, time.time() - ep_start))
            print('Train Loss Cycle loss           :  {:8.5f}'.format(tr_L_G_cycle_avg[-1]))
            print('Train Loss Generator_N2E        :  {:8.5f}'.format(tr_L_G_E_avg[-1]))
            print('Train Loss Discriminator_E      :  {:8.5f}'.format(tr_L_D_E_avg[-1]))
            print('Train Loss Generator_E2N        :  {:8.5f}'.format(tr_L_G_N_avg[-1]))
            print('Train Loss Discriminator_N      :  {:8.5f}'.format(tr_L_D_N_avg[-1]))

            self.sample_image_pretrain(epoch)
            if (epoch % interval == 0 or epoch + 1 == epochs):
                if (te_L_G_N_avg[-1] <= np.mean(te_L_G_N_avg)):
                    self.generator_E2N.save_weights('weight/gen_E2N_{}'.format(epoch+1))
                    self.discriminator_N.save_weights('weight/dis_N_{}'.format(epoch+1))
                if (te_L_G_E_avg[-1] <= np.mean(te_L_G_E_avg)):
                    self.generator_N2E.save_weights('weight/gen_N2E_{}'.format(epoch+1))
                    self.discriminator_E.save_weights('weight/dis_E_{}'.format(epoch+1))

        return tr_L_G_cycle_avg, te_L_G_cycle_avg, \
                [tr_L_G_N_avg, tr_L_G_adv_N_avg, tr_L_G_cls_N_avg], \
                [tr_L_G_E_avg, tr_L_G_adv_E_avg, tr_L_G_cls_E_avg], \
                [tr_L_D_N_avg, tr_L_D_adv_N_avg, tr_L_D_cls_N_avg], \
                [tr_L_D_E_avg, tr_L_D_adv_E_avg, tr_L_D_cls_E_avg], \
                [te_L_G_N_avg, te_L_G_E_avg], \
                [te_L_D_N_avg, te_L_D_E_avg]

    def sample_image_pretrain(self, epoch, path='picture/'):
        natural_source_train = load_image(get_batch_data(self.natural_train_roots, 0, 5))
        natural_source_test = load_image(get_batch_data(self.natural_test_roots, 0, 5))
        natural_source_sampling = tf.concat([natural_source_train, natural_source_test], axis=0)

        expression_source_train = load_image(get_batch_data(self.expression_train_roots, 0, 5))
        expression_source_test = load_image(get_batch_data(self.expression_test_roots, 0, 5))
        expression_source_sampling = tf.concat([expression_source_train, expression_source_test], axis=0)

        gen_natural = self.generator_E2N.predict(expression_source_sampling)
        gen_expression = self.generator_N2E.predict(natural_source_sampling)
        cycle_natural = self.generator_E2N.predict(gen_expression)
        cycle_expression = self.generator_N2E.predict(gen_natural)

        gen_natural = 0.5 * (gen_natural + 1)
        gen_expression = 0.5 * (gen_expression + 1)
        cycle_natural = 0.5 * (cycle_natural + 1)
        cycle_expression = 0.5 * (cycle_expression + 1)

        source = [natural_source_sampling, expression_source_sampling]
        gen = [gen_expression, gen_natural]
        cycle = [cycle_natural, cycle_expression]
        trans = ['NEN', 'ENE']

        for s, g, c, name in zip(source, gen, cycle, trans):
            row, col = 3, 10
            fig, axs = plt.subplots(row, col, sharex='col', sharey='row', figsize=(25, 25))
            plt.subplots_adjust(hspace=0.2)
            cnt = 0
            for j in range(col):
                axs[0, j].imshow(s[cnt], cmap='gray')
                axs[0, j].axis('off')
                axs[1, j].imshow(g[cnt], cmap='gray')
                axs[1, j].axis('off')
                axs[2, j].imshow(c[cnt], cmap='gray')
                axs[2, j].axis('off')
                cnt += 1
            fig.savefig(path + name + '_{}.png'.format(epoch + 1))
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

    attention_gan = AttentionGAN_celeA()
    attention_gan.generator_N2E.load_weights('pretrain_weight/ck_generator_weights_5')
    attention_gan.generator_E2N.load_weights('pretrain_weight/ck_generator_weights_5')
    attention_gan.discriminator_N.load_weights('pretrain_weight/ck_discriminator_weights_5')
    attention_gan.discriminator_E.load_weights('pretrain_weight/ck_discriminator_weights_5')
    tr_L_G_cycle_avg, te_L_G_cycle_avg, \
        [tr_L_G_N_avg, tr_L_G_adv_N_avg, tr_L_G_cls_N_avg], \
        [tr_L_G_E_avg, tr_L_G_adv_E_avg, tr_L_G_cls_E_avg], \
        [tr_L_D_N_avg, tr_L_D_adv_N_avg, tr_L_D_cls_N_avg], \
        [tr_L_D_E_avg, tr_L_D_adv_E_avg, tr_L_D_cls_E_avg], \
        [te_L_G_N_avg, te_L_G_E_avg], \
        [te_L_D_N_avg, te_L_D_E_avg] = attention_gan.train(epochs=20, interval=1)

    plt.plot(tr_L_G_cycle_avg)
    plt.plot(te_L_G_cycle_avg)
    plt.legend(['Train', 'Test'])
    plt.title('Generator cycle loss')
    plt.savefig('picture/_Generator cycle loss.jpg')
    plt.close()

    plt.plot(tr_L_G_N_avg)
    plt.plot(te_L_G_N_avg)
    plt.legend(['Train', 'Test'])
    plt.title('Generator E2N loss')
    plt.savefig('picture/_Generator E2N loss.jpg')
    plt.close()

    plt.plot(tr_L_G_E_avg)
    plt.plot(te_L_G_E_avg)
    plt.legend(['Train', 'Test'])
    plt.title('Generator N2E loss')
    plt.savefig('picture/_Generator N2E loss.jpg')
    plt.close()

    plt.plot(tr_L_D_N_avg)
    plt.plot(te_L_D_N_avg)
    plt.legend(['Train', 'Test'])
    plt.title('Discriminator N loss')
    plt.savefig('picture/_Discriminator N loss.jpg')
    plt.close()

    plt.plot(tr_L_D_E_avg)
    plt.plot(te_L_D_E_avg)
    plt.legend(['Train', 'Test'])
    plt.title('Discriminator E loss')
    plt.savefig('picture/_Discriminator E loss.jpg')
    plt.close()

    plt.plot(tr_L_G_adv_E_avg)
    plt.plot(tr_L_D_adv_E_avg)
    plt.legend(['Generator', 'Discriminator'])
    plt.title('Adversarial N2E loss')
    plt.savefig('picture/_Adversarial N2E loss.jpg')
    plt.close()

    plt.plot(tr_L_G_adv_N_avg)
    plt.plot(tr_L_D_adv_N_avg)
    plt.legend(['Generator', 'Discriminator'])
    plt.title('Adversarial E2N loss')
    plt.savefig('picture/_Adversarial E2N loss.jpg')
    plt.close()

    plt.plot(tr_L_G_cls_N_avg)
    plt.title('Generator E2N classify loss')
    plt.savefig('picture/_Generator E2N classify loss.jpg')
    plt.close()

    plt.plot(tr_L_G_cls_E_avg)
    plt.title('Generator N2E classify loss')
    plt.savefig('picture/_Generator N2E classify loss.jpg')
    plt.close()

    plt.plot(tr_L_D_cls_N_avg)
    plt.title('Discriminator E2N classify loss')
    plt.savefig('picture/_Discriminator E2N classify loss.jpg')
    plt.close()

    plt.plot(tr_L_D_cls_E_avg)
    plt.title('Discriminator N2E classify loss')
    plt.savefig('picture/_Discriminator N2E classify loss.jpg')
    plt.close()
