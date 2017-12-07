#! /usr/bin/python
# -*- coding: utf8 -*-
import tensorflow as tf
import tensorlayer as tl
import model, time, os
import numpy as np

""" Conditional Image Synthesis With Auxiliary Classifier GANs """

save_dir = "samples/experiment2"
tl.files.exists_or_mkdir(save_dir)
checkpoint_dir = "checkpoint/experiment2"
tl.files.exists_or_mkdir(checkpoint_dir)
save_dir_d = os.path.join(checkpoint_dir, 'd.npz')
save_dir_g = os.path.join(checkpoint_dir, 'g.npz')

def _data_aug_fn(im):
    im = tl.prepro.flip_axis(im, axis=1, is_random=True)
    im = im / 127.5 - 1
    return im

def main():
    X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3))

    batch_size = 64
    z_dim = 128
    t_image = tf.placeholder(tf.float32, [None, 32, 32, 3], 'image')
    t_z     = tf.placeholder(tf.float32, [None, z_dim]    , 'noise')
    t_class = tf.placeholder(tf.int64  , [None, ]        , 'class')

    net_g = model.G(t_z, t_class, is_train=True, reuse=False, batch_size=batch_size)
    net_d, d_fake, d_fake_class = model.D(net_g.outputs, is_train=True, reuse=False)
    net_d, d_real, d_real_class = model.D(t_image      , is_train=True, reuse=True)

    # for testing
    net_g_test = model.G(t_z, t_class, is_train=False, reuse=True, batch_size=100)

    # net_g.print_layers()          # show network info
    # net_d.print_params(False)
    # exit()

    # class loss
    ce_real = tl.cost.cross_entropy(d_real_class, t_class, name='d_real_class')
    ce_fake = tl.cost.cross_entropy(d_fake_class, t_class, name='d_fake_class')
    q_loss = ce_real + ce_fake

    # DC-GAN
    # d_loss1 = tl.cost.sigmoid_cross_entropy(d_real, tf.ones_like(d_real), name='d_real')
    # d_loss2 = tl.cost.sigmoid_cross_entropy(d_fake, tf.zeros_like(d_fake), name='d_fake')
    # g_loss1 = tl.cost.sigmoid_cross_entropy(d_fake, tf.ones_like(d_fake), name='g_fake')
    # LS-GAN
    d_loss1 = tl.cost.mean_squared_error(d_real, tf.ones_like(d_real), is_mean=True, name='d_real')
    d_loss2 = tl.cost.mean_squared_error(d_fake, tf.zeros_like(d_fake), is_mean=True, name='d_fake')
    g_loss1 = tl.cost.mean_squared_error(d_fake, tf.ones_like(d_fake), is_mean=True, name='g_fake')

    d_loss = d_loss1 + d_loss2 + ce_real + ce_fake
    g_loss = g_loss1 + ce_fake

    lr = 0.0002
    lr_decay = 0.5
    n_epoch = 50
    decay_every = 10
    beta1 = 0.5

    d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
    g_vars = tl.layers.get_variables_with_name('generator', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)

    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)

    # tl.files.load_and_assign_npz(sess, save_dir_d, net_d)     # (optional) load trained model
    # tl.files.load_and_assign_npz(sess, save_dir_d, net_g)

    print_freq = 5
    n_step_epoch = int(len(y_train) / batch_size)
    sample_z = np.random.normal(loc=0.0, scale=1.0, size=(100, z_dim))
    sample_c = [i for i in range(10)] * 10      #[[i]*10 for i in range(10)]
    # print(len(sample_c), sample_z.shape)
    # exit()
    for epoch in range(n_epoch):

        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr * new_lr_decay))
            log = " ** new learning rate: %f" % (lr * new_lr_decay)
            print(log)
            # logging.debug(log)
        elif epoch == 0:
            log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        step, total_D, total_G = 0, 0, 0
        for b_x, b_y in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            step_time = time.time()
            b_z = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, z_dim))
            b_x = tl.prepro.threading_data(b_x, _data_aug_fn)
            # update D
            errD, _ = sess.run([d_loss, d_optim], {t_image: b_x, t_class: b_y, t_z: b_z})
            # update G
            errG, _ = sess.run([g_loss, g_optim], {t_image: b_x, t_class: b_y, t_z: b_z})
            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                % (epoch, n_epoch, step, n_step_epoch, time.time() - step_time, errD, errG))
            step += 1; total_D += errD ; total_G += errG
        print("[*] Epoch [%2d/%2d] time: %4.4f, avg_d_loss: %.8f, avg_g_loss: %.8f" %
                (epoch, n_epoch, time.time()-epoch_time, total_D/step, total_G/step))

        out = sess.run(net_g_test.outputs, {t_z: sample_z, t_class: sample_c})        # ni = int(np.ceil(np.sqrt(batch_size)))
        tl.vis.save_images(out, [10, 10], save_dir + '/train_{:02d}c.png'.format(epoch))

        ## save model
        if (epoch != 0) and (epoch % 5) == 0:
            tl.files.save_npz(net_d.all_params, name=save_dir_d, sess=sess)
            tl.files.save_npz(net_g.all_params, name=save_dir_g, sess=sess)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="train",
                       help='train, test')

    args = parser.parse_args()

    if args.mode == "train":
        main()
