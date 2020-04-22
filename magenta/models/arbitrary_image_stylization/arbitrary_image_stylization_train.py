# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Trains a real-time arbitrary image stylization model.

For example of usage see start_training_locally.sh and start_training_on_borg.sh
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import os

from tensorflow_core.python.keras.optimizer_v2.adam import Adam
from tensorflow_core.python.ops.image_ops_impl import ResizeMethod

from magenta.models.arbitrary_image_stylization import arbitrary_image_stylization_build_model as build_model
from magenta.models.image_stylization import image_utils
from magenta.models.image_stylization import vgg
import tensorflow.compat.v1 as tf
from tensorflow.contrib import slim as contrib_slim
from torch.utils.data import DataLoader
from magenta.models.arbitrary_image_stylization.dataset import *
from magenta.models.arbitrary_image_stylization.utilities import *
from keras_applications.vgg16 import VGG16

slim = contrib_slim

DEFAULT_CONTENT_WEIGHTS = '{"vgg_16/conv3": 1}'
DEFAULT_STYLE_WEIGHTS = ('{"vgg_16/conv1": 0.5e-3, "vgg_16/conv2": 0.5e-3,'
                         ' "vgg_16/conv3": 0.5e-3, "vgg_16/conv4": 0.5e-3}')

flags = tf.app.flags
flags.DEFINE_float('clip_gradient_norm', 0, 'Clip gradients to this norm')
flags.DEFINE_float('learning_rate', 1e-5, 'Learning rate')
flags.DEFINE_float('total_variation_weight', 1e4, 'Total variation weight')
flags.DEFINE_string('content_weights', DEFAULT_CONTENT_WEIGHTS,
                    'Content weights')
flags.DEFINE_string('style_weights', DEFAULT_STYLE_WEIGHTS, 'Style weights')
flags.DEFINE_integer('batch_size', 8, 'Batch size.')
flags.DEFINE_integer('image_size', 256, 'Image size.')
flags.DEFINE_boolean('random_style_image_size', True,
                     'Wheather to augment style images or not.')
flags.DEFINE_boolean(
    'augment_style_images', True,
    'Wheather to resize the style images to a random size or not.')
flags.DEFINE_boolean('center_crop', False,
                     'Wheather to center crop the style images.')
flags.DEFINE_integer('ps_tasks', 0,
                     'Number of parameter servers. If 0, parameters '
                     'are handled locally by the worker.')
flags.DEFINE_integer('save_summaries_secs', 15,
                     'Frequency at which summaries are saved, in seconds.')
flags.DEFINE_integer('save_interval_secs', 15,
                     'Frequency at which the model is saved, in seconds.')
flags.DEFINE_integer('task', 0, 'Task ID. Used when training with multiple '
                                'workers to identify each worker.')
flags.DEFINE_integer('train_steps', 8000000, 'Number of training steps.')
flags.DEFINE_string('master', '', 'BNS name of the TensorFlow master to use.')
flags.DEFINE_string('style_dataset_file', None, 'Style dataset file.')
flags.DEFINE_string('train_dir', None,
                    'Directory for checkpoints and summaries.')
flags.DEFINE_string('inception_v3_checkpoint', None,
                    'Path to the pre-trained inception_v3 checkpoint.')

FLAGS = flags.FLAGS

L2distancematrix = lambda x,y : tf.compat.v1.losses.mean_squared_error(x,y, reduction='none')
L2distance = lambda x,y : tf.compat.v1.losses.mean_squared_error(x,y)

def train(epochs=20):
    dataloader = DataLoader(FlyingChairsDataset("../FlyingChairs2/"), batch_size=1)
    adam = Adam(learning_rate=FLAGS.learning_rate)
    train_op_adam = adam.minimize(loss)
    # Loads style images.
    [style_inputs_, _,
     style_inputs_orig_] = image_utils.arbitrary_style_image_inputs(
        FLAGS.style_dataset_file,
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.image_size,
        shuffle=True,
        center_crop=FLAGS.center_crop,
        augment_style_images=FLAGS.augment_style_images,
        random_style_image_size=FLAGS.random_style_image_size)
    style_GM = [gram_matrix(f) for f in style_inputs_]

    saver = tf.train.Saver()

    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
        # Process style and content weight flags.
        content_weights = ast.literal_eval(FLAGS.content_weights)
        style_weights = ast.literal_eval(FLAGS.style_weights)

    for epoch in range(epochs):
        for itr, (img1, img2, mask, flow) in enumerate(dataloader):
            flow = -flow

        with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
            # Define the model
            stylized_images, total_loss, loss_dict, _, feature_maps = build_model.build_model(
                [img1, img2],
                style_inputs_,
                trainable=True,
                is_training=True,
                inception_end_point='Mixed_6e',
                style_prediction_bottleneck=100,
                adds_losses=True,
                content_weights=content_weights,
                style_weights=style_weights,
                total_variation_weight=FLAGS.total_variation_weight)
            feature_map1, styled_img1 = stylized_images[0], feature_maps[0]
            feature_map2, styled_img2 = stylized_images[1], feature_maps[1]
            styled_img1 = normalize(styled_img1)
            styled_img2 = normalize(styled_img2)
            img1, img2 = normalize(img1), normalize(img2)

            styled_features1 = VGG16(input_tensor=styled_img1).output
            styled_features2 = VGG16(input_tensor=styled_img2).output
            img_features1 = VGG16(input_tensor=img1).output
            img_features2 = VGG16(input_tensor=img2).output

            feature_flow = tf.image.resize(
                flow, feature_map1.shape[2:], method=ResizeMethod.BILINEAR, )

            feature_flow[0, 0, :, :] *= float(feature_map1.shape[2]) / flow.shape[2]
            feature_flow[0, 1, :, :] *= float(feature_map1.shape[3]) / flow.shape[3]
            # print(flow.size(), feature_map1.shape[2:],feature_flow.size())

            feature_mask = tf.image.resize(
                mask.view(1, 1, 640, 360), size=feature_map1.shape[2:], mode='bilinear')
            # print(feature_map1.size(), feature_flow.size())
            warped_fmap = warp(feature_map1, feature_flow)

            # #Changed by KJ to multiply with feature mask
            # # print(L2distancematrix(feature_map2, warped_fmap).size()) #Should be a matrix not number
            # # mean replaced sum
            f_temporal_loss = torch.sum(feature_mask * (L2distancematrix(feature_map2, warped_fmap)))
            f_temporal_loss *= LAMBDA_F
            f_temporal_loss *= 1 / (feature_map2.shape[1] * feature_map2.shape[2] * feature_map2.shape[3])

            # # print(styled_img1.size(), flow.size())
            # # Removed unsqueeze methods in both styled_img1,flow in next line since already 4 dimensional
            warped_style = warp(styled_img1, flow)
            warped_image = warp(img1, flow)

            # print(img2.size())
            output_term = styled_img2[0] - warped_style[0]
            # print(output_term.shape, styled_img2.shape, warped_style.shape)
            input_term = img2[0] - warped_image[0]
            # print(input_term.size())
            # Changed the next few lines since dimension is 4 instead of 3 with batch size=1
            input_term = 0.2126 * input_term[0, :, :] + 0.7152 * \
                         input_term[1, :, :] + 0.0722 * input_term[2, :, :]
            input_term = input_term.expand(output_term.size())

            o_temporal_loss = torch.sum(mask * (L2distancematrix(output_term, input_term)))
            o_temporal_loss *= LAMBDA_O
            o_temporal_loss *= 1 / (img1.shape[2] * img1.shape[3])

            content_loss = 0
            content_loss += L2distance(styled_features1[2],
                                       img_features1[2].expand(styled_features1[2].size()))
            content_loss += L2distance(styled_features2[2],
                                       img_features2[2].expand(styled_features2[2].size()))
            content_loss *= ALPHA / (
                        styled_features1[2].shape[1] * styled_features1[2].shape[2] * styled_features1[2].shape[3])

            style_loss = 0

            for i, weight in enumerate(style_weights):
                gram_s = style_GM[i]
                # print(styled_features1[i].size())
                gram_img1 = gram_matrix(styled_features1[i])
                gram_img2 = gram_matrix(styled_features2[i])
                # print(gram_img1.size(), gram_s.size())
                style_loss += float(weight) * (L2distance(gram_img1, gram_s.expand(
                    gram_img1.size())) + L2distance(gram_img2, gram_s.expand(gram_img2.size())))
            style_loss *= BETA

            reg_loss = GAMMA * \
                       (torch.sum(torch.abs(styled_img1[:, :, :, :-1] - styled_img1[:, :, :, 1:])) +
                        torch.sum(torch.abs(styled_img1[:, :, :-1, :] - styled_img1[:, :, 1:, :])))

            reg_loss += GAMMA * \
                        (torch.sum(torch.abs(styled_img2[:, :, :, :-1] - styled_img2[:, :, :, 1:])) +
                         torch.sum(torch.abs(styled_img2[:, :, :-1, :] - styled_img2[:, :, 1:, :])))

            # print(f_temporal_loss.size(), o_temporal_loss.size(), content_loss.size(), style_loss.size(), reg_loss.size())
            loss = f_temporal_loss + o_temporal_loss + content_loss + style_loss + reg_loss
            # loss = content_loss + style_loss
            train_op_adam = adam.minimize(loss)

            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)

            if (itr + 1) % 1000 == 0:
                saver.save(sess,
                           '%s/final_reconet_epoch_%d_itr_%d.pth' % ("runs/output", epoch, itr // 1000))

            print('[%d/%d][%d/%d] SL: %.4f CL: %.4f FTL: %.4f OTL: %.4f RL: %.4f'
                  % (epoch, epochs, itr, len(dataloader),
                     style_loss, content_loss, f_temporal_loss, o_temporal_loss, reg_loss))
    saver.save(sess, '%s/reconet_epoch_%d.pth' % ("runs/output", epoch))

def main(unused_argv=None):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        # Forces all input processing onto CPU in order to reserve the GPU for the
        # forward inference and back-propagation.
        device = '/cpu:0' if not FLAGS.ps_tasks else '/job:worker/cpu:0'
        with tf.device(
                tf.train.replica_device_setter(FLAGS.ps_tasks, worker_device=device)):
            # Loads content images.
            content_inputs_, _ = image_utils.imagenet_inputs(FLAGS.batch_size,
                                                             FLAGS.image_size)

            # Loads style images.
            [style_inputs_, _,
             style_inputs_orig_] = image_utils.arbitrary_style_image_inputs(
                FLAGS.style_dataset_file,
                batch_size=FLAGS.batch_size,
                image_size=FLAGS.image_size,
                shuffle=True,
                center_crop=FLAGS.center_crop,
                augment_style_images=FLAGS.augment_style_images,
                random_style_image_size=FLAGS.random_style_image_size)

        with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
            # Process style and content weight flags.
            content_weights = ast.literal_eval(FLAGS.content_weights)
            style_weights = ast.literal_eval(FLAGS.style_weights)

            # Define the model
            stylized_images, total_loss, loss_dict, _ = build_model.build_model(
                content_inputs_,
                style_inputs_,
                trainable=True,
                is_training=True,
                inception_end_point='Mixed_6e',
                style_prediction_bottleneck=100,
                adds_losses=True,
                content_weights=content_weights,
                style_weights=style_weights,
                total_variation_weight=FLAGS.total_variation_weight)

            # Adding scalar summaries to the tensorboard.
            for key, value in loss_dict.items():
                tf.summary.scalar(key, value)

            # Adding Image summaries to the tensorboard.
            tf.summary.image('image/0_content_inputs', content_inputs_, 3)
            tf.summary.image('image/1_style_inputs_orig', style_inputs_orig_, 3)
            tf.summary.image('image/2_style_inputs_aug', style_inputs_, 3)
            tf.summary.image('image/3_stylized_images', stylized_images, 3)

            # Set up training
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            train_op = slim.learning.create_train_op(
                total_loss,
                optimizer,
                clip_gradient_norm=FLAGS.clip_gradient_norm,
                summarize_gradients=False)

            # Function to restore VGG16 parameters.
            init_fn_vgg = slim.assign_from_checkpoint_fn(vgg.checkpoint_file(),
                                                         slim.get_variables('vgg_16'))

            # Function to restore Inception_v3 parameters.
            inception_variables_dict = {
                var.op.name: var
                for var in slim.get_model_variables('InceptionV3')
            }
            init_fn_inception = slim.assign_from_checkpoint_fn(
                FLAGS.inception_v3_checkpoint, inception_variables_dict)

            # Function to restore VGG16 and Inception_v3 parameters.
            def init_sub_networks(session):
                init_fn_vgg(session)
                init_fn_inception(session)

            # Run training
            slim.learning.train(
                train_op=train_op,
                logdir=os.path.expanduser(FLAGS.train_dir),
                master=FLAGS.master,
                is_chief=FLAGS.task == 0,
                number_of_steps=FLAGS.train_steps,
                init_fn=init_sub_networks,
                save_summaries_secs=FLAGS.save_summaries_secs,
                save_interval_secs=FLAGS.save_interval_secs)


def console_entry_point():
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
