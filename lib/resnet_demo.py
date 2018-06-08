import tensorflow as tf
slim = tf.contrib.slim
from PIL import Image
from resnet_v2 import *
from utils import *
import resnet_utils
import numpy as np
from DataLayer import DataLayer


checkpoint_file = './models/resnet_v2_101.ckpt'
var_keep_dic = get_variables_in_checkpoint_file(checkpoint_file)
for k, v in var_keep_dic.items()[:10]:
    print(k)
data_path = './data/demo/'
sample_images = ['60_10.jpg']  # , '60_10.jpg', '93_96.jpg', '107_24.jpg']
#, '9.jpg', '21.jpg']
# Load the model
sess = tf.Session()
# arg_scope = inception_resnet_v2_arg_scope()
arg_scope = resnet_arg_scope()
input_tensor = tf.placeholder(tf.float32, [None, 225, 225, 3])
with slim.arg_scope(arg_scope):
    logits, end_points = resnet_v2_101(input_tensor, num_classes=1001, is_training=False)
saver = tf.train.Saver()
saver.restore(sess, checkpoint_file)
# data_in = np.zeros([64, 225, 225, 3], dtype=np.float32)
# for i, image in enumerate(sample_images):
#     im_path = data_path + image
#     im = Image.open(im_path).resize((225, 225))
#     im = np.array(im)
#     im = 2 * (im / 255.0) - 1.0
#     im = im.reshape(-1, 225, 225, 3)
# for i in range(data_in.shape[0]):
#     data_in[i] = im
imdb = DataLayer("train")
data_in, label = imdb.get_minibatch(16)
print(data_in.shape)
logit_values = sess.run([logits], feed_dict={input_tensor: data_in})
# a = np.asarray(logit_values)
# import pdb
# pdb.set_trace()
# print(len(logit_values[0]))
print(np.max(logit_values, axis=2))
print(np.argmax(logit_values, axis=2))
