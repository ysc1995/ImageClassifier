import tensorflow as tf
import data_helper


totalstep = 1000


def construct(images, pixels, classes,h):
  reg=0.1
  with tf.variable_scope('first'):
    create = tf.get_variable('w', [pixels, h],initializer=tf.random_normal_initializer(stddev=0.002),regularizer=tf.contrib.layers.l2_regularizer(reg))

    past = tf.nn.relu(tf.matmul(images, create) )


  with tf.variable_scope('second'):


    create = tf.get_variable('w', [h, classes],initializer=tf.random_normal_initializer(stddev=0.1),regularizer=tf.contrib.layers.l2_regularizer(reg))

    res = tf.matmul(past, create)

    tf.summary.histogram('l', res)
  return res


def main():
  labelholder = tf.placeholder(tf.int64, shape=[None], name='labels')
  imageholder = tf.placeholder(tf.float32, shape=[None, 3072], name='images')
  layers = construct(imageholder, 3072, 10, 100)

  with tf.name_scope('En'):
    L = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layers, labels=labelholder, name='cross_entropy'))
    tf.summary.scalar('en', L)

  currstep =tf.train.GradientDescentOptimizer(0.001).minimize(L, global_step=tf.Variable(0, name='global_step',trainable=False))
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    zdata = zip(data_helper.load_data()['images_train'], data_helper.load_data()['labels_train'])
    batches = data_helper.gen_batch(list(zdata), 400, totalstep)

    for i in range(totalstep):

      batch = next(batches)
      images_batch, labels_batch = zip(*batch)

      if i % 100 == 0:
        with tf.name_scope('a'):
          accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(layers, 1), labelholder), tf.float32))
          tf.summary.scalar('ta', accuracy)
        print('training accuracy for {:d} step is {:g}'.format(i, sess.run(accuracy, feed_dict={imageholder: images_batch, labelholder: labels_batch})))

      sess.run([currstep, L], feed_dict = {imageholder: images_batch, labelholder: labels_batch})


    print('Accuracy is ' + format(sess.run(accuracy, feed_dict={imageholder: data_helper.load_data()['images_test'], labelholder: data_helper.load_data()['labels_test']})))


if __name__ == '__main__':
  main()