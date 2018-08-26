import os
import shutil
import tensorflow as tf
import numpy as np
import dataset
import config
import cnn_model
import datetime


shutil.rmtree('./tmp')
os.mkdir('./tmp')

x_train, y_train, x_test, y_test = dataset.get_x_y_train_test()

print(len(y_train), len(y_test))

vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(dataset.MAX_LEN_X)

x_transform_train = vocab_processor.fit_transform(x_train)
x_transform_test = vocab_processor.transform(x_test)

vocab_processor.save('./tmp/vocab')

x_train = np.array(list(x_transform_train))
x_test = np.array(list(x_transform_test))

y_train = np.array(list(y_train))
y_test = np.array(list(y_test))
embedding_dim = 128
filter_sizes = '3,4,5'
l2_reg_lambda = 0.0
num_filters = 128

with tf.Session() as sess:
    cnn = cnn_model.TextCNN(
        sequence_length=dataset.get_max_len(),
        num_classes=len(config.classes),
        vocab_size=len(vocab_processor.vocabulary_),
        embedding_size=embedding_dim,
        filter_sizes=list(map(int, filter_sizes.split(","))),
        num_filters=num_filters,
        l2_reg_lambda=l2_reg_lambda
    )
    out_dir = os.path.abspath(os.path.join('tmp', "runs/"))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy

    loss_summary = tf.summary.scalar("loss", cnn.loss)
    acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary,
                                         cnn.grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

    sess.run(tf.global_variables_initializer())


    def train_step(x_train_batch, y_train_batch):
        start_time = datetime.datetime.now()
        feed_dict = {
            cnn.input_x: x_train_batch,
            cnn.input_y: y_train_batch
        }
        step, summaries, _, loss_ = sess.run([cnn.global_step, train_summary_op, cnn.train_op, cnn.loss],
                                             feed_dict=feed_dict)
        train_summary_writer.add_summary(summaries, step)
        end_time = datetime.datetime.now()
        print "TRAIN: step : {}, loss: {}, step_time : {}s".format(step, loss_, (end_time - start_time).seconds)

    def test_step(x_test_batch, y_test_batch):
        start_time = datetime.datetime.now()
        # zip(x_test_batch, y_test_batch)
        feed_dict = {
            cnn.input_x: x_test_batch[:64],
            cnn.input_y: y_test_batch[:64]
        }
        step, summaries, loss_ = sess.run([cnn.global_step, train_summary_op, cnn.loss],
                                             feed_dict=feed_dict)
        dev_summary_writer.add_summary(summaries, step)
        end_time = datetime.datetime.now()
        print "TEST: step : {}, loss: {}, step_time : {}s".format(step, loss_, (end_time - start_time).seconds)


    batches = dataset.batch_iter(zip(x_train, y_train), 64, 5)
    for i, batch in enumerate(batches):
        x_train_batch, y_train_batch = zip(*batch)
        train_step(x_train_batch, y_train_batch)

        if i % 5 == 0:
            test_step(x_test, y_test)
            current_step = tf.train.global_step(sess, cnn.global_step)
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
