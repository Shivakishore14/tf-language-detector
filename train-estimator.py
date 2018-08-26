import os
import shutil
import tensorflow as tf
import numpy as np
import dataset
import config

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


def model_fn(features, labels, mode, params):
    print ("print", features, labels, params)
    # tf.Print(features)
    num_filters = 128
    l2_reg_lambda = 0.0
    filter_sizes = '3,4,5'
    embedding_dim = 128
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore('./tmp/vocab')

    vocab_size = len(vocab_processor.vocabulary_)
    embedding_size = embedding_dim
    num_filters = num_filters
    filter_sizes = list(map(int, filter_sizes.split(",")))
    sequence_length = dataset.get_max_len()
    num_classes = len(config.classes)
    dropout_keep_prob = params["dropout_keep_prob"]

    # Keeping track of l2 regularization loss (optional)
    l2_loss = tf.constant(0.0)

    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        W = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            name="W")
        embedded_chars = tf.nn.embedding_lookup(W, features['x'])
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

    print "test", embedded_chars, embedded_chars_expanded
    # Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                embedded_chars_expanded,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            print 'tk', conv
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, sequence_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)

    print 'tk', pooled_outputs
    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    # Add dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, 0.5)

    print 'klk', pooled_outputs
    # Final (unnormalized) scores and predictions
    with tf.name_scope("output"):
        W = tf.get_variable(
            "W",
            shape=[num_filters_total, num_classes],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
        predictions = tf.argmax(scores, 1, name="predictions")

    # Calculate mean cross-entropy loss
    with tf.name_scope("loss"):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=labels)
        loss = tf.add(tf.reduce_mean(losses), l2_reg_lambda * l2_loss, name="loss_mean")

    # Accuracy
    with tf.name_scope("accuracy"):
        correct_predictions = tf.equal(predictions, tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    print 'klk', accuracy,loss


    export_outputs = {'predict_output': tf.estimator.export.PredictOutput(
        {"pred_output_classes": predictions, 'score': scores})}
    predictions_dict = {"predicted": predictions}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          export_outputs=export_outputs,
                                          predictions=predictions_dict)

    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
    # Define the evaluation metrics,
    # in this case the classification accuracy.

    metrics = \
        {
            "accuracy": tf.metrics.accuracy(labels, scores)
        }
    export_outputs = {'predict_output': tf.estimator.export.PredictOutput(
        {"pred_output_classes": predictions, 'score': scores})}
    predictions_dict = {"predicted": predictions}
    logging_hook = tf.train.LoggingTensorHook({"embedded": embedded_chars_expanded.shape}, every_n_iter=10)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics,
        export_outputs=export_outputs,
        predictions=predictions_dict,
        training_hooks=[logging_hook]
    )


train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': x_train},
      y=y_train,
      batch_size=len(x_train),
      num_epochs=None,
      shuffle=True)


params = {"learning_rate": 1e-4, "dropout_keep_prob": 0.5}
classifier = tf.estimator.Estimator(model_fn=model_fn,
                                    params=params,
                                    model_dir='./tmp')

classifier.train(input_fn=train_input_fn, steps=1000)

vocab_processor.save('./tmp/vocab')

test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': x_test}, y=y_test, num_epochs=1, shuffle=False)
predictions = classifier.predict(input_fn=test_input_fn)
# for i in predictions:
#     print i
y_predicted = np.array(list(p['classes'] for p in predictions))
print y_predicted
print y_test
y_predicted = y_predicted.reshape(np.array(y_test).shape)

