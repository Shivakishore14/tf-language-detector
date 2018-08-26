import tensorflow as tf
import dataset
import model
import numpy as np
import config
tf.logging.set_verbosity(tf.logging.INFO)

vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore('./tmp/vocab')

filenames = ['test.go', 'model.py', 'predict.py', 'dataset.py']

x = []
y_names = []
for filename in filenames:
    data = dataset.read_file(filename)
    y_names.extend([filename] * len(data))
    x.extend(data)

x = np.array(list(vocab_processor.transform(x)))
#
# test_input_fn = tf.estimator.inputs.numpy_input_fn(
#       x={'x': x}, num_epochs=1, shuffle=False)
# predictions = classifier.predict(input_fn=test_input_fn)
# y_predicted = np.array(list(p['classes'] for p in predictions))
# for p in y_predicted:
#     print config.classes[int(p)]


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


graph = load_graph('tmp/frozen_model.pb')

# We can verify that we can access the list of operations in the graph
for op in graph.get_operations():
    print(op.name)

# We access the input and output nodes
input_x = graph.get_tensor_by_name('prefix/input_x:0')
input_y = graph.get_tensor_by_name('prefix/input_y:0')
scores = graph.get_tensor_by_name('prefix/output/scores:0')
predictions = graph.get_tensor_by_name('prefix/output/predictions:0')

# We launch a Session
with tf.Session(graph=graph) as sess:
    scores, predictions = sess.run([scores, predictions], feed_dict={input_x:x})
    counts = np.bincount(predictions)
    for i, pred in enumerate(predictions):

        print y_names[i], config.classes[pred]