from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
import config
import dataset


app = Flask(__name__)


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

# We access the input and output nodes
input_x = graph.get_tensor_by_name('prefix/input_x:0')
input_y = graph.get_tensor_by_name('prefix/input_y:0')
scores = graph.get_tensor_by_name('prefix/output/scores:0')
predictions = graph.get_tensor_by_name('prefix/output/predictions:0')

sess = tf.Session(graph=graph)
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore('./tmp/vocab')

@app.route('/ping')
def health_check():
    return "pong"


@app.route('/predict', methods=['POST'])
def predict():

    data = request.data
    print data.split('\n')
    x_data = dataset.process_lines(data.split('\n'))
    print x_data
    x = np.array(list(vocab_processor.transform(x_data)))
    scores_, predictions_ = sess.run([scores, predictions], feed_dict={input_x: x})
    counts = np.bincount(predictions_)
    class_ = config.classes[np.argmax(counts)]
    return class_


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5454, threaded=True)
