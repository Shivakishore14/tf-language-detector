import tensorflow as tf
import config
import cnn_model
tf.logging.set_verbosity(tf.logging.INFO)


def create(max_len):
    feature_columns = [tf.feature_column.numeric_column("x", shape=[max_len])]

    my_checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_steps=10,
        keep_checkpoint_max=3,
        log_step_count_steps=10,
        save_summary_steps=10
    )

    classifier = tf.estimator.DNNClassifier(
     feature_columns=feature_columns,
     hidden_units=[256, 32],
     optimizer=tf.train.AdamOptimizer(1e-3),
     n_classes=len(config.classes),
     dropout=0.1,
     model_dir='./tmp',
     config=my_checkpointing_config,
     activation_fn=tf.nn.relu
    )
    params = {"learning_rate": 1e-4, "dropout_keep_prob": 0.5}
    classifier = tf.estimator.Estimator(model_fn=cnn_model.model_fn,
                                        params=params,
                                        model_dir='./tmp')
    return classifier
