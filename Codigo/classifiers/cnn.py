import numpy as np
import tensorflow.compat.v1 as tf

from sklearn.preprocessing import MinMaxScaler


class cnn:

    def __init__(self, Xtrain, Ytrain) -> None:
        tf.disable_v2_behavior()
        super().__init__()
        self.mnist_classifier = None
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.h = Xtrain.shape[0]
        self.n = Xtrain.shape[1]
        self.scaler = MinMaxScaler()
        self.Xtrain_scaler = self.scaler.fit_transform(self.Xtrain)
        self.feature_columns = [tf.feature_column.numeric_column('x', shape=self.Xtrain_scaler.shape[1:])]

    @staticmethod
    def cnn_model_fn(features, labels, mode):
        """Model function for CNN."""
        # Input Layer
        input_layer = tf.reshape(features["x"], [-1, 13, 1, 1])

        # Convolutional Layer
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=16,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 1], strides=1)

        # Convolutional Layer #2 and Pooling Layer
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=32,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 1], strides=1)

        # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 352])
        dense = tf.layers.dense(inputs=pool2_flat, units=352, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=10)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics Evaluation mode
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def train(self):
        self.mnist_classifier = tf.estimator.Estimator(
            model_fn=self.cnn_model_fn, model_dir="train/mnist_convnet_model")
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": self.Xtrain},
            y=self.Ytrain,
            batch_size=100,
            num_epochs=None,
            shuffle=True)
        self.mnist_classifier.train(
            input_fn=train_input_fn,
            steps=16000,
            hooks=[logging_hook])

    def calc_saida(self, Xtest, Ytest):
        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": Xtest},
            y=Ytest,
            num_epochs=1,
            shuffle=False)
        eval_results = self.mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)
