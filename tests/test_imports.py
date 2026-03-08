"""TensorFlow / tf.keras の import が通ることを確認する。"""
import pytest


def test_tensorflow_import():
    import tensorflow as tf
    assert tf is not None


def test_tf_keras_import():
    import tensorflow as tf
    from tensorflow import keras
    assert keras is not None
    assert hasattr(tf.keras, 'Model')
    assert hasattr(tf.keras.layers, 'Dense')
