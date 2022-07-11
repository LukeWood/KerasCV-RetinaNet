import keras_cv
import numpy as np
import tensorflow as tf
from absl import flags
from ml_collections.config_flags import config_flags
from tensorflow import keras

from retina_net import layers as layers_lib

# --- Building RetinaNet using a subclassed model ---
class RetinaNet(keras.Model):
    """A Keras model implementing the RetinaNet architecture."""

    def __init__(
        self,
        num_classes,
        bounding_box_format,
        include_rescaling=None,
        backbone=None,
        prediction_decoder=None,
        name="RetinaNet",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if backbone is None and include_rescaling is None:
            raise ValueError(
                "Either `backbone` OR `include_rescaling` must be set when "
                "constructing a `keras_cv.models.RetinaNet()` model. "
                f"Received backbone={backbone}, include_rescaling={include_rescaling}."
            )

        self.feature_pyramid = layers_lib.FeaturePyramid(backbone)
        self.bounding_box_format = bounding_box_format
        self.num_classes = num_classes

        self.backbone = backbone or _default_backbone(include_rescaling)

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))

        # TODO(lukewood): make configurable
        self.classification_head = layers_lib.PredictionHead(
            output_filters=9 * num_classes, bias_initializer=prior_probability
        )
        self.box_head = layers_lib.PredictionHead(
            output_filters=9 * 4, bias_initializer="zeros"
        )

        self.prediction_decoder = prediction_decoder or layers_lib.DecodePredictions(
            num_classes=num_classes, bounding_box_format=bounding_box_format
        )

    def call(self, x, training=False):
        features = self.fpn(x, training=training)
        N = tf.shape(x)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.classification_head(feature), [N, -1, self.num_classes])
            )

        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        train_preds = tf.concat([box_outputs, cls_outputs], axis=-1)

        # no-op if default decoder is used.
        pred_for_inference = bounding_box.convert_format(
            pred_for_inference,
            source=self.bounding_box_format,
            target=self.decoder.bounding_box_format,
        )
        pred_for_inference = self.decoder(x, pred_for_inference)
        pred_for_inference = bounding_box.convert_format(
            pred_for_inference,
            source=self.decoder.bounding_box_format,
            target=self.bounding_box_format,
        )
        return {"train_preds": train_preds, "inference": pred_for_inference}

    def _update_metrics(self, y_for_metrics, result):
        # COCO metrics are all stored in compiled_metrics
        # This tf.cond is needed to work around a TensorFlow edge case in Ragged Tensors
        tf.cond(
            tf.shape(result)[2] != 0,
            lambda: self.compiled_metrics.update_state(y_for_metrics, result),
            lambda: None,
        )

    def _metrics_result(self, loss):
        metrics_result = {m.name: m.result() for m in self.metrics}
        metrics_result["loss"] = loss
        return metrics_result

    def train_step(self, data, training=True):
        x, y = data
        y_for_metrics = y

        y = bounding_box.convert_format(
            y, source=self.bounding_box_format, target="xywh"
        )
        y_training_target = self.label_encoder.encode_batch(y)

        with tf.GradientTape() as tape:
            predictions = self(x, training=training)
            loss = self._loss(y_training_target, predictions["train_preds"])
            for extra_loss in self.losses:
                loss += extra_loss

        self._update_metrics(
            y_for_metrics,
        )

        # Training specific code
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # clip grads to prevent explosion
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Return metric result

        return self._metrics_result(loss)

    def inference(self, x):
        predictions = self.predict(x)
        return predictions["inference"]


# --- Building the ResNet50 backbone ---
def default_backbone(include_rescaling):
    """Builds ResNet50 with pre-trained imagenet weights"""
    # TODO(lukewood): include_rescaling
    if include_rescaling:
        raise ValueError("include_rescaling is a TODO.  KerasCV API will cover this.")
    backbone = keras.applications.ResNet50(
        include_top=False,
        input_shape=[None, None, 3],
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return keras.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )
