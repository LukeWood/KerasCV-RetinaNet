import retina_net
from loader import load_pascal_voc
import tensorflow as tf
import keras_cv
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras import callbacks as callbacks_lib
import metrics as metrics_lib
from retina_net.callbacks import VisualizeBoxes
# wandb.init(project="pascalvoc-retinanet", entity="keras-team-testing")

# train_ds is batched as a (images, bounding_boxes) tuple
# bounding_boxes are ragged
train_ds, train_dataset_info = load_pascal_voc(
    bounding_box_format="xywh", split="train", batch_size=2
)
val_ds, val_dataset_info = load_pascal_voc(
    bounding_box_format="xywh", split="validation", batch_size=2
)


def unpackage_dict(inputs):
    return inputs["images"] / 255.0, inputs["bounding_boxes"]


train_ds = train_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)

optimizer = tf.keras.optimizers.Adam(global_clipnorm=10.0)

# No rescaling
model = retina_net.RetinaNet(
    num_classes=20,
    bounding_box_format="xywh",
    backbone="resnet50",
    backbone_weights="imagenet",
    include_rescaling=True,
)
model.compile(
    optimizer=optimizer,
    loss=retina_net.FocalLoss(num_classes=20),
    metrics=metrics_lib.get_metrics(bounding_box_format="xywh", num_classes=20),
)

callbacks = [
    callbacks_lib.TensorBoard(log_dir="logs"),
    #WandbCallback(),
    callbacks_lib.EarlyStopping(patience=5),
    # retina_net.VisualizeBoxes(
    #     validation_data=val_ds,
    #     dataset_info=val_dataset_info,
    #     bounding_box_format="xywh",
    # ),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=500,
    callbacks=callbacks,
)
