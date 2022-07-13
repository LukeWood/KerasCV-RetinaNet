import retina_net
from loader import load_pascal_voc
import tensorflow as tf
import keras_cv
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras import callbacks as callbacks_lib


wandb.init(project="pascalvoc-retinanet", entity="keras-team-testing")

ids = list(range(20))
metrics = [
    keras_cv.metrics.COCOMeanAveragePrecision(
        class_ids=ids,
        bounding_box_format="xywh",
        name="Standard MaP",
    ),
    keras_cv.metrics.COCOMeanAveragePrecision(
        class_ids=ids,
        bounding_box_format="xywh",
        iou_thresholds=[0.5],
        name="MaP IoU=0.5",
    ),
    keras_cv.metrics.COCOMeanAveragePrecision(
        class_ids=ids,
        bounding_box_format="xywh",
        iou_thresholds=[0.75],
        name="MaP IoU=0.75",
    ),
    keras_cv.metrics.COCOMeanAveragePrecision(
        class_ids=ids,
        bounding_box_format="xywh",
        area_range=(0, 32**2),
        name="MaP Small Objects",
    ),
    keras_cv.metrics.COCOMeanAveragePrecision(
        class_ids=ids,
        bounding_box_format="xywh",
        area_range=(32**2, 96**2),
        name="MaP Medium Objects",
    ),
    keras_cv.metrics.COCOMeanAveragePrecision(
        class_ids=ids,
        bounding_box_format="xywh",
        area_range=(96**2, 1e9**2),
        name="MaP Large Objects",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=ids,
        bounding_box_format="xywh",
        max_detections=1,
        name="Recall 1 Detection",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=ids,
        bounding_box_format="xywh",
        max_detections=10,
        name="Recall 10 Detections",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=ids,
        bounding_box_format="xywh",
        max_detections=100,
        name="Standard Recall",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=ids,
        bounding_box_format="xywh",
        area_range=(0, 32**2),
        name="Recall Small Objects",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=ids,
        bounding_box_format="xywh",
        area_range=(32**2, 96**2),
        name="Recall Medium Objects",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=ids,
        bounding_box_format="xywh",
        area_range=(96**2, 1e9**2),
        name="Recall Large Objects",
    ),
]

# train_ds is batched as a (images, bounding_boxes) tuple
# bounding_boxes are ragged
train_ds = load_pascal_voc(bounding_box_format="xywh", split="train", batch_size=2)
val_ds = load_pascal_voc(bounding_box_format="xywh", split="validation", batch_size=2)


def unpackage_dict(inputs):
    return inputs["images"] / 255.0, inputs["bounding_boxes"]


# TODO(lukewood): preprocessing

train_ds = train_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

optimizer = tf.optimizers.SGD(
    learning_rate=learning_rate_fn, momentum=0.9, global_clipnorm=10.0
)

# No rescaling
model = retina_net.RetinaNet(
    num_classes=20, bounding_box_format="xywh", include_rescaling=False
)
model.compile(
    optimizer=optimizer, loss=retina_net.FocalLoss(num_classes=20), metrics=metrics
)

callbacks = [
    callbacks_lib.TensorBoard(log_dir="logs"),
    WandbCallback(),
    callbacks_lib.EarlyStopping(patience=5),
]

# model.fit(train_ds, validation_data=val_ds, epochs=500, callbacks=callbacks)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=500,
    callbacks=callbacks,
)
