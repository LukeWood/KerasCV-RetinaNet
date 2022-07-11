import retina_net
from loader import load_pascal_voc
import tensorflow as tf

train_ds = load_pascal_voc(bounding_box_format="rel_xyxy", split="train", batch_size=2)
model = retina_net.RetinaNet(
    num_classes=20, bounding_box_format="rel_xyxy", include_rescaling=False
)
model.compile(optimizer="adam", loss=retina_net.FocalLoss(num_classes=20))


def unpackage_dict(inputs):
    return inputs["images"], inputs["bounding_boxes"]


train_ds = train_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)

model.fit(train_ds)
