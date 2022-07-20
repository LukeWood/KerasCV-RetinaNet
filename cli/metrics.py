import keras_cv


def get_metrics(num_classes, bounding_box_format):
    ids = list(range(num_classes))
    metrics = [
        keras_cv.metrics.COCOMeanAveragePrecision(
            class_ids=ids,
            name="Standard MaP",
        ),
        keras_cv.metrics.COCOMeanAveragePrecision(
            class_ids=ids,
            bounding_box_format=bounding_box_format,
            iou_thresholds=[0.5],
            name="MaP IoU=0.5",
        ),
        keras_cv.metrics.COCOMeanAveragePrecision(
            class_ids=ids,
            bounding_box_format=bounding_box_format,
            iou_thresholds=[0.75],
            name="MaP IoU=0.75",
        ),
        keras_cv.metrics.COCOMeanAveragePrecision(
            class_ids=ids,
            bounding_box_format=bounding_box_format,
            area_range=(0, 32**2),
            name="MaP Small Objects",
        ),
        keras_cv.metrics.COCOMeanAveragePrecision(
            class_ids=ids,
            bounding_box_format=bounding_box_format,
            area_range=(32**2, 96**2),
            name="MaP Medium Objects",
        ),
        keras_cv.metrics.COCOMeanAveragePrecision(
            class_ids=ids,
            bounding_box_format=bounding_box_format,
            area_range=(96**2, 1e9**2),
            name="MaP Large Objects",
        ),
        keras_cv.metrics.COCORecall(
            class_ids=ids,
            bounding_box_format=bounding_box_format,
            max_detections=1,
            name="Recall 1 Detection",
        ),
        keras_cv.metrics.COCORecall(
            class_ids=ids,
            bounding_box_format=bounding_box_format,
            max_detections=10,
            name="Recall 10 Detections",
        ),
        keras_cv.metrics.COCORecall(
            class_ids=ids,
            bounding_box_format=bounding_box_format,
            max_detections=100,
            name="Standard Recall",
        ),
        keras_cv.metrics.COCORecall(
            class_ids=ids,
            bounding_box_format=bounding_box_format,
            area_range=(0, 32**2),
            name="Recall Small Objects",
        ),
        keras_cv.metrics.COCORecall(
            class_ids=ids,
            bounding_box_format=bounding_box_format,
            area_range=(32**2, 96**2),
            name="Recall Medium Objects",
        ),
        keras_cv.metrics.COCORecall(
            class_ids=ids,
            bounding_box_format=bounding_box_format,
            area_range=(96**2, 1e9**2),
            name="Recall Large Objects",
        ),
    ]
    return metrics
