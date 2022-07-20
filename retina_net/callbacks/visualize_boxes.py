import wandb
import tensorflow as tf


class VisualizeBoxes(tf.keras.callbacks.Callback):
    """
    The VisualizeBoxes:
      - logs the validation data or ground truth as W&B Tables,
      - performs inference to get model prediction `on_epoch_end`,
      - and logs the predictions as W&B Artifacts `on_epoch_end`,
      - it uses referencing thus data is uploaded just once.
    """

    def __init__(self, validation_data, dataset_info, **kwargs):

        super().__init__(**kwargs)
        self.validation_ds = validation_data
        self.int2str = dataset_info.features["objects"]["label"].int2str

        num_classes = len(self.int2str.keys())
        # A dictionary mapping class id to class label.
        self.class_id_to_label = {idx: self.int2str(idx) for idx in range(num_classes)}

        # When logging bounding boxes or segmentation masks along with W&B Tables,
        # a `wandb.Classes` instance is passed to `wandb.Image`.
        self.class_set = wandb.Classes(
            [
                {"id": idx, "name": label}
                for idx, label in self.class_id_to_label.items()
            ]
        )

    def on_train_begin(self, logs=None):
        # Initialize W&B table to log validation data
        self._init_data_table()
        # Add validation data to the table
        self._add_ground_truth()
        # Log the table to W&B
        self._log_data_table()

    def on_epoch_end(self, epoch, logs=None):
        # Initialize a prediction wandb table
        self._init_pred_table()
        # Add prediction to the table
        self._log_predictions()
        # Log the eval table to W&B
        self._log_eval_table(epoch)

    def _init_data_table(self):
        """Initialize the W&B Tables for validation data."""
        columns = ["image_name", "image"]
        self.data_table = wandb.Table(columns=columns, allow_mixed_types=True)

    def _init_pred_table(self):
        """Initialize the W&B Tables for model evaluation."""
        columns = ["image_name", "ground_truth", "prediction"]
        self.eval_table = wandb.Table(columns=columns)

    def _add_ground_truth(self):
        # Iterate through the samples and log them to the data_table.
        for i, (images, boxes) in enumerate(self.validation_ds):
            # Image identifier

            # Image
            image = sample["image"]
            assert sample["image"].ndim == 3

            # Get bbox and labels
            bboxes = sample["objects"]["bbox"].numpy()
            label_ids = sample["objects"]["label"].numpy()

            # Get dict of bounding boxes in the format required by `wandb.Image`.
            wandb_bboxes = {"ground_truth": self._get_wandb_bboxes(bboxes, label_ids)}

            # Log a row to the data table.
            self.data_table.add_data(
                image_name,
                wandb.Image(image, boxes=wandb_bboxes, classes=self.class_set),
            )

    def _log_predictions(self):
        # Get predicted detections
        detections, ratios = self._infer()
        print(detections)

        # Iterate through the samples.
        table_idxs = self.data_table_ref.get_index()
        for idx in table_idxs:
            detection, ratio = detections[idx], ratios[idx]
            num_detections = detection.valid_detections[0]
            pred_label_ids = [
                int(x) for x in detection.nmsed_classes[0][:num_detections]
            ]
            pred_bboxes = detection.nmsed_boxes[0][:num_detections] / ratio
            pred_scores = detection.nmsed_scores[0][:num_detections]

            # Get dict of bounding boxes in the format required by `wandb.Image`.
            wandb_bboxes = {
                "predictions": self._get_wandb_bboxes(
                    pred_bboxes, pred_label_ids, log_gt=False
                )
            }

            # Log a row to the eval table.
            self.eval_table.add_data(
                self.data_table_ref.data[idx][0],
                self.data_table_ref.data[idx][1],
                wandb.Image(
                    self.data_table_ref.data[idx][1],
                    boxes=wandb_bboxes,
                    classes=self.class_set,
                ),
            )

    def _infer(self):
        # Iterate through the samples.
        detections, ratios = [], []
        for i, sample in enumerate(self.validation_ds):
            image = tf.cast(sample["image"], dtype=tf.float32)
            input_image, ratio = self._prepare_image(image)
            predictions = self.model.inference(input_image)
            detections.append(predictions)
            ratios.append(ratio)

        return detections, ratios

    def _prepare_image(self, image):
        image, _, ratio = resize_and_pad_image(image, jitter=None)
        return tf.expand_dims(image, axis=0), ratio

    def _get_wandb_bboxes(self, bboxes, label_ids, log_gt=True, conf_scores=None):
        """
        Return a dict of bounding boxes in the format required by `wandb.Image`
        to log bounding boxes to W&B.

        To learn about the format check out the docs:
        https://docs.wandb.ai/guides/track/log/media#image-overlays
        """
        assert len(bboxes) == len(label_ids)

        box_data = []
        for i, (bbox, label_id) in enumerate(zip(bboxes, label_ids)):
            if log_gt:
                # corner configuration `(y1, x1, y2, x2)`
                position = dict(
                    minX=float(bbox[1]),
                    minY=float(bbox[0]),
                    maxX=float(bbox[3]),
                    maxY=float(bbox[2]),
                )
            else:
                # corner configuration `(x1, y1, x2, y2)`
                position = dict(
                    minX=float(bbox[0]),
                    minY=float(bbox[1]),
                    maxX=float(bbox[2]),
                    maxY=float(bbox[3]),
                )

            box_dict = {
                "position": position,
                "class_id": int(label_id),
                "box_caption": self.class_id_to_label[label_id],
            }

            if not log_gt:
                box_dict["domain"] = "pixel"
                if conf_scores is not None:
                    score = conf_scores[i]
                    caption = f"{self.class_id_to_label[label_id]}|{float(score)}"
                    box_dict["box_caption"] = caption

            box_data.append(box_dict)

        wandb_bboxes = {"box_data": box_data, "class_labels": self.class_id_to_label}

        return wandb_bboxes

    def _log_data_table(self):
        """Log the W&B Tables for validation data as artifact and calls
        `use_artifact` on it so that the evaluation table can use the reference
        of already uploaded images.
        This allows the data to be uploaded just once.
        """
        data_artifact = wandb.Artifact("val", type="dataset")
        data_artifact.add(self.data_table, "val_data")

        # Calling `use_artifact` uploads the data to W&B.
        wandb.run.use_artifact(data_artifact)
        data_artifact.wait()

        # We get the reference table.
        self.data_table_ref = data_artifact.get("val_data")

    def _log_eval_table(self, epoch):
        """Log the W&B Tables for model evaluation.
        The table will be logged multiple times creating new version. Use this
        to compare models at different intervals interactively.
        """
        pred_artifact = wandb.Artifact(f"run_{wandb.run.id}_pred", type="evaluation")
        pred_artifact.add(self.eval_table, "eval_data")
        wandb.run.log_artifact(pred_artifact)
