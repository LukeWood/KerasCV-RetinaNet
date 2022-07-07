import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.wandb_project_name = "KerasCV-RetinaNet"
    config.data_path = "data/"

    config.batch_size = 2
    config.num_classes = 1
    config.input_shape = (720, 1280, 3)

    config.epochs = 300
    config.steps_per_epoch = 1000
    config.validation_steps = 300

    config.metrics = "basic"

    return config
