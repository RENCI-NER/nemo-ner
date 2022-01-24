import os

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import torch
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.collections.nlp.models import TokenClassificationModel
from nemo.utils import logging


@hydra_runner(config_path="config", config_name="train.yaml")
def train(cfg: DictConfig) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    if not cfg.pretrained_model:
        logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
        model = TokenClassificationModel(cfg.model, trainer=trainer)
    else:
        if os.path.exists(cfg.pretrained_model):

            model = TokenClassificationModel.restore_from(cfg.pretrained_model, trainer=trainer, strict=False)
        elif cfg.pretrained_model in TokenClassificationModel.get_available_model_names():
            model = TokenClassificationModel.from_pretrained(cfg.pretrained_model)
        else:
            raise ValueError(
                f'Provide path to the pre-trained .nemo file or choose from {TokenClassificationModel.list_available_models()}'
            )
        data_dir = cfg.model.dataset.get('data_dir', None)
        if data_dir:
            if not os.path.exists(data_dir):
                raise ValueError(f'{data_dir} is not found at')

            # we can also do finetuning of the pretrained model but it will require
            # setup the data dir to get class weights statistics
            model.update_data_dir(data_dir=data_dir)
            # finally, setup train and validation Pytorch DataLoaders
            model.setup_training_data()
            model.setup_validation_data()
            # then we're setting up loss, use model.dataset.class_balancing,
            # if you want to add class weights to the CrossEntropyLoss
            model.setup_loss(class_balancing=cfg.model.dataset.class_balancing)
            logging.info(f'Using config file of the pretrained model')
        else:
            raise ValueError(
                'Specify a valid dataset directory that contains test_ds.text_file and test_ds.labels_file \
                with "model.dataset.data_dir" argument'
            )
    trainer.fit(model)
    save_path = os.path.join(os.path.dirname(__file__), "..", "trained_model.nemo")
    print("saving")
    model.save_to(save_path)

