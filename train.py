import os
import time
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


def train_flow(flow, train_loader, val_loader, test_loader, config):
    print(f"Starting to train model {config.model_name} on dataset {config.dataset} "
          f"size {config.size} for {config.epochs} epochs...")
    # Create a PyTorch Lightning trainer
    logger = WandbLogger(project="Normalizing_Flows")
    trainer = pl.Trainer(logger=logger,
                         gpus=1,
                         max_epochs=config.epochs,
                         gradient_clip_val=1.0,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_bpd"),
                                    LearningRateMonitor("epoch")])

    # Train
    trainer.fit(flow, train_loader, val_loader)

    # Test best model on validation and test set if no result has been found
    # Testing can be expensive due to the importance sampling.
    val_result = trainer.test(flow, val_loader, verbose=False)
    start_time = time.time()
    test_result = trainer.test(flow, test_loader, verbose=False)
    duration = time.time() - start_time
    result = {"test": test_result, "val": val_result, "time": duration / len(test_loader) / flow.import_samples}

    print("Done Training Network!")
    return flow, result


def load_flow(flow, config):
    print(f"Loading model from {config.trained_filepath}...")
    assert os.path.isfile(config.trained_filepath), f"Model file {config.trained_filepath} not found"
    ckpt = torch.load(config.trained_filepath)
    flow.load_state_dict(ckpt['state_dict'])
    result = ckpt.get("result", None)
    assert result is not None

    print("Done Loading Network!")
    return flow, result
