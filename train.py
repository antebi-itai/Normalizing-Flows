import os
import time
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from config import PL_TRAINER_PATH, trained_filepath, epochs


def train_flow(flow, train_loader, val_loader, test_loader, model_name="MNISTFlow"):
    print(f"Starting to train model {model_name} for {epochs} epochs...")
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(default_root_dir=os.path.join(PL_TRAINER_PATH, model_name),
                         gpus=1,
                         max_epochs=epochs,
                         gradient_clip_val=1.0,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_bpd"),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

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


def load_flow(flow):
    print(f"Loading model from {trained_filepath}...")
    assert os.path.isfile(trained_filepath), f"Model file {trained_filepath} not found"
    ckpt = torch.load(trained_filepath)
    flow.load_state_dict(ckpt['state_dict'])
    result = ckpt.get("result", None)
    assert result is not None

    print("Done Loading Network!")
    return flow, result
