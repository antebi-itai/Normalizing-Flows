import os
import time
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from config import PL_TRAINER_PATH, trained_filepath, epochs


def train_flow(flow, train_loader, val_loader, test_loader, model_name="MNISTFlow", pretrained=True):
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(default_root_dir=os.path.join(PL_TRAINER_PATH, model_name),
                         gpus=1,
                         max_epochs=epochs,
                         gradient_clip_val=1.0,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_bpd"),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    result = None

    # Check whether pretrained model exists. If yes, load it and skip training
    if pretrained:
        assert os.path.isfile(trained_filepath), "Pretrained model .ckpt file not found"
        print("Loading pretrained model...")
        ckpt = torch.load(trained_filepath)
        flow.load_state_dict(ckpt['state_dict'])
        result = ckpt.get("result", None)
    else:
        print("Starting training", f"{model_name}...")
        trainer.fit(flow, train_loader, val_loader)

    # Test best model on validation and test set if no result has been found
    # Testing can be expensive due to the importance sampling.
    if result is None:
        val_result = trainer.test(flow, val_loader, verbose=False)
        start_time = time.time()
        test_result = trainer.test(flow, test_loader, verbose=False)
        duration = time.time() - start_time
        result = {"test": test_result, "val": val_result, "time": duration / len(test_loader) / flow.import_samples}

    print("Done Training Network!")
    return flow, result
