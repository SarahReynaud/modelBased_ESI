### Script to train a model from the hydra configuration
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Tuple
import torch
import hydra
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torch import nn

os.environ['HYDRA_FULL_ERROR'] = "1"


# use the hydra decorator to use the configuration file
@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg:DictConfig) : 
    torch.cuda.empty_cache()
    pl.seed_everything(333) # seed for reproducibility
    ## DATA ##
    dm = hydra.utils.call(cfg.datamodule)
    dm.setup("train")
    # train_dl, val_dl = dm.train_dataloader(), dm.val_dataloader()
    # print(f"{next(iter(val_dl))[0].shape=}")

    litmodel = hydra.utils.call(cfg.litmodel)
    for p in litmodel.solver.prior_cost.parameters(): 
        print(p.mean())

    trainer = hydra.utils.call(cfg.trainer)
    
    start = time.time()
    trainer.fit(litmodel, dm)
    end = time.time()

    # write training time
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    with open( Path("./training_times.txt" ) , "a") as f: 
        f.write(f"{output_dir} :\n")
        f.write(f"Training time: {(end-start)/3600:.3f} hour \n") 
        f.write("______________________________________________________\n")
    trainer.save_checkpoint( Path(output_dir, "last_epoch.ckpt") ) ## also save last epoch checkpoint
if __name__ == "__main__": 
    
    main()