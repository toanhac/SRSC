import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import typer
from pytorch_lightning import Trainer, seed_everything

from srsc.datamodule import CROHMEDatamodule
from srsc.lit_srsc import LitSRSC

seed_everything(7)


def main(version: str, test_year: str):
    ckp_folder = os.path.join("lightning_logs", f"version_{version}", "checkpoints")
    fnames = [f for f in os.listdir(ckp_folder) if f.endswith(".ckpt")]
    if len(fnames) == 1:
        ckp_path = os.path.join(ckp_folder, fnames[0])
    else:
        best_val, best_f = -1.0, None
        for f in fnames:
            m = re.search(r"val_ExpRate[=_](\d*\.?\d+)", f)
            if m and float(m.group(1)) > best_val:
                best_val = float(m.group(1))
                best_f = f
        ckp_path = os.path.join(ckp_folder, best_f or fnames[0])
    print(f"Test with: {os.path.basename(ckp_path)}")

    dm = CROHMEDatamodule(test_year=test_year, eval_batch_size=4, use_relation_maps=False)
    model = LitSRSC.load_from_checkpoint(ckp_path)
    trainer = Trainer(logger=False, accelerator="gpu", devices=1)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    typer.run(main)
