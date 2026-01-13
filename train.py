from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI

from srsc.datamodule import CROHMEDatamodule
from srsc.lit_srsc import LitSRSC

cli = LightningCLI(
    LitSRSC,
    CROHMEDatamodule,
    save_config_overwrite=True,
    trainer_defaults={"plugins": DDPPlugin(find_unused_parameters=True)},
)
