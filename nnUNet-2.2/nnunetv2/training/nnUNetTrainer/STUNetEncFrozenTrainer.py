from nnunetv2.training.nnUNetTrainer.STUNetTrainer import STUNetTrainer_small_ft, STUNetTrainer_base_ft, STUNet
import torch
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler


# model = STUNet(input_channels=2, num_classes=2, 
#                pool_op_kernel_sizes=[[2,2,2]]*5,
#                conv_kernel_sizes=[[3,3,3]]*6)

# for name, param in model.named_parameters():
#     print(name, param.shape, param.requires_grad)

# for param in model.conv_blocks_context.parameters():
#     param.requires_grad = False

# for param in model.conv_blocks_context.parameters():
#     print(param.shape, param.requires_grad)


class STUNetEncFrozenTrainer_small_ft(STUNetTrainer_small_ft):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250
        self.initial_lr = 1e-3

    def configure_optimizers(self):
        for param in self.network.conv_blocks_context.parameters():
            param.requires_grad = False
        decoder_params = (
        list(self.network.conv_blocks_localization.parameters()) +
        list(self.network.upsample_layers.parameters()) +
        list(self.network.seg_outputs.parameters())
        )
        optimizer = torch.optim.SGD(decoder_params, self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

        return optimizer, lr_scheduler
    

class STUNetEncFrozenTrainer_base_ft(STUNetTrainer_base_ft):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250
        self.initial_lr = 1e-3

    def configure_optimizers(self):
        for param in self.network.conv_blocks_context.parameters():
            param.requires_grad = False
        decoder_params = (
        list(self.network.conv_blocks_localization.parameters()) +
        list(self.network.upsample_layers.parameters()) +
        list(self.network.seg_outputs.parameters())
        )
        optimizer = torch.optim.SGD(decoder_params, self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

        return optimizer, lr_scheduler

    


        

