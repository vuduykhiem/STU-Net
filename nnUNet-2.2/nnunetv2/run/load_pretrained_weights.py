import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP


def load_pretrained_weights(network, fname, verbose=False):
    """
    Transfers all weights between matching keys in state_dicts. matching is done by name and we only transfer if the
    shape is also the same. Segmentation layers (the 1x1(x1) layers that produce the segmentation maps)
    identified by keys ending with '.seg_layers') are not transferred!

    If the pretrained weights were optained with a training outside nnU-Net and DDP or torch.optimize was used,
    you need to change the keys of the pretrained state_dict. DDP adds a 'module.' prefix and torch.optim adds
    '_orig_mod'. You DO NOT need to worry about this if pretraining was done with nnU-Net as
    nnUNetTrainer.save_checkpoint takes care of that!

    """
    saved_model = torch.load(fname, weights_only=False)
    # nnUNetv2 checkpoints use 'network_weights'; nnUNetv1 / STU-Net pretrained
    # weights use 'state_dict'. Accept both.
    if 'network_weights' in saved_model:
        pretrained_dict = saved_model['network_weights']
    elif 'state_dict' in saved_model:
        pretrained_dict = saved_model['state_dict']
    else:
        raise KeyError(
            f"Pretrained weights file has neither 'network_weights' nor 'state_dict' key. "
            f"Found keys: {list(saved_model.keys())}"
        )

    skip_strings_in_pretrained = [
        '.seg_layers.',
    ]

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()
    # Log mismatches instead of asserting — shape mismatches (e.g. first conv
    # layer when pretrained on 1-channel CT but finetuning on 2-channel input)
    # are skipped and those layers keep their random initialisation.
    for key, _ in model_dict.items():
        if all([i not in key for i in skip_strings_in_pretrained]):
            if key not in pretrained_dict:
                print(f"  [load_pretrained] SKIP (missing in pretrained): {key}")
            elif model_dict[key].shape != pretrained_dict[key].shape:
                print(f"  [load_pretrained] SKIP (shape mismatch): {key} "
                      f"pretrained={pretrained_dict[key].shape} model={model_dict[key].shape}")

    # fun fact: in principle this allows loading from parameters that do not cover the entire network. For example pretrained
    # encoders. Not supported by this function though (see assertions above)

    # commenting out this abomination of a dict comprehension for preservation in the archives of 'what not to do'
    # pretrained_dict = {'module.' + k if is_ddp else k: v
    #                    for k, v in pretrained_dict.items()
    #                    if (('module.' + k if is_ddp else k) in model_dict) and
    #                    all([i not in k for i in skip_strings_in_pretrained])}

    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict.keys()
                       and model_dict[k].shape == v.shape
                       and all([i not in k for i in skip_strings_in_pretrained])}

    model_dict.update(pretrained_dict)

    print("################### Loading pretrained weights from file ", fname, '###################')
    if verbose:
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, value in pretrained_dict.items():
            print(key, 'shape', value.shape)
        print("################### Done ###################")
    mod.load_state_dict(model_dict)


