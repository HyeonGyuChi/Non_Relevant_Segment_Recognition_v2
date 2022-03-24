from core.model.timm_model import generate_timm_model


name_to_model = {
    'resnet18': generate_timm_model,
    'efficientnet_b0': generate_timm_model,
    'mobilenetv3_large_100': generate_timm_model,
    'ig_resnext101_32x48d': generate_timm_model, 
    'swin_large_patch4_window7_224': generate_timm_model, 
    'mobile_vit': generate_timm_model, 
}


def get_model(args):
    return name_to_model[args.model](args)