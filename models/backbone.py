import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def _decoder_out_channels(model) -> int:
    """Infer channel dimension of decoder output (input to segmentation head)."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        # Large enough spatial size so low-resolution heads (e.g. DeepLab ASPP) are not 1x1 (avoids BN issues).
        x = torch.zeros(1, 3, 512, 512)
        feats = model.encoder(x)
        dec = model.decoder(feats)
    if was_training:
        model.train()
    return int(dec.shape[1])


class SMPUNetWrapper(nn.Module):
    """
    U-Net with configurable encoder (default ResNet-50, ImageNet pre-trained).
    """

    def __init__(self, encoder_name='resnet50', encoder_weights='imagenet', in_channels=3, classes=1):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        self.feature_channels = _decoder_out_channels(self.model)

    def forward(self, x):
        features = self.model.encoder(x)
        decoder_output = self.model.decoder(features)
        logits = self.model.segmentation_head(decoder_output)
        prob = torch.sigmoid(logits)
        return logits, prob, decoder_output


class SMPDeepLabV3PlusWrapper(nn.Module):
    """
    DeepLabV3+ with configurable encoder; decoder output feeds Rejector like U-Net.
    """

    def __init__(self, encoder_name='resnet50', encoder_weights='imagenet', in_channels=3, classes=1):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        self.feature_channels = _decoder_out_channels(self.model)

    def forward(self, x):
        features = self.model.encoder(x)
        decoder_output = self.model.decoder(features)
        logits = self.model.segmentation_head(decoder_output)
        prob = torch.sigmoid(logits)
        return logits, prob, decoder_output


def get_backbone(
    model_name='unet',
    encoder_name='resnet50',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,
):
    """
    Args:
        model_name: 'unet' or 'deeplabv3plus'.
        encoder_name: SMP encoder (e.g. resnet50).
        encoder_weights: 'imagenet' or None for random init.
        in_channels / classes: passed to SMP.
    """
    kwargs = dict(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
    if model_name == 'unet':
        return SMPUNetWrapper(**kwargs)
    if model_name in ('deeplabv3plus', 'deeplabv3+'):
        return SMPDeepLabV3PlusWrapper(**kwargs)
    raise NotImplementedError(f"Backbone {model_name} is not implemented.")
