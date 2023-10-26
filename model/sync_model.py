import logging
import torch
import sys
from torch.nn import functional as F

sys.path.insert(0, '.')  # nopep8
from utils.utils import instantiate_from_config

logger = logging.getLogger(f'main.{__name__}')


class AVSyncModel(torch.nn.Module):

    def __init__(self, afeat_extractor, vfeat_extractor, a_bridge_cfg, v_bridge_cfg, transformer):
        super().__init__()
        self.vfeat_extractor = instantiate_from_config(vfeat_extractor)
        self.afeat_extractor = instantiate_from_config(afeat_extractor)
        # bridging the s3d latent dim (1024) into what is specified in the config
        # to match e.g. the transformer dim
        self.v_bridge = instantiate_from_config(v_bridge_cfg)
        self.a_bridge = instantiate_from_config(a_bridge_cfg)
        ## uncomment if doing crop start debugging
        self.transformer = instantiate_from_config(transformer)
        # Assuming we're using either both selectors and mixer or neither
        self.use_mixer = not transformer.params.mixed_selector_cfg.params.ablate_mixer

    def forward(self, vis: torch.Tensor, aud: torch.Tensor, targets: dict = None, return_attn_weights=False):
        '''
        Args:
            vis (torch.Tensor): RGB frames (B, Tv, C, H, W)
            aud (torch.Tensor): audio spectrograms (B, 1, F, Ta)
        Returns:
            tuple(Tensor, Tensor), tuple(Tensor, Tensor), Tensor: loss values, logits
        '''
        vis = self.vfeat_extractor(vis)
        aud = self.afeat_extractor(aud)

        vis = self.v_bridge(vis.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        aud = self.a_bridge(aud)

        if return_attn_weights:
            if self.use_mixer:
                logits, vsa1, asa1, vca1, aca1, vsa2, asa2, vca2, aca2 = self.transformer(vis, aud, return_attn_weights=return_attn_weight)
            else:
                logits, vsa1, asa1, vca1, aca1 = self.transformer(vis, aud, return_attn_weights=return_attn_weights)
        else:
            logits = self.transformer(vis, aud)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets['offset_target'])

        if return_attn_weights:
            if self.use_mixer:
                loss, logits, vsa1, asa1, vca1, aca1, vsa2, asa2, vca2, aca2
            else:
                return loss, logits, vsa1, asa1, vca1, aca1
        else:
            return loss, logits


if __name__ == '__main__':
    from omegaconf import OmegaConf
    from time import time

    cfg = OmegaConf.load('./configs/sparse_sync.yaml')
    cfg.training.use_half_precision = use_half_precision = False

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    model = instantiate_from_config(cfg.model)
    model = model.to(device)

    start_time = time()
    for i in range(3):
        vis = torch.rand(1, 125, 3, 224, 224, device=device)
        aud = torch.rand(1, 1, 257, 626, device=device)
        # cls_logits, off_logits, sync_logits = model(vis, aud)
        # inference in half precision
        with torch.cuda.amp.autocast(cfg.training.use_half_precision):
            out = model(vis, aud)
    print('Time:', round(time() - start_time, 3))
