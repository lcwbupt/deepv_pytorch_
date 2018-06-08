from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        self.image_height = cfg['image_height']
        self.image_width = cfg['image_width']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            print(f)
            for i, j in product(range(f[0]), range(f[1])):
                
                f_k_w = self.image_width / self.steps[k]
                f_k_h = self.image_height / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k_w
                cy = (i + 0.5) / f_k_h

                # aspect_ratio: 1
                # rel size: min_size
                if isinstance(self.min_sizes[k],list):
                    for m in range(len(self.min_sizes[k])):
                        s_k_w = self.min_sizes[k][m]/self.image_width
                        s_k_h = self.min_sizes[k][m]/self.image_height
                        mean += [cx, cy, s_k_w, s_k_h]
         
                        # aspect_ratio: 1
                        # rel size: sqrt(s_k * s_(k+1))
                        if len(self.max_sizes) > 0:
                            s_k_prime_w = sqrt(s_k_w * (self.max_sizes[k][m]/self.image_width))
                            s_k_prime_h = sqrt(s_k_h * (self.max_sizes[k][m]/self.image_height))
                            mean += [cx, cy, s_k_prime_w, s_k_prime_h]
     
                        # rest of aspect ratios
                        for ar in self.aspect_ratios[k]:
                            mean += [cx, cy, s_k_w*sqrt(ar), s_k_h/sqrt(ar)]
                            mean += [cx, cy, s_k_w/sqrt(ar), s_k_h*sqrt(ar)]
                else:
                    s_k_w = self.min_sizes[k]/self.image_width
                    s_k_h = self.min_sizes[k]/self.image_height
                    mean += [cx, cy, s_k_w, s_k_h]
    
                    # aspect_ratio: 1
                    # rel size: sqrt(s_k * s_(k+1))
                    if len(self.max_sizes) > 0:
                        s_k_prime_w = sqrt(s_k_w * (self.max_sizes[k]/self.image_width))
                        s_k_prime_h = sqrt(s_k_h * (self.max_sizes[k]/self.image_height))
                        mean += [cx, cy, s_k_prime_w, s_k_prime_h]

                    # rest of aspect ratios
                    for ar in self.aspect_ratios[k]:
                        mean += [cx, cy, s_k_w*sqrt(ar), s_k_h/sqrt(ar)]
                        mean += [cx, cy, s_k_w/sqrt(ar), s_k_h*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
