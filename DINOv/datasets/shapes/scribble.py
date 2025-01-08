import random

import torch

from .mask_generators import get_mask_by_input_strokes


class Scribble:
    def __init__(self, cfg, is_train):
        self.num_stroke = cfg['STROKE_SAMPLER']['SCRIBBLE']['NUM_STROKES']
        self.stroke_preset = cfg['STROKE_SAMPLER']['SCRIBBLE']['STROKE_PRESET']
        self.stroke_prob = cfg['STROKE_SAMPLER']['SCRIBBLE']['STROKE_PROB']
        self.eval_stroke = cfg['STROKE_SAMPLER']['EVAL']['MAX_ITER']
        self.is_train = is_train

    @staticmethod
    def get_stroke_preset(stroke_preset):
        if stroke_preset == 'rand_curve':
            return {
                "nVertexBound": [20, 50],
                "maxHeadSpeed": 30,
                "maxHeadAcceleration": (30, 0.5),
                "brushWidthBound": (3, 15),
                "nMovePointRatio": 0.5,
                "maxPiontMove": 6,
                "maxLineAcceleration": (9, 0.5),
                "boarderGap": None,
                "maxInitSpeed": 10
            }
        elif stroke_preset == 'rand_curve_small':
            return {
                "nVertexBound": [6, 22],
                "maxHeadSpeed": 12,
                "maxHeadAcceleration": (8, 0.5),
                "brushWidthBound": (2.5, 5),
                "nMovePointRatio": 0.5,
                "maxPiontMove": 1.5,
                "maxLineAcceleration": (3, 0.5),
                "boarderGap": None,
                "maxInitSpeed": 3
            }
        else:
            raise NotImplementedError(f'The stroke presetting "{stroke_preset}" does not exist.')

    def get_random_points_from_mask(self, mask, n=5):
        h,w = mask.shape
        view_mask = mask.reshape(h*w)
        non_zero_idx = view_mask.nonzero()[:,0]
        selected_idx = torch.randperm(len(non_zero_idx))[:n]
        non_zero_idx = non_zero_idx[selected_idx]
        y = (non_zero_idx // w)*1.0
        x = (non_zero_idx % w)*1.0
        return torch.cat((x[:,None], y[:,None]), dim=1).cpu().numpy()

    def draw(self, mask=None, box=None):
        if mask.sum() < 1:
            return torch.zeros(mask.shape).bool().cuda() # if mask is empty
        if not self.is_train:
            return self.draw_eval(mask=mask, box=box)
        stroke_preset_name = random.choices(self.stroke_preset, weights=self.stroke_prob, k=1)[0]
        preset = Scribble.get_stroke_preset(stroke_preset_name)
        nStroke = random.randint(1, min(self.num_stroke, mask.sum().item()))
        h,w = mask.shape
        points = self.get_random_points_from_mask(mask, n=nStroke)
        rand_mask = get_mask_by_input_strokes(
            init_points=points,
            imageWidth=w, imageHeight=h, nStroke=min(nStroke, len(points)), **preset)
        rand_mask = (~torch.from_numpy(rand_mask)).cuda() * mask
        return rand_mask

    def draw_eval(self, mask=None, box=None):
        stroke_preset_name = random.choices(self.stroke_preset, weights=self.stroke_prob, k=1)[0]
        preset = Scribble.get_stroke_preset(stroke_preset_name)
        nStroke = min(self.eval_stroke, mask.sum().item())
        h,w = mask.shape
        points = self.get_random_points_from_mask(mask, n=nStroke)
        rand_masks = []
        for i in range(len(points)):
            rand_mask = get_mask_by_input_strokes(
                init_points=points[:i+1],
                imageWidth=w, imageHeight=h, nStroke=min(i, len(points)), **preset)
            rand_mask = (~torch.from_numpy(rand_mask)).cuda() * mask
            rand_masks += [rand_mask]
        return torch.stack(rand_masks)

    @staticmethod
    def draw_by_points(points, mask, h, w):
        preset = Scribble.get_stroke_preset('rand_curve_small')
        rand_mask = get_mask_by_input_strokes(
            init_points=points,
            imageWidth=w, imageHeight=h, nStroke=len(points), **preset)[None,]
        rand_masks = (~torch.from_numpy(rand_mask)) * mask
        return rand_masks

    def __repr__(self,):
        return 'scribble'