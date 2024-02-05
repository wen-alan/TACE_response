import torch
from torch.autograd.function import InplaceFunction
from itertools import repeat
import numpy as np
from collections import Counter

class Dropout(InplaceFunction):

    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @staticmethod
    def symbolic(g, input, p=0.5, train=False, inplace=False):
        # See Note [Export inplace]
        r, _ = g.op("Dropout", input, ratio_f=p, is_test_i=not train, outputs=2)
        return r

    @classmethod
    def forward(cls, ctx, input, p=0.5, train=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace
        distribute_num = 20

        if ctx.p == 0 or not ctx.train:
            return input

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        ctx.noise = cls._make_noise(input)
        if ctx.p == 1:
            ctx.noise.fill_(0)
        else:
            ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
        

        # inputSize0 = input.size(0)
        # inputSize1 = input.size(1)
        # ctx.noise = ctx.noise.view(inputSize0, -1)
        # dim1 = range(inputSize1)
        # input1 = input.detach().cpu().numpy()
        # for dim in range(inputSize0):
        #     max_input = np.max(input1[dim])
        #     min_input = np.min(input1[dim])
        #     feat = input1[dim] - min_input     #begin with 0
        #     max_min = max_input-min_input
        #     feat_dif = distribute_num/max_min
        #     # print('feat',feat)
        #     feat_index = (feat[dim1]*feat_dif).astype(int)  #feature distribute index
        #     feat_index[np.where(distribute_num == feat_index)] = 19
            
        #     feat_num = np.zeros(distribute_num)
        #     # feat_num[feat_index] += 1
        #     feat_dict = Counter(feat_index) #get distribution
        #     # print('feat_dict',feat_dict)
        #     feat_num[list(feat_dict.keys())] = list(feat_dict.values())
        #     after_sorted = sorted(feat_num)
        #     biggest_index = np.where(feat_num == after_sorted[-1])
        #     # print('after_sorted',after_sorted)        
        #     # feat_num = feat_num / after_sorted[-2] * 0.9
        #     # feat_num[biggest_index] = 0.95
        #     print('feat_num',feat_num)



        ctx.noise = ctx.noise.expand_as(input)
        output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output * ctx.noise, None, None, None
        else:
            return grad_output, None, None, None


class FeatureDropout(Dropout):

    @staticmethod
    def symbolic(g, input, p=0.5, train=False, inplace=False):
        # See Note [Export inplace]
        # NB: In inference mode, FeatureDropout is exported as an identity op.
        from torch.onnx.symbolic import _unimplemented
        if train:
            return _unimplemented("FeatureDropout", "training mode")
        return input

    @staticmethod
    def _make_noise(input):
        return input.new().resize_(input.size(0), input.size(1),
                                   *repeat(1, input.dim() - 2))


class AlphaDropout(Dropout):

    @staticmethod
    def symbolic(g, input, p=0.5, train=False, inplace=False):
        # See Note [Export inplace]
        # NB: In inference mode, FeatureDropout is exported as an identity op.
        from torch.onnx.symbolic import _unimplemented
        if train:
            return _unimplemented("AlphaDropout", "training mode")
        return input

    @classmethod
    def forward(cls, ctx, input, p=0.5, train=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.p == 0 or not ctx.train:
            return input

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        ctx.noise = cls._make_noise(input)
        if ctx.p == 1:
            a = 0
            b = ctx.noise
        else:
            ctx.noise.bernoulli_(1 - ctx.p)
            alpha = 1.7580993408473766
            a = ((alpha ** 2 * ctx.p + 1) * (1 - ctx.p)) ** (-0.5)
            b = ctx.noise.add(-1).mul_(alpha * a).add_(alpha * a * ctx.p)
        ctx.noise = ctx.noise.mul_(a).expand_as(input)
        b = b.expand_as(input)
        output.mul_(ctx.noise).add_(b)

        return output


class FeatureAlphaDropout(AlphaDropout):

    @staticmethod
    def symbolic(g, input, p=0.5, train=False, inplace=False):
        # See Note [Export inplace]
        # NB: In inference mode, FeatureDropout is exported as an identity op.
        from torch.onnx.symbolic import _unimplemented
        if train:
            return _unimplemented("FeatureAlphaDropout", "training mode")
        return input

    @staticmethod
    def _make_noise(input):
        return input.new().resize_(input.size(0), input.size(1),
                                   *repeat(1, input.dim() - 2))