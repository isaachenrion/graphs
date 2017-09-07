import torch.nn as nn
import torch
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, model_output, G):
        pass

class GraphLoss(Loss):
    def __init__(self, graph_targets=None,  loss_fn=None, name=None, as_dict=False):
        super().__init__(name)
        self.graph_targets = graph_targets
        self.loss_fn = loss_fn
        self.as_dict = as_dict

    def forward(self, model_output, G):
        loss_dict = {}

        if self.graph_targets is not None:
            for target in self.graph_targets:
                loss_dict[target.name + ' ' + self.name] = self.loss_fn(model_output[target.name], G.graph[target.name])

        if self.as_dict:
            return loss_dict
        else:
            total_loss = 0.
            for target in self.graph_targets:
                total_loss += loss_dict[target.name + ' ' + self.name]
            return total_loss

class MSEGraphLoss(GraphLoss):
    def __init__(self, **kwargs):
        def mse(y, y_target):
            return torch.mean(torch.pow(y - y_target.squeeze(), 2))
        super().__init__(loss_fn=mse, name='MSE', **kwargs)

class MAEGraphLoss(GraphLoss):
    def __init__(self, **kwargs):
        def mae(y, y_target):
            return torch.mean(torch.abs(y - y_target.squeeze()))
        super().__init__(loss_fn=mae, name='MAE', **kwargs)

class BCEGraphLoss(GraphLoss):
    def __init__(self, **kwargs):
        super().__init__(loss_fn=F.binary_cross_entropy, name='BCE', **kwargs)

class BCEWithLogits(GraphLoss):
    def __init__(self, **kwargs):
        super().__init__(loss_fn=F.binary_cross_entropy_with_logits, name='BCE', **kwargs)

class CrossEntropy(GraphLoss):
    def __init__(self, **kwargs):
        super().__init__(loss_fn=F.cross_entropy, name='cross_entropy', **kwargs)


class R2(GraphLoss):
    def __init__(self, **kwargs):
        def r_squared(y, y_target):
            mse = torch.mean(torch.pow(y - y_target.squeeze(), 2))
            var = torch.var(y_target)
            R2 = 1 - (mse / var)
            return R2
        super().__init__(loss_fn=r_squared, name='R2', **kwargs)

class Accuracy(GraphLoss):
    def __init__(self, **kwargs):
        def accuracy(y, y_target):
            _, hard_prediction = torch.max(y, 1)
            acc = (hard_prediction == y_target) * 100
            return acc.float().mean()
        super().__init__(loss_fn=accuracy, name='accuracy', **kwargs)

class LossCollection(Loss):
    def __init__(self, primary_loss=None, other_losses=None):
        super().__init__(name='Collection')
        self.primary_loss = primary_loss
        self.other_losses = other_losses
        self.names = ['loss'] # for primary loss
        if self.other_losses is not None:
            for loss in self.other_losses:
                if loss.graph_targets is not None:
                    loss_names = [target.name + ' ' + loss.name for target in loss.graph_targets]
                self.names += loss_names

    def forward(self, model_output, G):
        out = {}
        if self.primary_loss is not None:
            assert not self.primary_loss.as_dict
            out['loss'] = self.primary_loss(model_output, G)
        for loss in self.other_losses:
            out.update(loss(model_output, G))
        return out
