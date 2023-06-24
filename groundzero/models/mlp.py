"""Multilayer perceptron (MLP) model implementation."""

# Imports PyTorch packages.
import torch
from torch import nn
from torch.nn.utils import prune
# Imports groundzero packages.
from groundzero.models.model import Model
import numpy as np
import copy
def relu():
    return nn.ReLU(inplace=False)

class MLP(Model):
    """MLP model implementation.
    
    This version has the same width at every hidden layer.
    """

    def __init__(self, args):
        """Initializes an MLP model.
        
        Args:
            args: The configuration dictionary.
        """

        super().__init__(args)
        
        activations = {"relu": relu, "sigmoid": nn.Sigmoid}
        activation = activations[args.mlp_activation]

        self.model = nn.Sequential()
        
        h = [args.mlp_hidden_dim] * (args.mlp_num_layers - 1)

        shapes = zip([args.mlp_input_dim] + h, h + [args.num_classes])
        for i, (n, k) in enumerate(shapes):
            if i == args.mlp_num_layers - 1:
                self.model.append(nn.Linear(n, k, bias=args.bias))
            else:
                self.model.append(nn.Linear(n, k, bias=args.bias))
                self.model.append(activation())
                # self.model.append(nn.Dropout(p=args.dropout_prob, inplace=False))
                
        if args.train_fc_only:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model[-1].parameters():
                p.requires_grad = True
        self.model_init = self.model.state_dict()
    def load_msg(self):
        return (
            f"Loading MLP with {self.hparams.mlp_num_layers} layers"
            f" and hidden dimension {self.hparams.mlp_hidden_dim}."
        )

    def forward(self, inputs):
        """Predicts using the model.

        Args:
            inputs: A torch.tensor of model inputs.
        
        Returns:
            The model prediction as a torch.tensor.
        """

        inputs = inputs.reshape(inputs.shape[0], -1)
        outputs = torch.squeeze(self.model(inputs), dim=-1)

        return outputs
    
    def prune_bottom_weights(self, percentage=10):
        for ori_name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Prune the weights
                prune.l1_unstructured(module, name='weight', amount=percentage/100)
                prune.l1_unstructured(module, 'bias', amount=percentage/100)

                weight_mask = copy.deepcopy(module.weight_mask.data)
                bias_mask = copy.deepcopy(module.bias_mask.data)
                prune.remove(module, name="weight")
                prune.remove(module, name="bias")
                # Copy the initial parameters for the unpruned weights
                module.load_state_dict({name: self.model_init[ori_name + "." + name] for name in module.state_dict()})
            
                module.weight.data = module.weight.data * weight_mask
                module.bias.data = module.bias.data * bias_mask