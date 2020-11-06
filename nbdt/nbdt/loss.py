import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from nbdt.data.custom import Node, dataset_to_dummy_classes
from nbdt.model import HardEmbeddedDecisionRules, SoftEmbeddedDecisionRules
from nbdt.utils import (
    Colors, dataset_to_default_path_graph, dataset_to_default_path_wnids,
    hierarchy_to_path_graph
)
import numpy as np

__all__ = names = ('HardTreeSupLoss', 'SoftTreeSupLoss', 'CrossEntropyLoss')
keys = (
    'path_graph', 'path_wnids', 'tree_supervision_weight',
    'classes', 'dataset', 'criterion'
)

def add_arguments(parser):
    parser.add_argument('--hierarchy',
                        help='Hierarchy to use. If supplied, will be used to '
                        'generate --path-graph. --path-graph takes precedence.')
    parser.add_argument('--path-graph', help='Path to graph-*.json file.')  # WARNING: hard-coded suffix -build in generate_fname
    parser.add_argument('--path-wnids', help='Path to wnids.txt file.')
    parser.add_argument('--tree-supervision-weight', type=float, default=1,
                        help='Weight assigned to tree supervision losses')


def set_default_values(args):
    assert not (args.hierarchy and args.path_graph), \
        'Only one, between --hierarchy and --path-graph can be provided.'
    if 'TreeSupLoss' not in args.loss:
        return
    if args.hierarchy and not args.path_graph:
        args.path_graph = hierarchy_to_path_graph(args.dataset, args.hierarchy)
    if not args.path_graph:
        args.path_graph = dataset_to_default_path_graph(args.dataset)
    if not args.path_wnids:
        args.path_wnids = dataset_to_default_path_wnids(args.dataset)


CrossEntropyLoss = nn.CrossEntropyLoss


class TreeSupLoss(nn.Module):

    accepts_criterion = lambda criterion, **kwargs: criterion
    accepts_dataset = lambda trainset, **kwargs: trainset.__class__.__name__
    accepts_path_graph = True
    accepts_path_wnids = True
    accepts_classes = True
    accepts_tree_supervision_weight = True
    accepts_classes = lambda trainset, **kwargs: trainset.classes

    def __init__(self,
            dataset,
            criterion,
            sample_nums,
            path_graph=None,
            path_wnids=None,
            classes=None,
            hierarchy=None,
            Rules=HardEmbeddedDecisionRules,
            **kwargs):
        super().__init__()

        if dataset and hierarchy and not path_graph:
            path_graph = hierarchy_to_path_graph(dataset, hierarchy)
        if dataset and not path_graph:
            path_graph = dataset_to_default_path_graph(dataset)
        if dataset and not path_wnids:
            path_wnids = dataset_to_default_path_wnids(dataset)
        if dataset and not classes:
            classes = dataset_to_dummy_classes(dataset)

        assert len(sample_nums) == len(classes)
        self.init(dataset, criterion, sample_nums, path_graph, path_wnids, classes,
            Rules=Rules, **kwargs)

    def init(self,
            dataset,
            criterion,
            sample_nums,
            path_graph,
            path_wnids,
            classes,
            Rules,
            tree_supervision_weight=1.):
        """
        Extra init method makes clear which arguments are finally necessary for
        this class to function. The constructor for this class may generate
        some of these required arguments if initially missing.
        """
        self.dataset = dataset
        self.num_classes = len(classes)
        self.nodes = Node.get_nodes(path_graph, path_wnids, classes)
        self.rules = Rules(dataset, path_graph, path_wnids, classes)
        self.tree_supervision_weight = tree_supervision_weight
        self.criterion = criterion
        self.sample_nums = np.array(sample_nums)
        self.node_depths = defaultdict(lambda: [])
        self.node_weights = defaultdict(lambda: [])
        effective_num = 1.0 - np.power(0.999, self.sample_nums)
        weights = (1.0 - 0.999) / np.array(effective_num)
        self.weights = weights
        for node in self.nodes:
            key = node.num_classes
            depth = node.get_depth()
            self.node_depths[key].append(depth)
            node_weight = []
            for new_label in range(node.num_classes):
                node_weight.append(weights[node.new_to_old_classes[new_label]])
            self.node_weights[key].append(node_weight)

    @staticmethod
    def assert_output_not_nbdt(outputs):
        """
        >>> x = torch.randn(1, 3, 224, 224)
        >>> TreeSupLoss.assert_output_not_nbdt(x)  # all good!
        >>> x._nbdt_output_flag = True
        >>> TreeSupLoss.assert_output_not_nbdt(x)  #doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        AssertionError: ...
        >>> from nbdt.model import NBDT
        >>> import torchvision.models as models
        >>> model = models.resnet18()
        >>> y = model(x)
        >>> TreeSupLoss.assert_output_not_nbdt(y)  # all good!
        >>> model = NBDT('CIFAR10', model, arch='ResNet18')
        >>> y = model(x)
        >>> TreeSupLoss.assert_output_not_nbdt(y)  #doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        AssertionError: ...
        """
        assert getattr(outputs, '_nbdt_output_flag', False) is False, (
            "Uh oh! Looks like you passed an NBDT model's output to an NBDT "
            "loss. NBDT losses are designed to take in the *original* model's "
            "outputs, as input. NBDT models are designed to only be used "
            "during validation and inference, not during training. Confused? "
            " Check out github.com/alvinwan/nbdt#convert-neural-networks-to-decision-trees"
            " for examples and instructions.")


class HardTreeSupLoss(TreeSupLoss):

    def calculate_weight(self, weight, key, inds, mode):
        weight_ = []
        if mode == 'mean':
            for wei in weight:
                weight_.append(np.mean(wei))
            weight_ = weight_ / np.sum(weight_) * key
            weight_ = torch.tensor(weight_).float()
        elif mode == 'sum':
            for wei in weight:
                weight_.append(np.sum(wei))
            weight_ = weight_ / np.sum(weight_) * key
            weight_ = torch.tensor(weight_).float()
        elif mode == 'max':
            inds = np.array(inds).T
            weight_ = self.weights[inds]
            if weight_.ndim == 1:
                weight_ = weight_[np.newaxis, :]
            weight_sum = np.repeat(np.reshape(np.sum(weight_, axis=1), (-1,1)), key, axis= 1)
            weight_ = weight_ / weight_sum * key
            weight_ = torch.tensor(weight_).float()

        return weight_

    def forward(self, outputs, targets, mode='mean'):
        """
        The supplementary losses are all uniformly down-weighted so that on
        average, each sample incurs half of its loss from standard cross entropy
        and half of its loss from all nodes.

        The code below is structured weirdly to minimize number of tensors
        constructed and moved from CPU to GPU or vice versa. In short,
        all outputs and targets for nodes with 2 children are gathered and
        moved onto GPU at once. Same with those with 3, with 4 etc. On CIFAR10,
        the max is 2. On CIFAR100, the max is 8.
        """
        self.assert_output_not_nbdt(outputs)

        fg_idx = (targets!=0)
        outputs_ = outputs[fg_idx, 1:]
        targets_ = (targets[fg_idx] - 1)

        losses = []
        num_losses = outputs_.size(0) * len(self.nodes) / 2.

        outputs_subs = defaultdict(lambda: [])
        targets_subs = defaultdict(lambda: [])
        inds_subs = defaultdict(lambda: [])
        targets_ints = [int(target) for target in targets_.cpu().long()]
        for node in self.nodes:
            _, outputs_sub, targets_sub, inds = \
                HardEmbeddedDecisionRules.get_node_logits_filtered(
                    node, outputs_, targets_ints, mode)
            key = node.num_classes
            assert outputs_sub.size(0) == len(targets_sub)
            outputs_subs[key].append(outputs_sub)
            targets_subs[key].append(targets_sub)
            inds_subs[key].append(inds)
        for key in outputs_subs:
            outputs_sub_ = torch.cat(outputs_subs[key], dim=0)

            assert len(outputs_subs[key]) == len(targets_subs[key])
            assert len(self.node_depths[key]) == len(outputs_subs[key])
            assert len(self.node_weights[key]) == len(outputs_subs[key])
            if not outputs_sub_.size(0):
                continue
            fraction = outputs_sub_.size(0) / float(num_losses) \
                       * self.tree_supervision_weight

            losses_sub = []
            for (outputs_sub, targets_sub, inds, depth, weight) in zip(outputs_subs[key], targets_subs[key], inds_subs[key], self.node_depths[key], self.node_weights[key]):
                if len(targets_sub):

                    weight_ = self.calculate_weight(weight, key, inds, mode)
                    weight_ = weight_.to(outputs_sub.device)
                    targets_sub = torch.Tensor(targets_sub).long().to(outputs_sub.device)
                    if mode == 'mean' or mode == 'sum':
                        temp = F.cross_entropy(input=outputs_sub, target=targets_sub, weight=weight_)
                    else:
                        targets_sub = F.one_hot(targets_sub, key).float()
                        temp = F.binary_cross_entropy_with_logits(input=outputs_sub, target=targets_sub,
                                                      weight=weight_)
                    losses_sub.append(temp * (depth / 10.0 + 1))
            losses_sub = sum(losses_sub) / len(losses_sub)
            losses.append(losses_sub * fraction)
        loss = sum(losses)
        return loss


class SoftTreeSupLoss(TreeSupLoss):

    def __init__(self, *args, Rules=None, **kwargs):
        super().__init__(*args, Rules=SoftEmbeddedDecisionRules, **kwargs)

    def forward(self, outputs, targets):
        self.assert_output_not_nbdt(outputs)

        fg_idx = (targets != 0)
        outputs_ = outputs[fg_idx, 1:]
        targets_ = (targets[fg_idx] - 1)
        bayesian_outputs = self.rules(outputs_)
        loss = self.criterion(bayesian_outputs, targets_) * self.tree_supervision_weight
        return loss
