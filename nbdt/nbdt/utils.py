'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import numpy as np

from urllib.request import urlopen, Request
from PIL import Image
import torch.nn as nn
import torch.nn.init as init
from pathlib import Path
import io
import nltk

# tree-generation consntants
METHODS = ('wordnet', 'random', 'induced')
DATASETS = ('CIFAR10', 'CIFAR100', 'TinyImagenet200', 'Imagenet1000', 'VG150', 'VG150_head', 'VG150_head1', 'VG150_head2', 'VG150_head3')
DATASET_TO_NUM_CLASSES = {
    'CIFAR10': 10,
    'CIFAR100': 100,
    'TinyImagenet200': 200,
    'Imagenet1000': 1000,
    'VG150': 50
}
DATASET_TO_CLASSES = {
    'CIFAR10': [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
        'horse', 'ship', 'truck'
    ],
    'VG150': ['above', 'across', 'against', 'along',
              'and', 'at', 'attached to', 'behind', 'belonging to',
              'between','carrying', 'covered in', 'covering', 'eating',
              'flying in', 'for', 'from', 'growing on', 'hanging from',
              'has', 'holding', 'in', 'in front of', 'laying on',
              'looking at', 'lying on', 'made of', 'mounted on', 'near',
              'of', 'on', 'on back of', 'over', 'painted on',
              'parked on', 'part of', 'playing', 'riding', 'says',
              'sitting on', 'standing on', 'to', 'under', 'using',
              'walking in', 'walking on', 'watching', 'wearing', 'wears',
              'with']
}


def maybe_install_wordnet():
    try:
        nltk.data.find('corpora/wordnet')
    except Exception as e:
        print(e)
        nltk.download('wordnet')


def fwd():
    """Get file's working directory"""
    return Path(__file__).parent.absolute()


def dataset_to_default_path_graph(dataset):
    return hierarchy_to_path_graph(dataset, 'induced')


def hierarchy_to_path_graph(dataset, hierarchy):
    return os.path.join(fwd(), f'hierarchies/{dataset}/graph-{hierarchy}.json')


def dataset_to_default_path_wnids(dataset):
    return os.path.join(fwd(), f'wnids/{dataset}.txt')


def generate_kwargs(args, object, name='Dataset', keys=(), globals={}, kwargs=None):
    kwargs = kwargs or {}

    for key in keys:
        accepts_key = getattr(object, f'accepts_{key}', False)
        if not accepts_key:
            continue
        assert key in args or callable(accepts_key)

        value = getattr(args, key, None)
        if callable(accepts_key):
            kwargs[key] = accepts_key(**globals)
            Colors.cyan(f'{key}:\t(callable)')
        elif accepts_key and value:
            kwargs[key] = value
            Colors.cyan(f'{key}:\t{value}')
        elif value:
            Colors.red(
                f'Warning: {name} does not support custom '
                f'{key}: {value}')
    return kwargs


def load_image_from_path(path):
    """Path can be local or a URL"""
    headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'
    }
    if 'http' in path:
      request = Request(path, headers=headers)
      file = io.BytesIO(urlopen(request).read())
    else:
      file = path
    return Image.open(file)


class Colors:
    RED = '\x1b[31m'
    GREEN = '\x1b[32m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    CYAN = '\x1b[36m'

    @classmethod
    def red(cls, *args):
        print(cls.RED + args[0], *args[1:], cls.ENDC)

    @classmethod
    def green(cls, *args):
        print(cls.GREEN + args[0], *args[1:], cls.ENDC)

    @classmethod
    def cyan(cls, *args):
        print(cls.CYAN + args[0], *args[1:], cls.ENDC)

    @classmethod
    def bold(cls, *args):
        print(cls.BOLD + args[0], *args[1:], cls.ENDC)


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
except Exception as e:
    print(e)
    term_width = 50

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def set_np_printoptions():
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def generate_fname(dataset, arch, path_graph, wnid=None, name='',
        trainset=None, include_labels=(), exclude_labels=(),
        include_classes=(), num_samples=0, tree_supervision_weight=0.5,
        fine_tune=False, loss='CrossEntropyLoss',
        **kwargs):
    fname = 'ckpt'
    fname += '-' + dataset
    fname += '-' + arch
    if name:
        fname += '-' + name
    if path_graph:
        path = Path(path_graph)
        fname += '-' + path.stem.replace('graph-', '', 1)
    if include_labels:
        labels = ",".join(map(str, include_labels))
        fname += f'-incl{labels}'
    if exclude_labels:
        labels = ",".join(map(str, exclude_labels))
        fname += f'-excl{labels}'
    if include_classes:
        labels = ",".join(map(str, include_classes))
        fname += f'-incc{labels}'
    if num_samples != 0 and num_samples is not None:
        fname += f'-samples{num_samples}'
    if loss != 'CrossEntropyLoss':
        fname += f'-{loss}'
        if tree_supervision_weight is not None and tree_supervision_weight != 1:
            fname += f'-tsw{tree_supervision_weight}'
    return fname
