from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from dcsg import datasets
from dcsg.models import resnet50part
from dcsg.evaluators import Evaluator
from dcsg.utils.data import transforms as T
from dcsg.utils.data.preprocessor import Preprocessor
from dcsg.utils.logging import Logger
from dcsg.utils.serialization import load_checkpoint, copy_state_dict


def get_data(name, data_dir, height, width, batch_size, workers):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root)
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])
    test_loader = DataLoader(Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                                          root=dataset.images_dir, transform=test_transformer),
                             batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)

    return dataset, test_loader


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    cudnn.benchmark = True

    log_dir = osp.dirname(args.resume)
    sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # dataset
    dataset, test_loader = get_data(args.dataset, args.data_dir, args.height, args.width, args.batch_size, args.workers)

    # model
    model = resnet50part(num_parts=args.part, num_classes=3000)
    model.cuda()
    model = nn.DataParallel(model)

    # load a checkpoint
    checkpoint = load_checkpoint(args.resume)
    copy_state_dict(checkpoint, model)

    # evaluate
    evaluator = Evaluator(model)
    print("Test on {}:".format(args.dataset))
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True, rerank=args.rerank)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing the model")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=384, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, '../data'))

    # testing configs
    parser.add_argument('--resume', type=str, required=True, metavar='PATH')
    parser.add_argument('--rerank', action='store_true', help="evaluation only")
    parser.add_argument('--seed', type=int, default=1)

    # model configs
    parser.add_argument('--part', type=int, default=3, help="number of part")

    main()
