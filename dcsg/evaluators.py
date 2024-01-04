from __future__ import print_function, absolute_import
import time
import collections
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import random
import copy

from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils import to_torch


def extract_cnn_feature(model, inputs):
    inputs = to_torch(inputs).cuda()
    outputs = model(inputs)
    outputs = outputs.data.cpu()
    return outputs


def extract_features(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels


def extract_all_features(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features_g = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)
            inputs = to_torch(imgs).cuda()
            if isinstance(model, nn.DataParallel):
                outputs_g = model.module.extract_all_features(inputs)
            else:
                outputs_g = model.extract_all_features(inputs)
            outputs_g = outputs_g.data.cpu()

            for fname, output_g, pid in zip(fnames, outputs_g, pids):
                features_g[fname] = output_g
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

        return features_g, labels


def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # dist_m.addmm_(1, -2, x, y.t())
    dist_m.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist_m, x.numpy(), y.numpy()


def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False):
        features, _ = extract_features(self.model, data_loader)
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)


from pplr.utils.data import transforms as T
from torch.utils.data import DataLoader
from pplr.utils.data.preprocessor import Preprocessor

def get_refresh_loader(dataset, height, width, batch_size, workers, aut_mode=''):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    aut_method = [
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        T.Pad(10),
        normalizer
    ]
    if aut_mode == 'crop':
        aut_method.append(T.RandomCrop((height, width)))
    elif aut_mode == 'flip':
        aut_method.append(T.RandomHorizontalFlip(p=1))
    elif aut_mode == 'eras':
        aut_method.append(T.RandomErasing(probability=1, mean=[0.485, 0.456, 0.406]))
    elif aut_mode == 'gray':
        aut_method.append(T.Grayscale(num_output_channels=3))  # num_output_channels=1
    elif aut_mode == 'add':
        aut_method.append(T.RandomHorizontalFlip(p=1))
        aut_method.append(T.RandomCrop((height, width)))  # num_output_channels=1

    test_transformer = T.Compose(aut_method)

    test_loader = DataLoader(Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                                          root=dataset.images_dir, transform=test_transformer),
                             batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)

    return test_loader


class Evaluator_sum(object):
    def __init__(self, model):
        super(Evaluator_sum, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, dataset, args, cmc_flag=False, rerank=False):
        features, _ = extract_features(self.model, data_loader)

        class_features = features
        # aut_mode = ['crop', 'flip', 'eras']
        aut_mode = ['crop', 'flip', 'add']
        for aut in aut_mode:
            class_loader = get_refresh_loader(dataset, args.height, args.width, args.batch_size, args.workers,
                                              aut_mode=aut)
            features_aut, _ = extract_features(self.model, class_loader)
            for f, feature_item in tqdm(features_aut.items()):
                class_features[f] += feature_item

        for f, feature_item in class_features.items():
            class_features[f] = class_features[f] / (len(aut_mode) + 1)

        distmat, query_features, gallery_features = pairwise_distance(class_features, query, gallery)
        results, = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery,
                                cmc_flag=cmc_flag)

        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)