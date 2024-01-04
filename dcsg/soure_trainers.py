from __future__ import print_function, absolute_import
import time

from .evaluation_metrics import accuracy
from .loss import SoftTripletLoss, CrossEntropyLabelSmooth
from .utils.meters import AverageMeter


class Trainer(object):
    def __init__(self, model, num_class=500):
        super(Trainer, self).__init__()
        self.model = model
        self.num_class = num_class

        self.criterion_ce = CrossEntropyLabelSmooth(num_classes=num_class).cuda()
        self.criterion_tri = SoftTripletLoss().cuda()

    def train(self, epoch, train_dataloader, optimizer, print_freq=1, train_iters=200):
        self.model.train()

        batch_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        precisions = AverageMeter()

        time.sleep(1)
        end = time.time()
        for i in range(train_iters):
            data = train_dataloader.next()
            inputs, targets = self._parse_data(data)

            # feedforward
            emb_g, logits_g = self.model(inputs)

            # loss
            loss_ce = self.criterion_ce(logits_g, targets)
            loss_tri = self.criterion_tri(emb_g, targets)

            loss = loss_ce + loss_tri

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # summing-up
            prec, = accuracy(logits_g.data, targets.data)

            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            precisions.update(prec[0])

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'L_CE {:.3f} ({:.3f})\t'
                      'L_TRI {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(train_dataloader),
                              batch_time.val, batch_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, idxs = inputs
        return imgs.cuda(), pids.cuda()