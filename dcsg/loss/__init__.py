from __future__ import absolute_import

from .triplet import TripletLoss, SoftTripletLoss
from .crossentropy import CrossEntropyLabelSmooth, CrossEntropy, SoftEntropy


__all__ = [
    'TripletLoss',
    'CrossEntropyLabelSmooth',
    'CrossEntropy',
    'SoftTripletLoss',
    'SoftEntropy',
]
