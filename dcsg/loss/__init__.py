from __future__ import absolute_import

from .triplet import TripletLoss, SoftTripletLoss
from .crossentropy import CrossEntropyLabelSmooth, CrossEntropy, SoftEntropy
from .loss import AALS, PGLR, InterCamProxy

__all__ = [
    'TripletLoss',
    'CrossEntropyLabelSmooth',
    'CrossEntropy',
    'SoftTripletLoss',
    'SoftEntropy',
    'AALS',
    'PGLR',
    'InterCamProxy'
]