import argparse
from xmlrpc.client import boolean
import numpy as np
import os
import tabulate
import torch
import torch.nn.functional as F

import data
import models
import curves
import utils


parser = argparse.ArgumentParser(description='DNN curve ensemble evaluation')

parser.add_argument('--ensemble_size', type=int, default=11, metavar='N',
                    help='size of ensemble to use')
parser.add_argument('--random_ensemble', type=bool, default=False, metavar='N',
                    help='size of ensemble to use')                   

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                    help='curve type to use (default: None)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')

parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to eval (default: None)')

parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')

args = parser.parse_args()

torch.backends.cudnn.benchmark = True

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test,
    shuffle_train=False
)

architecture = getattr(models, args.model)
curve = getattr(curves, args.curve)
model = curves.CurveNet(
    num_classes,
    curve,
    architecture.curve,
    args.num_bends,
    architecture_kwargs=architecture.kwargs,
)

if torch.cuda.is_available():
    model.cuda()
else:
    device = torch.device('cpu')
    model.to(device)

checkpoint = torch.load(args.ckpt)
model.load_state_dict(checkpoint['model_state'])

criterion = F.cross_entropy
regularizer = curves.l2_regularizer(args.wd)

previous_weights = None

columns = ['Ensemble size', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)']

tr_res = utils.test_ensemble(
    test_loader = loaders['train'],
    model = model,
    criterion = criterion,
    regularizer = regularizer,
    ensemble_size = args.ensemble_size,
    random_ensemble=args.random_ensemble,
    train_set = True)
te_res = utils.test_ensemble(
    test_loader = loaders['test'],
    model = model,
    criterion = criterion,
    regularizer = regularizer,
    ensemble_size = args.ensemble_size) 
tr_loss = tr_res['loss']
tr_nll = tr_res['nll']
tr_acc = tr_res['accuracy']
tr_err = 100.0 - tr_acc
te_loss = te_res['loss']
te_nll = te_res['nll']
te_acc = te_res['accuracy']
te_err = 100.0 - te_acc

values = [args.ensemble_size, tr_loss, tr_nll, tr_err, te_nll, te_err]
table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
table = table.split('\n')
table = '\n'.join([table[1]] + table)
print(table)





