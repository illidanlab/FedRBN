"""Federated learning with different aggregation strategies.

One-to-Many experiment where we use one node for AT and others not.
Cross-device federated setting.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim

import time
from utils.data_utils import DigitsDataset
import argparse
import numpy as np
import wandb
from utils.utils import AverageMeter

from federated.core import AdversaryCreator, Federation

from federated.learning import train, test, train_prox
from federated.baselines import train_meta
from utils.utils import CosineAnnealingLR, str2bool, set_seed
from utils.config import CHECKPOINT_ROOT, make_if_not_exist


def fedat_test(fed, running_model, val_loaders, val_adversaries, att_BNn, detector,
               loss_fun, device, client_num, set_name='Val'):
    acc_list = [None for _ in range(client_num)]
    loss_mt = AverageMeter()
    for client_idx in range(client_num):
        fed.model_accum.load_model(running_model, client_idx)
        loss, acc = test(
            running_model, val_loaders[client_idx], loss_fun, device,
            adversary=val_adversaries[client_idx],
            att_BNn=att_BNn,
            detector=detector,
        )
        loss_mt.append(loss)
        acc_list[client_idx] = acc

        print(' {:<11s}| {}  Acc: {:.1f}%'.format(fed.clients[client_idx], set_name, acc*100))

    return {f"loss": loss_mt.avg, f"acc": np.mean(acc_list)}, acc_list


def copy_client_bn(fed, src_weight_mode):
    # if args.noise_ratio > 0:  # need to copy BN
    noise_train_idx = [i for i, ds_name in enumerate(fed.clients) if 'noised' in ds_name]
    clean_train_idx = [i for i, ds_name in enumerate(fed.clients) if 'clean' in ds_name]
    assert len(noise_train_idx) > 0, "Not found noised users."
    if len(clean_train_idx) == 0:
        print("Not found clean users. Not copy any BN.")
        return
    print(f"Copy BN: {noise_train_idx} -> {clean_train_idx}")
    for dst_model_idx in clean_train_idx:
        print(f" * copy {noise_train_idx} -> {fed.clients[dst_model_idx]}")
        fed.model_accum.copy_multi_dual_noise_bn(
            noise_train_idx,
            dst_model_idx,
            src_weight_mode=src_weight_mode
        )
    # else:
    #     clean_train_idx = [i for i, ds_name in enumerate(fed.clients) if 'clean' in ds_name]
    #     print(f"Locally copy BNc to BNa for clean users...")
    #     for idx in clean_train_idx:
    #         fed.model_accum.duplicate_dual_clean_bn(idx)  # use clean bn for noise case.


def get_model_fh(data, model):
    if data == 'Digits':
        if model in ['digit']:
            from nets.models import DigitModel
            ModelClass = DigitModel
        else:
            raise ValueError(f"Invalid model: {model}")
    elif data in ['DomainNet']:
        if model in ['alexnet']:
            from nets.models import AlexNet
            ModelClass = AlexNet
        else:
            raise ValueError(f"Invalid model: {model}")
    else:
        raise ValueError(f"Unknown dataset: {data}")
    return ModelClass


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--data', type=str, default='Digits',
                        choices=['Digits', 'DomainNet'])
    parser.add_argument('--model', type=str, default='digit', choices=['digit', 'alexnet'])
    parser.add_argument('--mode', choices=[
        'fedrbn', 'fedravg', 'fedbn', 'fedavg', 'fedprox', 'cent', 'fedmeta',
        # baselines
    ], type=str.lower, default='fedrbn')
    # federated
    Federation.add_argument(parser)
    # opt
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--lr_sch', type=str, default='none', help='learning rate schedule')
    parser.add_argument('--rounds', type=int, default=300, help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization rounds in local worker per communication')

    # FedRBN parameters
    parser.add_argument('--src_weight_mode', choices=['eq', 'cos'],
                        type=str.lower, default='cos', help='only useful for test or non-zero pnc.')
    parser.add_argument('--pnc', type=float, default=-1.,
                        help='Coefficient of Pseudo Noise Calibration (PNC) loss. Set to negative '
                             'to disable PNC loss.')
    parser.add_argument('--pnc_warmup', type=int, default=10,
                        help='# of steps to use pnc=0 at the beginning.')

    # fine-tuning params
    parser.add_argument('--AT_iters', type=int, default=-1,
                        help='Run AT users at the first `AT_iters` iterations and run ST in the '
                             'rest. If AT_iters=-1, then do ordinary run.')
    parser.add_argument('--freeze_n_layer', type=int, default=0,
                        help='Freeze the first `freeze_n_layer` layers (most close to the input).'
                             ' Only used when `AT_iters` > 0.')
    # about dual BN
    parser.add_argument('--te_att_BNn', action='store_true', help='attack noised BN at test')
    parser.add_argument('--oracle_detector', type=str.lower,
                        choices=['none', 'clean', 'gt', 'noised', 'rgt'], default='none',
                        help='use oracle instead of fited detector where `gt` for ground-truth')

    # test
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    parser.add_argument('--test_noise', type=str, default='LinfPGD', help='noise the test data')

    # control
    parser.add_argument('--resume', action='store_true',
                        help='resume training from the save path checkpoint')
    parser.add_argument('--no_log', action='store_true', help='disable wandb log')
    args = parser.parse_args()

    set_seed(args.seed)

    run_name = f'{args.mode}'
    run_name += Federation.render_run_name(args)
    if args.model != 'digit': run_name += f'__{args.model}'
    if args.seed != 1: run_name += f'__s{args.seed}'
    if args.wk_iters != 1: run_name += f'__wk_iters_{args.wk_iters}'
    if args.batch != 32: run_name += f'__b{args.batch}'
    if args.lr_sch != 'none': run_name += f'__lrs_{args.lr_sch}'
    if args.pnc >= 0.: run_name += f'__pnc{args.pnc}'
    if args.AT_iters != -1:
        run_name += f'__AT_iters_{args.AT_iters}'
        if args.freeze_n_layer != 0:
            run_name += f'__frz{args.freeze_n_layer}'
    else:
        if args.freeze_n_layer != 0:
            raise ValueError(f"Cannot freeze layers when AT_iters is {args.AT_iters}")

    args.save_path = os.path.join(CHECKPOINT_ROOT, args.mode, args.data)
    make_if_not_exist(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, run_name)

    wandb.init(group=run_name, project='FedRBN_release',
               config={**vars(args)}, mode='offline' if args.no_log else 'online')
    fed = Federation(args.data, args)

    all_domains = fed.all_domains
    train_loaders, val_loaders, test_loaders = fed.get_data()
    mean_batch_iters = int(np.mean([len(tl) for tl in train_loaders]))
    print(f"  mean_batch_iters: {mean_batch_iters}")

    # setup model
    ModelClass = get_model_fh(args.data, args.model)
    running_model = ModelClass(
        bn_type='dbn' if args.mode.lower() in ['fedrbn', 'fedravg'] else 'bn').to(device)
    loss_fun = nn.CrossEntropyLoss()

    client_num = len(fed.clients)
    fed.make_aggregator(running_model)
    best_changed = False

    # //////// Test ///////
    if args.test:
        print(f'Loading chkpt from {SAVE_PATH}')
        checkpoint = torch.load(SAVE_PATH)
        best_round, best_acc = checkpoint['best_round'], checkpoint['best_acc']
        wandb.summary[f'best_round'] = best_round
        start_rnd = int(checkpoint['round']) + 1

        print('Resume training from epoch {} with best acc:'.format(start_rnd))
        for client_idx, acc in enumerate(best_acc):
            print(' Best site-{:<10s}| Epoch:{} | Val Acc: {:.4f}'.format(
                fed.clients[client_idx], best_round, acc))
        # server_model.load_state_dict(checkpoint['server_model'])
        fed.model_accum.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() in ['fedrbn']:
            copy_client_bn(fed, args.src_weight_mode)

        make_adv = AdversaryCreator(args.test_noise)
        adversaries = [make_adv(running_model) for _ in range(client_num)]

        test_summary, _ = fedat_test(fed, running_model, test_loaders, adversaries, args.te_att_BNn,
                                     args.oracle_detector if args.oracle_detector != 'none' else 'clean',
                                     loss_fun, device, client_num,
                                     set_name='Test')

        print(f"\n Average Test Acc: {test_summary['acc']}")
        wandb.summary.update({'test_' + k: v for k, v in test_summary.items()})
        wandb.finish()

        exit(0)

    # ///// Resume models //////
    if args.resume and os.path.exists(SAVE_PATH):
        print(f'Loading chkpt from {SAVE_PATH}')
        checkpoint = torch.load(SAVE_PATH)
        fed.model_accum.load_state_dict(checkpoint['server_model'])
        best_round, best_acc = checkpoint['best_round'], checkpoint['best_acc']
        start_rnd = int(checkpoint['round']) + 1

        print('Resume training from round {} with best acc:'.format(start_rnd))
        for client_idx, acc in enumerate(best_acc):
            print(' Best client-{:<10s}| Epoch:{} | Val Acc: {:.4f}'.format(
                fed.clients[client_idx], best_round, acc))
    else:
        # log the best for each model on all datasets
        best_round = 0
        best_acc = [0. for _ in range(client_num)]
        start_rnd = 0

    make_adv = AdversaryCreator(args.noise)
    adversaries = [make_adv(running_model) if 'noised' in ds_name else None
                   for ds_name in fed.clients]
    test_make_adv = AdversaryCreator(args.test_noise)
    val_adversaries = [test_make_adv(running_model) if 'noised' in ds_name else None
                       for ds_name in fed.clients]

    if args.lr_sch == 'cos':
        lr_sch = CosineAnnealingLR(args.rounds, eta_max=args.lr, eta_min=0.1 * args.lr,
                                   last_epoch=start_rnd)
    else:
        assert args.lr_sch == 'none'
        lr_sch = None

    # //////// Federated Training ///////
    for round in range(start_rnd, args.rounds):
        if args.noise == 'Free':
            if round >= args.rounds / make_adv.steps:
                print(f"Stop as epoch reach the limit of FreeAttack {args.rounds / make_adv.steps}")
                break

        global_lr = args.lr if lr_sch is None else lr_sch.step()
        optimizers = [optim.SGD(params=running_model.parameters(), lr=global_lr) for idx in
                      range(client_num)]

        # ----------- Train ---------------
        train_loss_mt, train_acc_mt = AverageMeter(), AverageMeter()
        # for wi in range(1):  # range(args.wk_iters):
        print("============ Round {} ============".format(round))

        if args.pnc >= 0:
            copy_client_bn(fed, args.src_weight_mode)

        for client_idx in fed.client_sampler.iter():  # range(client_num):
            if args.AT_iters > 0:
                if (round < args.AT_iters and adversaries[client_idx] is None) \
                        or (round >= args.AT_iters and adversaries[client_idx] is not None):  # skip the AT or ST users
                    continue
                else:
                    if round >= args.AT_iters and args.freeze_n_layer != 0:
                        if adversaries[client_idx] is None:  # ST user
                            # freeze shallow layers
                            running_model.freeze_shallow_layers(n_layer=args.freeze_n_layer,
                                                                freeze=True)
                        else:
                            # defrozen
                            running_model.freeze_shallow_layers(freeze=False)

            # load model
            fed.download(running_model, client_idx)

            start_time = time.process_time()

            # Local train
            if args.mode.lower() == 'fedprox':
                # skip the first server model(random initialized)
                if round > 0:
                    train_loss, train_acc = train_prox(
                        running_model, fed.model_accum.server_state_dict,
                        train_loaders[client_idx],
                        optimizers[client_idx], loss_fun, device,
                        adversary=adversaries[client_idx],
                        max_iter=mean_batch_iters * args.wk_iters if args.partition_mode != 'uni' else len(
                            train_loaders[client_idx]) * args.wk_iters,
                        adv_lmbd=args.adv_lmbd)
                else:
                    train_loss, train_acc = train(running_model, train_loaders[client_idx],
                                                  optimizers[client_idx], loss_fun, device,
                                                  max_iter=mean_batch_iters * args.wk_iters if args.partition_mode != 'uni' else len(
                                                      train_loaders[client_idx]) * args.wk_iters,
                                                  adv_lmbd=args.adv_lmbd)

            elif args.mode.lower() == 'fedmeta':

                if args.partition_mode != 'uni':
                    raise RuntimeError(f"Only support uniform partition since we do not "
                                       f"limit the max iter for unequal user sample size.")
                train_loss, train_acc = train_meta(running_model, train_loaders[client_idx],
                                                   optimizers[client_idx], loss_fun, device,
                                                   adversary=adversaries[client_idx])

            else:
                train_loss, train_acc = train(running_model, train_loaders[client_idx],
                                              optimizers[client_idx], loss_fun, device,
                                              adversary=adversaries[client_idx],
                                              max_iter=mean_batch_iters * args.wk_iters if args.partition_mode != 'uni' else len(
                                                  train_loaders[client_idx]) * args.wk_iters,
                                              att_BNn=args.mode.lower() in ['fedrbn',
                                                                            'fedravg'],
                                              adv_lmbd=args.adv_lmbd,
                                              pnc_coef=args.pnc if round > args.pnc_warmup else 0.,
                                              )

            fed.upload(running_model, client_idx)

            elapsed = time.process_time() - start_time
            train_loss_mt.append(train_loss), train_acc_mt.append(train_acc)
            print(
                ' Client-{:<10s}| Train Loss: {:.3f} | Train Acc: {:.1f}% | Elapsed: {:.2f} s'.format(
                    fed.clients[client_idx], train_loss, train_acc*100, elapsed))
            wandb.log({
                f"{fed.clients[client_idx]} train_loss": train_loss,
                f"{fed.clients[client_idx]} train_acc": train_acc,
            }, commit=False)

        # Aggregation
        fed.aggregate()

        # Validation
        val_summary, val_acc_list = fedat_test(
            fed, running_model, val_loaders, val_adversaries,
            args.mode.lower() in ['fedrbn', 'fedravg'],
            'noised' if args.mode.lower() in ['fedrbn', 'fedravg'] and args.pnc >= 0 else 'gt',
            loss_fun, device, client_num, set_name='Val'
        )
        wandb.log({
            f"train_loss": train_loss_mt.avg,
            f"train_acc": train_acc_mt.avg,
            **{'val_'+k: v for k, v in val_summary.items()}
        }, commit=False)
        print(f" Avg Val Acc: {val_summary['acc']:.3f}")

        # Record best
        if val_summary['acc'] > np.mean(best_acc):
            print(f"Update best Acc.")
            best_round = round
            best_changed = True
            for client_idx in range(client_num):
                best_acc[client_idx] = val_acc_list[client_idx]
                print(' Client-{:<10s}| Epoch:{} | Val Acc: {:.4f}'.format(
                    fed.clients[client_idx], best_round, best_acc[client_idx]))

        if best_changed:
            print(' Saving the local and server checkpoint to {}...'.format(SAVE_PATH))
            save_dict = {
                'server_model': fed.model_accum.state_dict(),
                'best_round': best_round,
                'best_acc': best_acc,
                'round': round,
                'all_domains': all_domains,
            }
            torch.save(save_dict, SAVE_PATH)
            best_changed = False
        wandb.log({
            "round": round,
            "best_val_acc": np.mean(best_acc),
        }, commit=True)


if __name__ == '__main__':
    main()
