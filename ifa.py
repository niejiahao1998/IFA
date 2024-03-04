from model.IFA_matching import IFA_MatchingNet
from util.utils import count_params, set_seed, mIOU

import argparse
from copy import deepcopy
import os
import time
import torch
from torch.nn import CrossEntropyLoss, DataParallel
import torch.nn.functional as F
from torch.optim import SGD
from tqdm import tqdm
from data.dataset import FSSDataset


def parse_args():
    parser = argparse.ArgumentParser(description='IFA for CD-FSS')
    # basic arguments
    parser.add_argument('--data-root',
                        type=str,
                        required=True,
                        help='root path of training dataset')
    parser.add_argument('--dataset',
                        type=str,
                        default='fss',
                        choices=['fss', 'deepglobe', 'isic', 'lung'],
                        help='training dataset')
    parser.add_argument('--batch-size',
                        type=int,
                        default=4,
                        help='batch size of training')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='learning rate')
    parser.add_argument('--crop-size',
                        type=int,
                        default=473,
                        help='cropping size of training samples')
    parser.add_argument('--backbone',
                        type=str,
                        choices=['resnet50', 'resnet101'],
                        default='resnet50',
                        help='backbone of semantic segmentation model')
    parser.add_argument('--refine', dest='refine', action='store_true', default=False)
    parser.add_argument('--shot',
                        type=int,
                        default=1,
                        help='number of support pairs')
    parser.add_argument('--episode',
                        type=int,
                        default=24000,
                        help='total episodes of training')
    parser.add_argument('--snapshot',
                        type=int,
                        default=1200,
                        help='save the model after each snapshot episodes')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed to generate tesing samples')

    args = parser.parse_args()
    return args

def evaluate(model, dataloader, args):
    tbar = tqdm(dataloader)

    if args.dataset == 'fss':
        num_classes = 1000
    elif args.dataset == 'deepglobe':
        num_classes = 6
    elif args.dataset == 'isic':
        num_classes = 3
    elif args.dataset == 'lung':
        num_classes = 1

    metric = mIOU(num_classes)

    for i, (img_s_list, mask_s_list, img_q, mask_q, cls, _, id_q) in enumerate(tbar):

        img_s_list = img_s_list.permute(1,0,2,3,4)
        mask_s_list = mask_s_list.permute(1,0,2,3)
            
        img_s_list = img_s_list.numpy().tolist()
        mask_s_list = mask_s_list.numpy().tolist()

        img_q, mask_q = img_q.cuda(), mask_q.cuda()

        for k in range(len(img_s_list)):
            img_s_list[k], mask_s_list[k] = torch.Tensor(img_s_list[k]), torch.Tensor(mask_s_list[k])
            img_s_list[k], mask_s_list[k] = img_s_list[k].cuda(), mask_s_list[k].cuda()
        cls = cls[0].item()
        cls = cls + 1

        with torch.no_grad():
            out_ls = model(img_s_list, mask_s_list, img_q, mask_q)
            pred = torch.argmax(out_ls[0], dim=1)

        pred[pred == 1] = cls
        mask_q[mask_q == 1] = cls

        metric.add_batch(pred.cpu().numpy(), mask_q.cpu().numpy())

        tbar.set_description("Testing mIOU: %.2f" % (metric.evaluate() * 100.0))

    return metric.evaluate() * 100.0

def main():
    path_dir = 'ifa'

    args = parse_args()
    print('\n' + str(args))

    ### Please modify the following paths with your trained model paths.
    if args.dataset == 'deepglobe':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './outdir/models/deepglobe/resnet50_1shot_avg_44.40.pth'
            if args.shot == 5:
                checkpoint_path = './outdir/models/deepglobe/resnet50_5shot_avg_52.78.pth'
    if args.dataset == 'isic':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './outdir/models/isic/resnet50_1shot_avg_55.50.pth'
            if args.shot == 5:
                checkpoint_path = './outdir/models/isic/resnet50_5shot_avg_62.60.pth'
    if args.dataset == 'lung':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './outdir/models/lung/resnet50_1shot_avg_72.64.pth'
            if args.shot == 5:
                checkpoint_path = './outdir/models/lung/resnet50_5shot_avg_73.07.pth'
    if args.dataset == 'fss':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './outdir/models/fss/resnet50_1shot_avg_77.16.pth'
            if args.shot == 5:
                checkpoint_path = './outdir/models/fss/resnet50_5shot_avg_79.37.pth'
    
    miou = 0
    save_path = 'outdir/models/%s/%s' % (args.dataset, path_dir)
    os.makedirs(save_path, exist_ok=True)
 
    FSSDataset.initialize(img_size=400, datapath=args.data_root)
    train_dataset = args.dataset+'ifa'
    trainloader = FSSDataset.build_dataloader(train_dataset, args.batch_size, 4, '0', 'val', args.shot)
    FSSDataset.initialize(img_size=400, datapath=args.data_root)
    testloader = FSSDataset.build_dataloader(args.dataset, args.batch_size, 4, '0', 'val', args.shot)

    print('Do we use SSP refinement?', args.refine)
    model = IFA_MatchingNet(args.backbone, args.refine, args.shot)
    print('\nParams: %.1fM' % count_params(model))
    
    print('Loaded model:', checkpoint_path)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    for param in model.layer0.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = False

    criterion = CrossEntropyLoss(ignore_index=255)
    optimizer = SGD([param for param in model.parameters() if param.requires_grad],
                    lr=args.lr, momentum=0.9, weight_decay=5e-4)

    model = DataParallel(model).cuda()
    best_model = None

    iters = 0
    total_iters = args.episode // args.batch_size
    lr_decay_iters = [total_iters // 3, total_iters * 2 // 3]

    previous_best = float(miou)
    # each snapshot is considered as an epoch
    for epoch in range(args.episode // args.snapshot):
        
        print("\n==> Epoch %i, learning rate = %.5f\t\t\t\t Previous best = %.2f"
              % (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()

        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

        total_loss = 0.0

        tbar = tqdm(trainloader)
        set_seed(int(time.time()))

        for i, (img_s_list, mask_s_list, img_q, mask_q, _, _, _) in enumerate(tbar):

            img_s_list = img_s_list.permute(1,0,2,3,4)
            mask_s_list = mask_s_list.permute(1,0,2,3)      
            img_s_list = img_s_list.numpy().tolist()
            mask_s_list = mask_s_list.numpy().tolist()

            img_q, mask_q = img_q.cuda(), mask_q.cuda()

            for k in range(len(img_s_list)):
                img_s_list[k], mask_s_list[k] = torch.Tensor(img_s_list[k]), torch.Tensor(mask_s_list[k])
                img_s_list[k], mask_s_list[k] = img_s_list[k].cuda(), mask_s_list[k].cuda()

            out_ls = model(img_s_list, mask_s_list, img_q, mask_q)
            
            mask_s = torch.cat(mask_s_list, dim=0)
            mask_s = mask_s.long()

            if args.refine:
                ### iter = 3
                loss = criterion(out_ls[0], mask_q) + criterion(out_ls[1], mask_q) + criterion(out_ls[2], mask_q) + criterion(out_ls[3], mask_s) * 0.2 + \
                    criterion(out_ls[4], mask_s) * 0.4 + criterion(out_ls[5], mask_q) * 0.1 + criterion(out_ls[6], mask_s) * 0.1 + \
                        criterion(out_ls[7], mask_q) * 0.1 + criterion(out_ls[8], mask_s) * 0.1
            else:
                loss = criterion(out_ls[0], mask_q) + criterion(out_ls[1], mask_q) + criterion(out_ls[2], mask_s) * 0.4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters += 1
            if iters in lr_decay_iters:
                optimizer.param_groups[0]['lr'] /= 10.0

            tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))

        model.eval()
        set_seed(args.seed)
        miou = evaluate(model, testloader, args)

        if miou >= previous_best:
            best_model = deepcopy(model)
            previous_best = miou
            torch.save(best_model.module.state_dict(),
                os.path.join(save_path, '%s_%ishot_%.2f.pth' % (args.backbone, args.shot, miou)))
            
    print('\nEvaluating on 5 seeds.....')
    total_miou = 0.0
    for seed in range(5):
        print('\nRun %i:' % (seed + 1))
        set_seed(args.seed + seed)

        miou = evaluate(best_model, testloader, args)
        total_miou += miou

    print('\n' + '*' * 32)
    print('Averaged mIOU on 5 seeds: %.2f' % (total_miou / 5))
    print('*' * 32 + '\n')

    torch.save(best_model.module.state_dict(),
               os.path.join(save_path, '%s_%ishot_avg_%.2f.pth' % (args.backbone, args.shot, total_miou / 5)))



if __name__ == '__main__':
    main()
