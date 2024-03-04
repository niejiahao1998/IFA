r""" FSS-1000 few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np

import albumentations as A
import random


class DatasetDeepglobeIFA(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, num=6):
        self.split = split
        self.benchmark = 'deepglobe'
        self.shot = shot
        self.num = num

        self.base_path = os.path.join(datapath, 'Deepglobe', '04_train_cat')

        self.categories = ['1','2','3','4','5','6']

        self.class_ids = range(0, 6)
        self.img_metadata_classwise = self.build_img_metadata_classwise()

        self.transform = transform

        self.q_transform = A.Compose([
            A.HorizontalFlip(p=0.8),
            A.VerticalFlip(p=0.8),
            A.RandomRotate90(),
            A.RandomGridShuffle(grid=(2,2),always_apply=False,p=1)
        ])

        self.q_transform2 = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(hue_shift_limit=20,sat_shift_limit=30,val_shift_limit=20,always_apply=False,p=1)
        ])

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img_test, query_mask_test, support_imgs, support_masks = self.load_frame(query_name, support_names)

        if self.shot == 1:
            idx = 0
        else:
            idx = random.randint(0, self.shot-1)
        q_img = support_imgs[idx]
        q_img = np.array(q_img)
        q_mask = support_masks[idx]
        q_mask = np.array(Image.open(q_mask).convert('L'))
        pair_transform = self.q_transform(image=q_img, mask=q_mask)
        query_img = pair_transform['image']
        query_mask = pair_transform['mask']
        q_img_transform = self.q_transform2(image=query_img)
        query_img = q_img_transform['image']
        query_img = Image.fromarray(query_img)
        query_img = self.transform(query_img)
        query_mask = self.process_mask(query_mask)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_mask = query_mask.long()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks_tmp = []
        for smask in support_masks:
            smask = self.read_mask(smask)
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        return support_imgs, support_masks, query_img, query_mask, class_sample, support_names, query_name


    def load_frame(self, query_name, support_names):
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        query_id = query_name.split('/')[-1].split('.')[0]
        q_msk_id = query_name.split('/')[-1][:-11] + '_mask_' + query_name.split('/')[-1][-6:-4]
        ann_path = os.path.join(self.base_path, query_name.split('/')[-4], 'test', 'groundtruth')
        query_name = os.path.join(ann_path, q_msk_id) + '.png'
        support_ids = [name.split('/')[-1].split('.')[0] for name in support_names]
        s_msk_ids = [name.split('/')[-1][:-11] + '_mask_' + name.split('/')[-1][-6:-4] for name in  support_names]
        support_names = [os.path.join(ann_path, sid) + '.png' for name, sid in zip(support_names, s_msk_ids)]

        query_mask = self.read_mask(query_name)

        return query_img, query_mask, support_imgs, support_names

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask
    
    def process_mask(self, img):
        mask = torch.tensor(img)
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        class_id = idx % len(self.class_ids)
        class_sample = self.categories[class_id]

        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_id


    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for cat in self.categories:
            img_metadata_classwise[cat] = []

        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, cat, 'test', 'origin'))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg':
                    img_metadata_classwise[cat] += [img_path]
        return img_metadata_classwise
