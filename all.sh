### CD-FSS - Train
### deepglobe 1-shot
CUDA_VISIBLE_DEVICES=0 python -W ignore train.py \
  --dataset deepglobe --data-root ./dataset \
  --backbone resnet50 --batch-size 12 --shot 1 --refine
### deepglobe 5-shot
CUDA_VISIBLE_DEVICES=0,1 python -W ignore train.py \
  --dataset deepglobe --data-root ./dataset \
  --backbone resnet50 --batch-size 6 --shot 5 --refine
### isic 1-shot
CUDA_VISIBLE_DEVICES=0 python -W ignore train.py \
  --dataset isic --data-root ./dataset \
  --backbone resnet50 --batch-size 12 --shot 1 --refine
### isic 5-shot
CUDA_VISIBLE_DEVICES=0,1 python -W ignore train.py \
  --dataset isic --data-root ./dataset \
  --backbone resnet50 --batch-size 6 --shot 5 --refine
### lung 1-shot
CUDA_VISIBLE_DEVICES=0 python -W ignore train.py \
  --dataset lung --data-root ./dataset \
  --backbone resnet50 --batch-size 12 --shot 1 --refine
### lung 5-shot
CUDA_VISIBLE_DEVICES=0,1 python -W ignore train.py \
  --dataset lung --data-root ./dataset \
  --backbone resnet50 --batch-size 6 --shot 5 --refine
### fss 1-shot
CUDA_VISIBLE_DEVICES=0 python -W ignore train.py \
  --dataset fss --data-root ./dataset \
  --backbone resnet50 --batch-size 12 --shot 1 --refine
### fss 5-shot
CUDA_VISIBLE_DEVICES=0,1 python -W ignore train.py \
  --dataset fss --data-root ./dataset \
  --backbone resnet50 --batch-size 6 --shot 5 --refine


### CD-FSS - Adaptation
### deepglobe 1-shot
CUDA_VISIBLE_DEVICES=0 python -W ignore ifa.py \
  --dataset deepglobe --data-root ./dataset \
  --backbone resnet50 --batch-size 12 --shot 1 --refine --lr 0.0005
### deepglobe 5-shot
CUDA_VISIBLE_DEVICES=0,1 python -W ignore ifa.py \
  --dataset deepglobe --data-root ./dataset \
  --backbone resnet50 --batch-size 6 --shot 5 --refine --lr 0.0005
### isic 1-shot
CUDA_VISIBLE_DEVICES=0 python -W ignore ifa.py \
  --dataset isic --data-root ./dataset \
  --backbone resnet50 --batch-size 12 --shot 1 --refine --lr 0.0005
### isic 5-shot
CUDA_VISIBLE_DEVICES=0,1 python -W ignore ifa.py \
  --dataset isic --data-root ./dataset \
  --backbone resnet50 --batch-size 6 --shot 5 --refine --lr 0.0005
### lung 1-shot
CUDA_VISIBLE_DEVICES=0 python -W ignore ifa.py \
  --dataset lung --data-root ./dataset \
  --backbone resnet50 --batch-size 12  --shot 1 --refine --lr 0.000005
### lung 5-shot
CUDA_VISIBLE_DEVICES=0,1 python -W ignore ifa.py \
  --dataset lung --data-root ./dataset \
  --backbone resnet50 --batch-size 6  --shot 5 --refine --lr 0.000005
### fss 1-shot
CUDA_VISIBLE_DEVICES=0 python -W ignore finetune_self_consistency.py \
  --dataset fss --data-root ./dataset \
  --backbone resnet50 --batch-size 12 --shot 1 --refine --lr 0.0005
### fss 5-shot
CUDA_VISIBLE_DEVICES=0,1 python -W ignore ifa.py \
  --dataset fss --data-root ./dataset \
  --backbone resnet50 --batch-size 6 --shot 5 --refine --lr 0.0005



### CD-FSS - Test
### deepglobe 1-shot
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py \
  --dataset deepglobe --data-root ./dataset \
  --backbone resnet50 --batch-size 96 --shot 1 --refine
### deepglobe 5-shot
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py \
  --dataset deepglobe --data-root ./dataset \
  --backbone resnet50 --batch-size 48 --shot 5 --refine
### isic 1-shot
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py \
  --dataset isic --data-root ./dataset \
  --backbone resnet50 --batch-size 96 --shot 1 --refine
### isic 5-shot
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py \
  --dataset isic --data-root ./dataset \
  --backbone resnet50 --batch-size 48 --shot 5 --refine
### lung 1-shot
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py \
  --dataset lung --data-root ./dataset \
  --backbone resnet50 --batch-size 96 --shot 1 --refine
### lung 5-shot
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py \
  --dataset lung --data-root ./dataset \
  --backbone resnet50 --batch-size 48 --shot 5 --refine
### fss 1-shot
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py \
  --dataset fss --data-root ./dataset \
  --backbone resnet50 --batch-size 96 --shot 1 --refine
### fss 5-shot
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py \
  --dataset fss --data-root ./dataset \
  --backbone resnet50 --batch-size 48 --shot 5 --refine