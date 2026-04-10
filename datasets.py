

import random

import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
import math

from timm.data import create_transform

from continual_datasets.continual_datasets import *

import utils

class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes
    
    def __call__(self, img):
        return self.lambd(img, self.nb_classes)

def target_transform(x, nb_classes):
    return x + nb_classes

def build_continual_dataloader(args):
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    if args.dataset.startswith('Split-'):
        # print('Splitting dataset')
        dataset_train, dataset_val = get_dataset(args.dataset.replace('Split-',''), transform_train, transform_val, args)
        
        # print(dataset_train.classes)
        # print(dataset_train.class_to_idx)
        # print(dataset_train.targets)
        args.nb_classes = len(dataset_val.classes)


        splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
    else:
        if args.dataset == '5-datasets':
            dataset_list = ['SVHN', 'MNIST', 'CIFAR10', 'NotMNIST', 'FashionMNIST']
        else:
            dataset_list = args.dataset.split(',')
        
        if args.shuffle:
            random.shuffle(dataset_list)
        print(dataset_list)
    
        args.nb_classes = 0

    for i in range(args.num_tasks):
        if args.dataset.startswith('Split-'):
            dataset_train, dataset_val = splited_dataset[i]

        else:
            dataset_train, dataset_val = get_dataset(dataset_list[i], transform_train, transform_val, args)

            transform_target = Lambda(target_transform, args.nb_classes)

            if class_mask is not None:
                class_mask.append([i + args.nb_classes for i in range(len(dataset_val.classes))])
                args.nb_classes += len(dataset_val.classes)

            if not args.task_inc:
                dataset_train.target_transform = transform_target
                dataset_val.target_transform = transform_target
        
        if args.distributed and utils.get_world_size() > 1:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()

            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        dataloader.append({'train': data_loader_train, 'val': data_loader_val})
    # print(dataset_train.targets)
    # print(class_mask)
    # exit(0)
    return dataloader, class_mask

def get_dataset(dataset, transform_train, transform_val, args,):
    if dataset == 'CIFAR100':
        dataset_train = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_val)
        # print(dataset_train.targets)
        # exit(0)

    elif dataset == 'CIFAR10':
        dataset_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'MNIST':
        dataset_train = MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'FashionMNIST':
        dataset_train = FashionMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = FashionMNIST(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'SVHN':
        dataset_train = SVHN(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = SVHN(args.data_path, split='test', download=True, transform=transform_val)
    
    elif dataset == 'NotMNIST':
        dataset_train = NotMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = NotMNIST(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'Flower102':
        dataset_train = Flowers102(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = Flowers102(args.data_path, split='test', download=True, transform=transform_val)
    
    elif dataset == 'Cars196':
        dataset_train = StanfordCars(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = StanfordCars(args.data_path, split='test', download=True, transform=transform_val)
        
    elif dataset == 'CUB200':
        dataset_train = CUB200(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = CUB200(args.data_path, train=False, download=True, transform=transform_val).data
    
    elif dataset == 'Scene67':
        dataset_train = Scene67(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Scene67(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'TinyImagenet':
        dataset_train = TinyImagenet(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = TinyImagenet(args.data_path, train=False, download=True, transform=transform_val).data
        
    elif dataset == 'Imagenet-R':
        dataset_train = Imagenet_R(args.data_path, train=True, download=False, transform=transform_train).data
        dataset_val = Imagenet_R(args.data_path, train=False, download=False, transform=transform_val).data
        # print(dataset_train)
    # elif dataset == 'Domainnet':
    #     dataset_train = domainnet(args.data_path, split='train', download=True, transform=transform_train).data
    #     dataset_val = domainnet(args.data_path, split='val', download=True, transform=transform_val).data
    
    elif dataset == 'StanfordCars':
        dataset_train = StanfordCars(args.data_path, split='train', download=False, transform=transform_train)
        dataset_val = StanfordCars(args.data_path, split='test', download=False, transform=transform_val)
        # print(dataset_train.targets)

    elif dataset == 'Domainnet':
        dataset_train = Domainnet(args.data_path, train=True, download=False, transform=transform_train).data
        dataset_val = Domainnet(args.data_path, train=False, download=False, transform=transform_val).data
        # print(dataset_train.targets)
        # for idx, (path, target) in enumerate(dataset_val.samples):
        #     # print(path.split('/'))
        #     # self.data.samples[idx] = (path, label_map[path.split('/')[-2]])
        #     print(path,target)
        # exit(0)
        # print(dataset_train.samples)
        # print(dataset_val.samples)
        # exit(0)
    else:
        raise ValueError('Dataset {} not found.'.format(dataset))
    
    return dataset_train, dataset_val


def split_single_dataset(dataset_train, dataset_val, args):
    nb_classes = len(dataset_val.classes)
    # print('nb_classes',nb_classes)
    # assert nb_classes % args.num_tasks == 0
    classes_per_task=math.ceil(nb_classes/args.num_tasks)
    # classes_per_task = nb_classes // args.num_tasks
    print('class_per_task',classes_per_task)
    labels = [i for i in range(nb_classes)]
    
    split_datasets = list()
    mask = list()

    if args.shuffle:
        shuffle_labels = [i for i in range(nb_classes)]
        random.shuffle(shuffle_labels)
        class_map={}
        for i in range(len(shuffle_labels)):
            class_map[i]=shuffle_labels[i]
        for idx in range(len(dataset_train.targets)):
            
            dataset_train.samples[idx] =(dataset_train.samples[idx][0],class_map[dataset_train.targets[idx]])
            # print((path, class_map[dataset_train.targets[idx]]))
            # exit(0)
            dataset_train.targets[idx] =  class_map[dataset_train.targets[idx]]

        for idx in range(len(dataset_val.targets)):
            
            dataset_val.samples[idx] = (dataset_val.samples[idx][0],class_map[dataset_val.targets[idx]])
            dataset_val.targets[idx] =  class_map[dataset_val.targets[idx]]

    for _ in range(args.num_tasks):
        train_split_indices = []
        test_split_indices = []
        
        scope = labels[:classes_per_task]
        labels = labels[classes_per_task:]

        mask.append(scope)
        # print(scope)
        for k in range(len(dataset_train.targets)):
            # print(dataset_train.targets[k])
            if int(dataset_train.targets[k]) in scope:
                # print(dataset_train.targets[k])
                train_split_indices.append(k)
                
        for h in range(len(dataset_val.targets)):
            if int(dataset_val.targets[h]) in scope:
                test_split_indices.append(h)
        
        
        subset_train, subset_val =  Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices)
        # exit(0)
        

        split_datasets.append([subset_train, subset_val])
    
    return split_datasets, mask

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    
    return transforms.Compose(t)