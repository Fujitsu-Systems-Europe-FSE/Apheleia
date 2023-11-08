from torch.utils import data
from torchvision import transforms
from apheleia import ProjectLogger
from torch.utils.data import DistributedSampler
from apheleia.dataset.abstract_dataset import ImageDataset


def build_sampler(args, dataset):
    sampler = DistributedSampler(dataset) if args.distributed else None
    return sampler


def build_dataset(dataset_class, path, args, additional_transforms=None):
    trans = []
    if issubclass(dataset_class, ImageDataset):
        if not hasattr(args, 'means') or not hasattr(args, 'stds') or args.means is None or args.stds is None:
            args.means = [0.485, 0.456, 0.406]
            args.stds = [0.229, 0.224, 0.225]
            ProjectLogger().warning('Using ImageNet dataset standardization.')

        if hasattr(args, 'im_size'):
            args.im_size = args.im_size if isinstance(args.im_size, tuple) or isinstance(args.im_size, list) else (args.im_size, args.im_size)
            w, h = args.im_size
            trans = [transforms.Resize((h, w))]
        else:
            ProjectLogger().warning('Dataset won\'t be resizable : Missing --im-size on cli parser.')

        trans += [
            transforms.ToTensor(),  # range [0;1]
            transforms.Normalize(args.means, args.stds)
        ]

    # prepend additional transforms
    if additional_transforms is not None:
        trans = additional_transforms + trans
    return dataset_class(args, path, transform=transforms.Compose(trans))


def build_dataloader(dataset_class, path, batch_size, workers, args, dataset_factory_fn=build_dataset,
                     sampler_fn=build_sampler, collate_fn=None, additional_transforms=None):
    """
    Build a dataloader
    :param dataset_class:
    :param path:
    :param batch_size:
    :param workers:
    :param args:
    :param dataset_factory_fn:
    :param sampler_fn:
    :param collate_fn:
    :param additional_transforms:
    :return:
    """
    dataset = dataset_factory_fn(dataset_class, path, args, additional_transforms)
    sampler = sampler_fn(args, dataset)
    opts = {
        'batch_size': batch_size,
        'drop_last': False,
        'shuffle': True if sampler is None else False,
        'num_workers': workers,
        'sampler': sampler
    }
    if collate_fn is not None:
        opts['collate_fn'] = collate_fn

    if workers > 0:
        ProjectLogger().warning('DataLoader workers are not seeded.')

    return data.DataLoader(dataset, **opts)


def build_dataloaders(opts, dataset_clazz, **kwargs):
    """
    Build all three (train/val/test) dataloaders
    :param opts:
    :param kwargs:
    :return:
    """
    train_data = build_dataloader(dataset_clazz, opts.dataset, opts.batch_size, opts.workers, opts, **kwargs)

    val_data = train_data
    if opts.val_dataset is not None:
        val_data = build_dataloader(dataset_clazz, opts.val_dataset, opts.batch_size, opts.workers, opts, **kwargs)
    else:
        ProjectLogger().warning('Train dataset will be reused for validation.')

    test_data = None
    if hasattr(opts, 'test_dataset') and opts.test_dataset is not None:
        test_data = build_dataloader(dataset_clazz, opts.test_dataset, opts.batch_size, opts.workers, opts, **kwargs)
    else:
        opts.test_interval = None
        ProjectLogger().warning('Test eval disabled.')

    return train_data, val_data, test_data
