from torch.utils import data
from apheleia import ProjectLogger
from torchvision import transforms
from torch.utils.data import DistributedSampler
from .abstract_dataset import ImageDataset


def build_sampler(args, dataset):
    sampler = DistributedSampler(dataset) if args.distributed else None
    return sampler


def build_dataset(dataset_class, path, args):
    trans = []
    if type(dataset_class) == ImageDataset:
        args.means = [0.485, 0.456, 0.406] if 'means' not in args or args.means is None else args.means
        args.stds = [0.229, 0.224, 0.225] if 'stds' not in args or args.stds is None else args.stds

        if args.im_size is not None:
            args.im_size = args.im_size if isinstance(args.im_size, tuple) else (args.im_size, args.im_size)
            w, h = args.im_size
            trans = [transforms.Resize((h, w))]

        trans += [
            transforms.ToTensor(),  # range [0;1]
            transforms.Normalize(args.means, args.stds)
        ]

    # dat = dataset_class(path, **{'transform': transforms.Compose(trans)})

    return dataset_class(args, path)


def build_dataloader(dataset_class, path, batch_size, workers, args, dataset_factory_fn=build_dataset, sampler_fn=build_sampler, collate_fn=None):
    dataset = dataset_factory_fn(dataset_class, path, args)
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
