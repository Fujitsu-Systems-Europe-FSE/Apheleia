from torch.utils import data
from torch.utils.data import DistributedSampler
from torchvision import transforms
from apheleia import ProjectLogger


def get_dataset(dataset_class, mode, path, args):
    dataset_cls = DATASETS[dataset_class]['dataset'][mode]
    dataset_type = DATASETS[dataset_class]['type']

    args.means = [0.485, 0.456, 0.406] if 'means' not in args or args.means is None else args.means
    args.stds = [0.229, 0.224, 0.225] if 'stds' not in args or args.stds is None else args.stds

    trans = []
    if args.im_size is not None:
        args.im_size = args.im_size if isinstance(args.im_size, tuple) else (args.im_size, args.im_size)
        w, h = args.im_size
        trans = [transforms.Resize((h, w))]

    trans += [
        transforms.ToTensor(),  # range [0;1]
        transforms.Normalize(args.means, args.stds)
    ]

    dat = dataset_cls(path, **{'transform': transforms.Compose(trans)})
    return dat, dataset_type


def get_dataloader(dataset_class, mode, path, batch_size, workers, args, dataset_fn=get_dataset, collate_fn=None):
    dataset, dataset_type = dataset_fn(dataset_class, mode, path, args)
    opts = {
        'batch_size': batch_size,
        'drop_last': False,
        'shuffle': True if not args.distributed else False,
        'num_workers': workers,
        'sampler': DistributedSampler(dataset) if args.distributed else None
    }
    if collate_fn is not None:
        opts['collate_fn'] = collate_fn

    if workers > 0:
        ProjectLogger().warning('DataLoader workers are not seeded.')

    return data.DataLoader(dataset, **opts), dataset_type
