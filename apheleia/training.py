from apheleia import seed, ProjectLogger
from apheleia.dataset import get_dataloader

import os


def init_train(args, ctx, train_data, val_data, test_data):
    trainer = TrainingPipelineFactory(args, ctx).build()

    try:
        trainer.start(train_data, val_data, test_data)
    except KeyboardInterrupt as interrupt:
        ProjectLogger().warning('Keyboard interrupt received. Checkpointing current epoch.')
        trainer.do_interrupt_backup()
        raise interrupt


def train(args, ctx):
    seed(args)
    train_data, dataset_type = get_dataloader(args.dataset_class, 'train', args.dataset, args.batch_size, args.workers, args)

    val_data = None
    if args.val_dataset is not None:
        val_data, _ = get_dataloader(args.dataset_class, 'val', args.val_dataset, args.batch_size, args.workers, args)
    else:
        args.val_interval = None
        ProjectLogger().warning('Validation eval disabled.')

    test_data = None
    if args.test_dataset is not None:
        test_data, _ = get_dataloader(args.dataset_class, 'test', args.test_dataset, args.batch_size, args.workers, args)
    else:
        args.test_interval = None
        ProjectLogger().warning('Test eval disabled.')

    args.num_features = train_data.dataset.num_features()
    args.num_classes = train_data.dataset.num_classes()
    args.dataset_type = dataset_type

    if args.runs > 1:
        root_path = args.outdir
        basename = os.path.basename(root_path)
        for i in range(1, args.runs + 1):
            args.outdir = os.path.join(root_path, f'{basename}-{i}')
            init_train(args, ctx, train_data, val_data, test_data)
    else:
        init_train(args, ctx, train_data, val_data, test_data)
