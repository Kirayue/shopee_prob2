import torch
import numpy as np
from datetime import datetime
from packages.datasets.dataset import DatasetShopee
from packages.networks.ResNext import ResNext
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ignite.engine import Events, Engine
from ignite.utils import setup_logger
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from packages.utils.augmentation import get_transforms
from packages.utils.radam import RAdam
from packages.utils.metrics import Predict, TOP_1


def get_data_loaders(train_cfg, eval_cfg, test_cfg, cfg_augment):
    train_transforms, test_transforms = get_transforms(cfg_augment)

    train_set = DatasetShopee(train_cfg.CSV, train_cfg.DATA_DIR, train_transforms)

    train_loader = DataLoader(
        train_set,
        train_cfg.BATCH_SIZE,
        pin_memory=True,
        num_workers=train_cfg.NUM_WORKERS,
        shuffle=True
    )

    valid_set = DatasetShopee(eval_cfg.CSV, eval_cfg.DATA_DIR, test_transforms)
    valid_loader = DataLoader(
        valid_set,
        eval_cfg.BATCH_SIZE,
        pin_memory=True,
        num_workers=eval_cfg.NUM_WORKERS,
    )

    test_set = DatasetShopee(test_cfg.CSV, test_cfg.DATA_DIR, test_transforms)
    test_loader = DataLoader(
        test_set,
        test_cfg.BATCH_SIZE,
        pin_memory=True,
        num_workers=test_cfg.NUM_WORKERS,
    )
    return train_loader, valid_loader, test_loader


def create_supervised_trainer(model, optimizer, loss_fn, device):
    model.to(device)

    def _update(engine, batch_samples):
        images, labels, _ = batch_samples
        model.train()
        optimizer.zero_grad()
        images = images.to(device)
        preds = model(images)

        labels = labels.to(device)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics, device):
    model.to(device)

    def _inference(engine, batch_samples):
        model.eval()
        outputs = []
        with torch.no_grad():
            model.to(device)
            images, labels, filenames = batch_samples
            images = images.to(device)
            preds = model(images)
            preds = torch.nn.functional.softmax(preds, dim=1)
            preds = torch.argmax(preds, dim=1)
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            return preds, labels, filenames

    engine = Engine(_inference)
    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    images, _, _ = next(data_loader_iter)
    try:
        writer.add_graph(model, images)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def run(mode, cfg):
    device = 'cuda' if cfg.SYSTEM.USE_CUDA else 'cpu'
    print(cfg.MODEL.NAME)
    model = ResNext()
    train_loader, val_loader, test_loader = get_data_loaders(cfg.TRAIN, cfg.EVALUATE, cfg.TEST, cfg.AUGMENT)
    if cfg.MODEL.CHECKPOINT:
        model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))
        print(f"Load {cfg.MODEL.NAME} weight ({cfg.MODEL.CHECKPOINT}) sucessfully!")
    loss = torch.nn.CrossEntropyLoss()
    pbar = ProgressBar()

    if mode == 'train':
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        dir_name = f'{timestamp}_{cfg.MODEL.NAME}{cfg.TAG}'
        writer = create_summary_writer(model, train_loader, f"runs/{dir_name}")

        optimizer = RAdam(model.parameters(), lr=cfg.OPTIM.INIT_LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.LR_SCHEDULER.STEP_SIZE)
        trainer = create_supervised_trainer(model, optimizer, loss, device)
        trainer.logger = setup_logger("trainer")
        pbar.attach(trainer)

        evaluator = create_supervised_evaluator(
            model, {"TOP_1": TOP_1()}, device
        )
        evaluator.logger = setup_logger("evaluator")
        pbar.attach(evaluator)


        model_saver = ModelCheckpoint(
            f"checkpoints/{dir_name}", f"{cfg.MODEL.NAME}{cfg.TAG}", n_saved=10, create_dir=True
        )

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            trainer.logger.info(trainer.state)
            trainer.logger.info(
                "Epoch[{}_{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.iteration, trainer.state.output)
            )
            writer.add_scalar(
                "training/loss", trainer.state.output, trainer.state.iteration
            )
            writer.add_scalar(
                "training/lr", optimizer.param_groups[0]['lr'], trainer.state.iteration
            )

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            model_saver(engine, {"model": model})
            trainer.logger.info("Model saved!")
            scheduler.step()

        @trainer.on(Events.EPOCH_COMPLETED(every=cfg.EVAL_MODEL_EVERY_EPOCH))
        def log_validation_results(engine):
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            evaluator.logger.info(
                "Training Results - Epoch: {} TOP_1: {:.2f}".format(
                    trainer.state.epoch, metrics['TOP_1']
                )
            )
            writer.add_scalar(
                "training/TOP_1", metrics['TOP_1'], trainer.state.iteration
            )

            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            evaluator.logger.info(
                "Validation Results - Epoch: {} TOP_1: {:.2f}".format(
                    trainer.state.epoch, metrics['TOP_1']
                )
            )
            writer.add_scalar(
                "validation/TOP_1", metrics['TOP_1'], trainer.state.iteration
            )

        trainer.run(train_loader, max_epochs=cfg.EPOCH)


    elif mode == 'infer':
        predictor = create_supervised_evaluator(
            model, {"Predict": Predict(cfg.TEST)}, device
        )
        pbar.attach(predictor)
        predictor.logger = setup_logger("predictor")
        predictor.run(test_loader)
        predictor.logger.info("Inference Done.")

    elif mode == 'eval':
        evaluator = create_supervised_evaluator(
            model, {"TOP_1": TOP_1()}, device
        )
        pbar.attach(evaluator)
        evaluator.logger = setup_logger('evaluator')
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        evaluator.logger.info(
            "Validation Results - TOP_1: {:.2f}".format(metrics['TOP_1'])
        )
