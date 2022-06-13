from __future__ import division

import os
import tqdm
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from models import load_model
from utils.logger import Logger
from utils.utils import to_cpu, load_classes, provide_determinism, worker_seed_set
from utils.datasets import ListDataset
from utils.augmentations import AUGMENTATION_TRANSFORMS
# from utils.transforms import DEFAULT_TRANSFORMS
from utils.parse_config import parse_data_config
from utils.loss import compute_loss
from test import _evaluate, _create_validation_data_loader

from terminaltables import AsciiTable

from torchsummary import summary


def _create_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_training=False):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set
    )
    return dataloader


def run():

    # ##############
    # Set parameters
    # ##############

    model = "config/yolov3-tiny.cfg"
    data = "config/custom.data"
    epochs = 201
    verbose = True
    n_cpu = 16
    pretrained_weights = "yolov3-tiny.weights"
    checkpoint_interval = 40
    evaluation_interval = 1
    multiscale_training = True
    iou_thres = 0.5
    conf_thres = 0.1
    nms_thres = 0.5
    logdir = "logs"
    seed = -1

    if seed != -1:
        provide_determinism(seed)

    logger = Logger(logdir)  # Tensorboard logger

    # Create output directories if missing
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ############
    # Create model
    # ############

    model = load_model(model, pretrained_weights)

    # Print model
    if verbose:
        summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))

    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']

    # #################
    # Create Dataloader
    # #################

    # Load training dataloader
    dataloader = _create_data_loader(
        train_path,
        mini_batch_size,
        model.hyperparams['height'],
        n_cpu,
        multiscale_training
    )

    # Load validation dataloader
    validation_dataloader = _create_validation_data_loader(
        valid_path,
        mini_batch_size,
        model.hyperparams['height'],
        n_cpu
    )

    # ################
    # Create optimizer
    # ################

    params = [p for p in model.parameters() if p.requires_grad]

    if (model.hyperparams['optimizer'] in [None, "adam"]):
        optimizer = optim.Adam(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
        )
    elif (model.hyperparams['optimizer'] == "sgd"):
        optimizer = optim.SGD(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum']
        )
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")

    for epoch in range(epochs):

        print("\n---- Training Model ----")

        model.train()  # Set model to training mode

        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            outputs = model(imgs)

            loss, loss_components = compute_loss(outputs, targets, model)

            loss.backward()

            ###############
            # Run optimizer
            ###############

            if batches_done % model.hyperparams['subdivisions'] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = model.hyperparams['learning_rate']
                if batches_done < model.hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value
                # Log the learning rate
                logger.scalar_summary("train/learning_rate", lr, batches_done)
                # Set learning rate
                for g in optimizer.param_groups:
                    g['lr'] = lr

                # Run optimizer
                optimizer.step()
                # Reset gradients
                optimizer.zero_grad()

            # ############
            # Log progress
            # ############
            if verbose:
                print(AsciiTable([
                    ["Type", "Value"],
                    ["IoU loss", float(loss_components[0])],
                    ["Object loss", float(loss_components[1])],
                    ["Class loss", float(loss_components[2])],
                    ["Loss", float(loss_components[3])],
                    ["Batch loss", to_cpu(loss).item()],
                ]).table)

            # Tensorboard logging
            tensorboard_log = [
                ("train/iou_loss", float(loss_components[0])),
                ("train/obj_loss", float(loss_components[1])),
                ("train/class_loss", float(loss_components[2])),
                ("train/loss", to_cpu(loss).item())
            ]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            model.seen += imgs.size(0)

        # #############
        # Save progress
        # #############

        # Save model to checkpoint file
        if epoch % checkpoint_interval == 0:
            checkpoint_path = f"checkpoints/yolov3-tiny-epoch-{epoch}.weights"
            print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            # torch.save(model.state_dict(), checkpoint_path)
            model.save_darknet_weights(checkpoint_path)

        # ########
        # Evaluate
        # ########

        if epoch % evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = _evaluate(
                model,
                validation_dataloader,
                class_names,
                img_size=model.hyperparams['height'],
                iou_thres=iou_thres,
                conf_thres=conf_thres,
                nms_thres=nms_thres,
                verbose=verbose
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                    ("validation/precision", precision.mean()),
                    ("validation/recall", recall.mean()),
                    ("validation/mAP", AP.mean()),
                    ("validation/f1", f1.mean())
                ]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)


if __name__ == "__main__":
    run()
