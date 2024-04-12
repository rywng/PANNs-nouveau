import argparse
import datetime
import logging
import os
import time

import torch.optim as optim
import torch.utils.data
from pytorch.pytorch_utils import move_data_to_device, count_parameters, do_mixup
from utils import config
from utils.data_generator import AudioSetDatasetCsv, collate_fn, CsvTrainSampler
from utils.utilities import (
    create_folder,
    get_filename,
    create_logging,
    Mixup,
)
from pytorch import models
import tqdm

from pytorch.losses import get_loss_func

from torch.utils.tensorboard import SummaryWriter


def train(
    workspace: str,
    data_type: str,
    sample_rate: int,
    window_size: int,
    hop_size: int,
    mel_bins: int,
    fmin: int,
    fmax: int,
    model_type: str,
    loss_type: str,
    balanced: bool,
    augmentation: str,
    batch_size: int,
    learning_rate: float,
    resume_iteration: int,
    early_stop: int,
    cuda: bool,
    train_csv_path: str,
    classes_num: int,
    filename: str,
):
    """Train AudioSet tagging model.

    Args:
      dataset_dir: str
      workspace: str
      data_type: 'balanced_train' | 'full_train'
      window_size: int
      hop_size: int
      mel_bins: int
      model_type: str
      loss_type: 'clip_bce'
      balanced: 'none' | 'balanced' | 'alternate'
      augmentation: 'none' | 'mixup'
      batch_size: int
      learning_rate: float
      resume_iteration: int
      early_stop: int
      accumulation_steps: int
      cuda: bool
    """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Convert cuda argument to a boolean value
    cuda = cuda and torch.cuda.is_available()

    device = torch.device("cuda") if cuda else torch.device("cpu")

    # Your existing code goes here

    num_workers = 8
    loss_func = get_loss_func(loss_type)

    # Paths
    checkpoints_dir = os.path.join(
        workspace,
        "checkpoints",
        filename,
        "sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}".format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax
        ),
        "data_type={}".format(data_type),
        model_type,
        "loss_type={}".format(loss_type),
        "balanced={}".format(balanced),
        "augmentation={}".format(augmentation),
        "batch_size={}".format(batch_size),
        current_time,
    )
    create_folder(checkpoints_dir)

    statistics_dir = os.path.join(
        workspace,
        "statistics",
        filename,
        "sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}".format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax
        ),
        "data_type={}".format(data_type),
        model_type,
        "loss_type={}".format(loss_type),
        "balanced={}".format(balanced),
        "augmentation={}".format(augmentation),
        "batch_size={}".format(batch_size),
        current_time,
    )
    create_folder(statistics_dir)

    logs_dir = os.path.join(
        workspace,
        "logs",
        filename,
        "sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}".format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax
        ),
        "data_type={}".format(data_type),
        model_type,
        "loss_type={}".format(loss_type),
        "balanced={}".format(balanced),
        "augmentation={}".format(augmentation),
        "batch_size={}".format(batch_size),
        current_time,
    )

    create_logging(logs_dir, filemode="w")
    writer = SummaryWriter(log_dir=statistics_dir)

    # Model
    # Model = models.Cnn14
    Model = eval(f"models.{model_type}")
    assert model_type in str(Model.__name__), f"Wrong model type: {model_type} and {str(Model.__name__)}"
    model = Model(
        sample_rate=sample_rate,
        window_size=window_size,
        hop_size=hop_size,
        mel_bins=mel_bins,
        fmin=fmin,
        fmax=fmax,
        classes_num=classes_num,
    )

    params_num = count_parameters(model)
    # flops_num = count_flops(model, clip_samples)
    logging.info("Parameters num: {}".format(params_num))
    # logging.info('Flops num: {:.3f} G'.format(flops_num / 1e9))

    # Dataset will be used by DataLoader later. Dataset takes a meta as input
    # and return a waveform and a target.
    dataset = AudioSetDatasetCsv(classes_num, sample_rate=sample_rate)

    train_sampler = CsvTrainSampler(train_csv_path, batch_size)

    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    if "mixup" in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.0)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.0,
        amsgrad=True,
    )

    # Parallel
    print("Number of GPU available: {}".format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if "cuda" in str(device):
        model.to(device)

    pbar = tqdm.tqdm(total=early_stop)
    iteration = 0

    for epoch in range(early_stop):
        checkpoint = {
            "iteration": epoch,
            "model": model.module.state_dict(),
        }

        checkpoint_path = os.path.join(
            checkpoints_dir, "{}_iterations.pth".format(epoch)
        )

        torch.save(checkpoint, checkpoint_path)
        pbar.update()
        for batch_data_dict in train_loader:
            """batch_data_dict: {
                'audio_name': (batch_size [*2 if mixup],), 
                'waveform': (batch_size [*2 if mixup], clip_samples), 
                'target': (batch_size [*2 if mixup], classes_num), 
                (ifexist) 'mixup_lambda': (batch_size * 2,)}
            """
            # Mixup lambda
            if "mixup" in augmentation:
                batch_data_dict["mixup_lambda"] = mixup_augmenter.get_lambda(
                    batch_size=len(batch_data_dict["waveform"])
                )

            # Move data to device
            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

            # Forward
            model.train()

            if "mixup" in augmentation:
                batch_output_dict = model(
                    batch_data_dict["waveform"], batch_data_dict["mixup_lambda"]
                )
                """{'clipwise_output': (batch_size, classes_num), ...}"""

                batch_target_dict = {
                    "target": do_mixup(
                        batch_data_dict["target"], batch_data_dict["mixup_lambda"]
                    )
                }
                """{'target': (batch_size, classes_num)}"""
            else:
                batch_output_dict = model(batch_data_dict["waveform"], None)
                """{'clipwise_output': (batch_size, classes_num), ...}"""
                # print(batch_output_dict)

                batch_target_dict = {"target": batch_data_dict["target"]}
                """{'target': (batch_size, classes_num)}"""
                # print(batch_target_dict)

            # Loss
            loss = loss_func(batch_output_dict, batch_target_dict)
            writer.add_scalar("Loss/train", loss, iteration)
            iteration += 1
            pbar.set_postfix(loss=loss)

            # Backward
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

    writer.flush()
    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use this program to train")

    parser.add_argument("--workspace", type=str, required=True)
    parser.add_argument(
        "--data_type",
        type=str,
        default="full_train",
        choices=["balanced_train", "full_train"],
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--hop_size", type=int, default=320)
    parser.add_argument("--mel_bins", type=int, default=64)
    parser.add_argument("--fmin", type=int, default=50)
    parser.add_argument("--fmax", type=int, default=14000)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument(
        "--loss_type", type=str, default="clip_bce", choices=["clip_bce"]
    )
    parser.add_argument(
        "--balanced",
        type=str,
        default="balanced",
        choices=["none", "balanced", "alternate"],
    )
    parser.add_argument(
        "--augmentation", type=str, default="none", choices=["none", "mixup"]
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--resume_iteration", type=int, default=0)
    parser.add_argument("--early_stop", type=int, default=400)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--train_csv_path")
    parser.add_argument("--classes_num", default=config.classes_num, type=int)

    args = parser.parse_args()
    args.filename = get_filename(__file__)

    train(
        args.workspace,
        args.data_type,
        args.sample_rate,
        args.window_size,
        args.hop_size,
        args.mel_bins,
        args.fmin,
        args.fmax,
        args.model_type,
        args.loss_type,
        args.balanced,
        args.augmentation,
        args.batch_size,
        args.learning_rate,
        args.resume_iteration,
        args.early_stop,
        args.cuda,
        args.train_csv_path,
        args.classes_num,
        args.filename,
    )
