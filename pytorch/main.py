import argparse
import random
import datetime
import logging
import os

import torch.optim as optim
import torch.utils.data
from pytorch.pytorch_utils import move_data_to_device, count_parameters
from utils import config
from utils.data_generator import AudioSetDatasetCsv, collate_fn, CsvTrainSampler
from utils.utilities import (
    create_folder,
    get_filename,
    create_logging,
)
from pytorch import models  # noqa: F401
import tqdm

from pytorch.losses import Loss_functions

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
    batch_size: int,
    learning_rate: float,
    resume_iteration: int,
    early_stop: int,
    cuda: bool,
    train_csv_path: str,
    classes_num: int,
    filename: str,
    audio_len_sec: int,
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
    loss_func = Loss_functions().get_loss_func(loss_type)

    # Paths
    postfix = os.path.join(
        filename,
        "sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}".format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax
        ),
        "data_type={}".format(data_type),
        "loss_type={}".format(loss_type),
        "balanced={}".format(balanced),
        "batch_size={}".format(batch_size),
        current_time,
    )
    checkpoints_dir = os.path.join(workspace, "checkpoints", postfix)
    create_folder(checkpoints_dir)

    statistics_dir = os.path.join(workspace, "statistics", postfix)
    create_folder(statistics_dir)

    logs_dir = os.path.join(workspace, "logs", postfix)

    create_logging(logs_dir, filemode="w")
    writer = SummaryWriter(log_dir=statistics_dir)

    # Model
    # Model = models.Cnn14
    Model = eval(f"models.{model_type}")
    assert model_type in str(
        Model.__name__
    ), f"Wrong model type: {model_type} and {str(Model.__name__)}"
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
    dataset = AudioSetDatasetCsv(
        classes_num, sample_rate=sample_rate, audio_len_sec=audio_len_sec
    )

    train_sampler = CsvTrainSampler(train_csv_path, batch_size)

    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.0,
        amsgrad=True,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    # Parallel
    print("Number of GPU available: {}".format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if "cuda" in str(device):
        model.to(device)

    pbar = tqdm.tqdm(total=early_stop)
    iteration = 0

    for epoch in range(early_stop):
        for batch_data_dict in train_loader:
            """batch_data_dict: {
                'audio_name': (batch_size,), 
                'waveform': (batch_size, clip_samples), 
                'target': (batch_size, classes_num), 
            """
            optimizer.zero_grad()

            # Move data to device
            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

            # Forward
            model.train()

            batch_output_dict = model(batch_data_dict["waveform"], None)
            batch_target_dict = {"target": batch_data_dict["target"]}

            # Loss
            loss = loss_func(batch_output_dict, batch_target_dict)
            writer.add_scalar("Loss/train", float(loss), iteration)
            pbar.set_postfix(loss=float(loss), iteration=iteration)

            # Backward
            loss.backward()
            optimizer.step()

            if iteration % 100 == 0:
                random_loc = random.randint(0, batch_size - 1)
                writer.add_audio(
                    "Prediction/audio",
                    batch_data_dict["waveform"][random_loc].reshape(1, -1),
                    iteration,
                    sample_rate=sample_rate,
                )
                writer.add_text(
                    "Prediction/audio",
                    f"target: {batch_data_dict['target'][random_loc]}\nprediction: {batch_output_dict['clipwise_output'][random_loc]}\nname: {batch_data_dict['audio_name'][random_loc]}",
                    iteration,
                )

            iteration += 1

        scheduler.step()
        pbar.update()
        save_checkpoint(iteration, model, checkpoints_dir)
        writer.add_scalar("Epoch/Iteration", epoch, iteration)

    writer.flush()
    pbar.close()


def save_checkpoint(iteration, model, checkpoints_dir):
    checkpoint = {
        "iteration": iteration,
        "model": model.module.state_dict(),
    }

    checkpoint_path = os.path.join(
        checkpoints_dir, "{}_iterations.pth".format(iteration)
    )

    torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use this program to train")

    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument(
        "--data_type",
        type=str,
        default="full_train",
        choices=["balanced_train", "full_train"],
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument("--hop_size", type=int, default=128)
    parser.add_argument("--mel_bins", type=int, default=64)
    parser.add_argument("--fmin", type=int, default=50)
    parser.add_argument("--fmax", type=int, default=14000)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument(
        "--loss_type", type=str, default="ce", choices=["clip_bce", "ce"]
    )
    parser.add_argument(
        "--balanced",
        type=str,
        default="balanced",
        choices=["none", "balanced", "alternate"],
    )
    parser.add_argument("--batch_size", type=int, default=288)  # 12 * 24
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--resume_iteration", type=int, default=0)
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--train_csv_path", default="../data/metadata/train.csv")
    parser.add_argument(
        "--classes_num", default=config.classes_num, type=int
    )  # Change this when doing classification
    parser.add_argument("--audio_len_sec", default=10, type=int)

    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.workspace is None:
        args.workspace = os.path.join("workspaces", args.model_type)

    if args.cuda:
        # Free performance boost, since input size is static
        torch.backends.cudnn.benchmark = True

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
        args.batch_size,
        args.learning_rate,
        args.resume_iteration,
        args.early_stop,
        args.cuda,
        args.train_csv_path,
        args.classes_num,
        args.filename,
        args.audio_len_sec,
    )
