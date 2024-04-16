import argparse
import os

import librosa
import numpy as np
import torch
import polars as pl

from pytorch import models  # noqa: F401


def move_data_to_device(x, device):
    if "float" in str(x.dtype):
        x = torch.Tensor(x)
    elif "int" in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def get_infer_session(
    checkpoint_path,
    model_type,
    classes_num,
    labels,
    sample_rate=16000,
    window_size=512,
    hop_size=128,
    mel_bins=64,
    fmin=50,
    fmax=14000,
):
    """Inference audio tagging result of an audio clip."""

    # Arugments & parameters
    device = (
        torch.device("cuda")
        if args.cuda and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Model
    model_class = eval(f"models.{model_type}")
    model = model_class(
        sample_rate=sample_rate,
        window_size=window_size,
        hop_size=hop_size,
        mel_bins=mel_bins,
        fmin=fmin,
        fmax=fmax,
        classes_num=classes_num,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    if "cuda" in str(device):
        model.to(device)
        model = torch.nn.DataParallel(model)

    return device, model


def infer_audio(
    audio_path: str,
    sample_rate: int,
    device: torch.device,
    model,
    labels: list,
    verbose=False,
):
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]  # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    clipwise_output = batch_output_dict["clipwise_output"].data.cpu().numpy()[0]

    sorted_indexes = np.argsort(clipwise_output)[::-1]

    pred_labels = []
    # Print audio tagging top probabilities
    for k in range(3):
        if verbose:
            print(
                "{}: {:.3f}".format(
                    np.array(labels)[sorted_indexes[k]],
                    clipwise_output[sorted_indexes[k]],
                )
            )
        pred_labels.append(str(np.array(labels)[sorted_indexes[k]]))

    return clipwise_output, pred_labels


def infer_directory(
    dirpath: str, sample_rate: int, device: torch.device, model, labels: list
):
    res_list = []
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        _, pred_label = infer_audio(filepath, sample_rate, device, model, labels)
        pred_label = pred_label[0]  # get the most likely one
        res_list.append({"path": filepath, "pred_label": pred_label})

    res_df = pl.DataFrame(res_list)
    print("Statistics:")
    for label in labels[:3]:
        print(f"{label}: {res_df.filter(pl.col('pred_label') == label).shape[0]}")

    save_path = f"prediction-{dirpath.replace('/', '-')}.csv"
    print(f"Saving to {save_path}")
    res_df.write_csv(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="infer audio given model checkpoint")

    parser.add_argument("checkpoint_path", type=str, help="模型检查点位置")
    parser.add_argument("audio_path", type=str, help="输入音频位置")
    parser.add_argument("model_type", type=str, help="模型类型")
    parser.add_argument("classes_num", type=int, help="分类数量")
    parser.add_argument(
        "--cuda", action="store_true", default=True, help="是否使用cuda"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="模型和输入音频的采样率，音频会自动被重采样",
    )

    args = parser.parse_args()
    labels = ["Cough", "Humming"]
    for _ in range(10):
        labels.append("Negative")

    device, model = get_infer_session(
        args.checkpoint_path, args.model_type, args.classes_num, labels
    )

    if os.path.isdir(args.audio_path):
        infer_directory(args.audio_path, args.sample_rate, device, model, labels)
    elif args.audio_path.endswith(".csv"):
        pass
    else:
        print(
            infer_audio(
                args.audio_path, args.sample_rate, device, model, labels, verbose=True
            )
        )
