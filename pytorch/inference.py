import argparse
import csv
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from pytorch import models  # noqa: F401


def three_classes_one_hot_encode(input: int, start=1) -> np.ndarray:
    res = []
    for i in range(3):
        res.append(1 if input - start == i else 0)
    if input > 3:
        res[2] = 1
    return np.array(res, dtype=np.float_)


def convert_to_3classes(input_list: np.ndarray) -> np.ndarray:
    input_list[2] = np.max(input_list[2:])
    return np.resize(input_list, (3,))


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
    label_names,
    sample_rate=16000,
    window_size=400,
    hop_size=160,
    mel_bins=64,
    fmin=20,
    fmax=7800,
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
    label_names: list,
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
                    np.array(label_names)[sorted_indexes[k]],
                    clipwise_output[sorted_indexes[k]],
                )
            )
        pred_labels.append(str(np.array(label_names)[sorted_indexes[k]]))

    return clipwise_output, pred_labels


def infer_directory(
    dirpath: str, sample_rate: int, device: torch.device, model, label_names: list
):
    res_list = []
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        _, pred_label = infer_audio(filepath, sample_rate, device, model, label_names)
        pred_label = pred_label[0]  # get the most likely one
        res_list.append({"path": filepath, "pred_label": pred_label})

    res_df = pl.DataFrame(res_list)
    print("Statistics:")
    for label in label_names[:3]:
        print(f"{label}: {res_df.filter(pl.col('pred_label') == label).shape[0]}")

    save_path = f"prediction-{dirpath.replace('/', '-')}.csv"
    print(f"Saving to {save_path}")
    res_df.write_csv(save_path)


def infer_csv(
    csv_path: str,
    sample_rate: int,
    device: torch.device,
    model,
    label_names: list,
    output_name: str,
):
    # TODO: refactor code

    truth_list = []
    score_list = []

    score_label_list = []  # This one doesn't use one-hot encoding

    with open(csv_path) as in_csv:
        reader = csv.DictReader(in_csv)
        for line in reader:
            audio_path = line["path"]

            truth_list.append(three_classes_one_hot_encode(int(line["label"])))

            infer_result = infer_audio(
                audio_path, sample_rate, device, model, label_names
            )

            score_list.append(convert_to_3classes(infer_result[0]))
            score_label_list.append({"path": audio_path, "label": infer_result[1][0]})

    pl.DataFrame(score_label_list).write_csv(f"{output_name}.csv")

    # Compute precision and recall for each class
    truth_np = np.array(truth_list)
    score_np = np.array(score_list)

    plot_pr(truth_np, score_np, label_names, output_name)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(truth_np.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(truth_np[:, i], score_np[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure()
    for i in range(truth_np.shape[1]):
        plt.plot(fpr[i], tpr[i], label='Class: {} (AUC = {:.2f})'.format(label_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="best")
    plt.savefig(f"{output_name}-ROC.png")


def plot_pr(truth_np, score_np, label_names, output_name):
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(truth_np.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(
            truth_np[:, i], score_np[:, i]
        )
        average_precision[i] = average_precision_score(truth_np[:, i], score_np[:, i])

    # Plot precision-recall curves for each class
    plt.figure()
    for i in range(truth_np.shape[1]):
        plt.plot(
            recall[i],
            precision[i],
            label="Class: {} (AP = {:.2f})".format(
                label_names[i], average_precision[i]
            ),
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        "Precision-Recall curve (mAP = {:.3f})".format(
            np.mean([i for i in average_precision.values()])
        )
    )
    plt.legend(loc="best")
    plt.savefig(f"{output_name}-PR.png")


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
    label_names = ["Cough", "Humming"]
    for _ in range(10):
        label_names.append("Negative")

    device, model = get_infer_session(
        args.checkpoint_path, args.model_type, args.classes_num, label_names
    )

    if os.path.isdir(args.audio_path):
        infer_directory(args.audio_path, args.sample_rate, device, model, label_names)
    elif args.audio_path.endswith(".csv"):
        infer_csv(
            args.audio_path,
            args.sample_rate,
            device,
            model,
            label_names,
            f"{args.model_type}-{args.classes_num}",
        )
    else:
        print(
            infer_audio(
                args.audio_path,
                args.sample_rate,
                device,
                model,
                label_names,
                verbose=True,
            )
        )
