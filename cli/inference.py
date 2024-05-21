import argparse
import csv
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import quanto
import torch
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from common import models  # noqa: F401


def one_hot_encode(input: int, start=1, max_classes=3) -> np.ndarray:
    res = []
    for i in range(max_classes):
        res.append(1 if input - start == i else 0)
    if input > max_classes:
        res[max_classes - 1] = 1
    return np.array(res, dtype=np.float_)


def truncate_classes(input_list: np.ndarray, max_classes=3) -> np.ndarray:
    input_list[max_classes - 1] = np.max(input_list[max_classes - 1 :])
    return np.resize(input_list, (max_classes,))


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
    quantize=False,
    sample_rate=16000,
    window_size=512,
    hop_size=160,
    mel_bins=64,
    fmin=0,
    fmax=8000,
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
        if not quantize:
            model = torch.nn.DataParallel(model)

    model.eval()

    if quantize:
        quanto.quantize(model, weights=quanto.qint8)
        quanto.freeze(model)

    return device, model


def infer_audio(
    audio_path: str,
    sample_rate: int,
    device: torch.device,
    model,
    label_names: list,
    verbose=False,
    audio_length: float = 5,  # defaults to 5 seconds
):
    # Load audio
    (waveform, _) = librosa.core.load(
        audio_path, sr=sample_rate, mono=True, duration=audio_length
    )

    waveform = waveform[None, :]  # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
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

            truth_list.append(one_hot_encode(int(line["label"])))

            infer_result = infer_audio(
                audio_path, sample_rate, device, model, label_names
            )

            score_list.append(truncate_classes(infer_result[0]))
            score_label_list.append(
                {
                    "path": audio_path,
                    "label": label_names.index(infer_result[1][0]),
                    "score": str(infer_result[0]).replace("\n", ""),
                }
            )

    pl.DataFrame(score_label_list).write_csv(f"{output_name}.csv")

    truth_np = np.array(truth_list)
    score_np = np.array(score_list)

    plot_confusion_matrix(
        np.argmax(truth_np, axis=1),
        np.array([a["label"] for a in score_label_list]),
        label_names,
        output_name,
    )

    # Output a 2x2 confusion matrix
    replace_dict = {1: 0, 2: 1, 0: 0}
    plot_confusion_matrix(
        np.vectorize(replace_dict.get)(np.argmax(truth_np, axis=1)),
        np.vectorize(replace_dict.get)(
            np.array([a["label"] for a in score_label_list])
        ),
        label_names,
        output_name + "-2x2",
    )

    plot_pr(truth_np, score_np, label_names, output_name)
    plot_roc(truth_np, score_np, label_names, output_name)

    print_recall_fa(
        np.argmax(truth_np, axis=1),
        np.array([a["label"] for a in score_label_list]),
        label_names,
    )


def print_recall_fa(truth_labels, prediction_labels, label_names):
    for i in range(np.max(truth_labels)):
        # Select label i as current pred target, and only calculate recall, fa for
        # predictions whose target should be i
        pred_tp_fn = prediction_labels[truth_labels == i]
        pred_tp = pred_tp_fn[pred_tp_fn == i]
        recall = len(pred_tp) / len(pred_tp_fn)
        print(f"Recall for label {label_names[i]}:\t{recall}")

    negative_index = np.max(truth_labels)
    pred_tn_fp = prediction_labels[truth_labels == negative_index]
    pred_fp = pred_tn_fp[pred_tn_fp != negative_index]
    fpr = len(pred_fp) / len(pred_tn_fp)
    print(f"False positive rate for label {label_names[negative_index]}:\t{fpr}")


def plot_confusion_matrix(truth_np, score_np, label_names, output_name):
    cm = confusion_matrix(truth_np, score_np)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot()
    plt.savefig(f"{output_name}-confusion.png")


def plot_roc(truth_np, score_np, label_names, output_name):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(truth_np.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(truth_np[:, i], score_np[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure()
    for i in range(truth_np.shape[1]):
        plt.plot(
            fpr[i],
            tpr[i],
            label="Class: {} (AUC = {:.2f})".format(label_names[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
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
    parser.add_argument(
        "--quantize", action="store_true", default=False, help="是否量化模型至int8"
    )

    args = parser.parse_args()
    label_names = ["Cough", "Humming"]
    for _ in range(10):
        label_names.append("Negative")

    device, model = get_infer_session(
        args.checkpoint_path,
        args.model_type,
        args.classes_num,
        label_names,
        quantize=args.quantize,
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
