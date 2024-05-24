import torch
from torch.profiler import profile, record_function, ProfilerActivity
import argparse

from cli.inference import get_infer_session

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Profiler")
    parser.add_argument("checkpoint_path")
    parser.add_argument("model_type")
    parser.add_argument("classes_num")
    parser.add_argument(
        "--quantize",
        "-q",
        help="Whether or not quantize the model before execution",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    device, model = get_infer_session(
        args.checkpoint_path, args.model_type, args.classes_num
    )

    input = torch.randn(1, 5 * 16000)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        with record_function("model_inference"):
            model(input)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
