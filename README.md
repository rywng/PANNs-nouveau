# PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition

This is a fork of PANNs, main differences are shown below:

- Code quality
  - Use relative imports instead of `sys.path.insert` (this is why python sucks)
  - Use a directory structure that makes sense, rather than one flat directory with everything inside, or even worse, two flat directories with interlinked dependency
- Redundant features removed
  - The whole hdf5 thing is removed, it doesn't make any sense when you can simply load waveforms from csvs. This complicates everything for devs and users.
  - No "black list csv" thing, you don't need it.
- Latest dependency, latest python, so that it's fast and works everywhere

## Install & Setup

Python 3.11 is used, higher versions might work, but there's no guarantees.

```bash
git clone https://github.com/rywng/PANNs-nouveau.git
cd PANNs-nouveau
pip install -r requirements.txt
```

## Usage

> [!TIP]
> Append `--help` flag to commands to see usage

### Prepare dataset

#### Train dataset

Should be the following format:

```csv
path,label
/<path_to_audio>/YhVTw6Xmi0oQ.wav,7
/<path_to_audio>/YZWM2LZFNEng.wav,10
/<path_to_audio>/Y0AWF9zOT8YY.wav,1
/<path_to_audio>/YvwdFUtLKZzU.wav,1
/<path_to_audio>/Y6RAfmkFCinY.wav,10
/<path_to_audio>/YIIWE3piniI8.wav,4
/<path_to_audio>/YGhFNQ7LRPos.wav,7
/<path_to_audio>/Yut_F-DG1hOY.wav,7
/<path_to_audio>/YuVGj89IopGY.wav,10
```

legend:

- Path: the path to the audio wav, the path is specified in feishu docs
- Label: the label of the audio, see feishu docs for more info

#### Test dataset

The format is same as training one, see feishu docs for more info.

```csv
path,label
/<path_to_audio>/audio_file_1.wav,12
/<path_to_audio>/audio_file_2.wav,1
/<path_to_audio>/audio_file_3.wav,1
/<path_to_audio>/audio_file_4.wav,7
/<path_to_audio>/audio_file_5.wav,4
/<path_to_audio>/audio_file_6.wav,3
/<path_to_audio>/audio_file_7.wav,7
/<path_to_audio>/audio_file_8.wav,3
/<path_to_audio>/audio_file_9.wav,2
```

### Train

Train model using the following command:

```bash
python -m cli.train --model_type Cnn10_BCE --loss_type clip_bce --train_csv_path ../data/metadata/12classes/processed/train.csv --classes_num 12
```

### Profile

Use `python -m cli.profile` to profile the model, for example:

```bash
python -m cli.profile workspaces/Cnn10_BCE/checkpoints/train/sample_rate=16000,window_size=512,hop_size=160,mel_bins=64,fmin=0,fmax=8000/data_type=full_train/loss_type=clip_bce/batch_size=288/classes_num=12/20240515-135456/1379_iterations.pth Cnn10_BCE 12  --save-quantized
```

### Inference

Use `python -m cli.inference` to benchmark the model against test data, for example:

```bash
p -m cli.inference ./workspaces/Cnn10_BCE/checkpoints/train/sample_rate=16000,window_size=512,hop_size=160,mel_bins=64,fmin=0,fmax=8000/data_type=full_train/loss_type=clip_bce/batch_size=288/classes_num=12/20240525-150100/591_iter.pth ../data/metadata/12classes/processed/test_filtered.csv Cnn10_BCE 12
```
