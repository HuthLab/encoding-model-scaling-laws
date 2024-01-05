# Encoding Model Scaling Laws
Repository for the 2023 NeurIPS paper "[Scaling laws for language encoding models in fMRI](https://arxiv.org/abs/2305.11863)".

![Encoding model performance for OPT-30B](https://github.com/HuthLab/encoding-model-scaling-laws/blob/main/corrs.png)

This repository provides feature extraction code, as well as encoding model features and weights from the analyses in the paper “Scaling Laws for Language Encoding Models in fMRI”.

The repository uses a [Box folder](https://utexas.box.com/v/EncodingModelScalingLaws) to host larger data files, including weights, response data, and features.

Please see the tutorial notebook or the boxnote for instructions on how to use the provided data. If you use this repository or any derivatives, please cite our paper:

```
@article{antonello2023scaling,
  title={Scaling laws for language encoding models in fMRI}, 
  author={Richard J. Antonello and Aditya R. Vaidya and Alexander G. Huth},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023}
}
```

## Speech models

Feature extraction from audio-based models is not as straightforward as for LMs because audio models are usually bidirectional, and because of this we created a separate feature extraction pipeline.
To maintain the causality of the features, we extract features from these models with a sliding window over the stimulus.
In this paper, the stride is 0.1 s and the size is 16.1 s.
At every iteration of the sliding window $[t-16.1, t]$, we select the output vector for the final "token" of the model’s output, and consider it _the_ feature vector for time $t$.
This ensures that features at time $t$ are only computed given the first $t$ seconds of audio.

Because feature extraction is more complex, it is broken out into a separate script: https://github.com/HuthLab/encoding-model-scaling-laws/blob/main/extract_speech_features.py .
The function `extract_speech_features` implements feature extraction (with striding, etc.) for a single audio file.
The rest of the script mainly handles stimulus selection and saving data.
(If you want to extract features without saving them, you can import this function into another script or notebook.)

Download the folder `story_data/story_audio` from the Box.
This command will then extract features from whisper-tiny for all audio files (using the above sliding window parameters), and save the features to the folder `features_cnk0.1_ctx16.0/whisper-tiny`:

```bash
python3 ./extract_speech_features.py --stimulus_dir story_audio/ --model whisper-tiny --chunksz 100 --contextsz 16000 --use_featext --batchsz 64
```

`chunksz` denotes the stride (in milliseconds), and the window size is `chunksz+contextsz`.
`--use_featext` passes the audio to the model's `FeatureExtractor` before a forward pass.
You can extract features for specific stories with the `--stories <stories>` option.

Features from each layer will be saved in a different directory. The script also saves the associated timestamps (i.e. `time[t]` is the time as which hidden state `features[t]` occurred).

Here is the directory structure after running the script with `--stories wheretheressmoke` (this takes ~10 min. on a Titan Xp):
```
$ tree features_cnk0.1_ctx16.0/

features_cnk0.1_ctx16.0/
└── whisper-tiny
    ├── encoder.0
    │   └── wheretheressmoke.npz
    ├── encoder.1
    │   └── wheretheressmoke.npz
    ├── encoder.2
    │   └── wheretheressmoke.npz
    ├── encoder.3
    │   └── wheretheressmoke.npz
    ├── encoder.4
    │   └── wheretheressmoke.npz
    ├── wheretheressmoke.npz
    └── wheretheressmoke_times.npz
```

### Using extracted speech features

As with word-level features, features from speech models must be downsampled to the rate of fMRI acquisition before being using in encoding models.
This code will downsample the features:

```python
from pathlib import Path

chunk_sz, context_sz = 0.1, 16.0
model = 'whisper-tiny'

base_features_path = Path(f"features_cnk{chunk_sz:0.1f}_ctx{context_sz:0.1f}/{model}")

story = 'wheretheressmoke'

times = np.load(base_features_path / f"{story}_times.npz")['times'][:,1] # shape: (time,)
features = np.load(base_features_path / f"{story}.npz")['features'] # shape: (time, model dim.)

# you will need `wordseqs` from the notebook
downsampled_features = lanczosinterp2D(features, times, wordseqs[story].tr_times)
```

`downsampled_features` can then be used like features from OPT or LLaMa.
(Note that features in the Box are already downsampled, so this step is not necessary.)
