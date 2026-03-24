<p align="center">
  <img src="docs/image/utmosv2.PNG" alt="utmosv2" width=500>
</p>

<h1 align="center">
  UTMOSv2: UTokyo-SaruLab MOS Prediction System
  <a href="https://github.com/sarulab-speech/UTMOSv2">
    <img width="94%" height="5px" src="docs/image/titleLine.svg">
  </a>
</h1>

<p align="center">
  🎤✨ Official implementation of ✨🎤<br>
  “<a href="http://arxiv.org/abs/2409.09305">The T05 System for The VoiceMOS Challenge 2024:</a><br>
  <a href="http://arxiv.org/abs/2409.09305">Transfer Learning from Deep Image Classifier to Naturalness MOS Prediction of High-Quality Synthetic Speech</a>”<br>
  🏅🎉&ensp;accepted by IEEE Spoken Language Technology Workshop (SLT) 2024.&ensp;🎉🏅
</p>

<p align="center">
  ꔫ･-･ꔫ･-･ꔫ･-･ꔫ･-･ꔫ･-･ꔫ･-･ꔫ･-･ꔫ
</p>

<p align="center">
  ✨&emsp;&emsp;UTMOSv2 achieved 1st place in 7 out of 16 metrics&emsp;&emsp;✨<br>
  ✨🏆&emsp;&emsp;&emsp;&emsp;and 2nd place in the remaining 9 metrics&emsp;&emsp;&emsp;&emsp;🏆✨<br>
  ✨&emsp;&emsp;&emsp;&emsp;in the <a href="https://sites.google.com/view/voicemos-challenge/past-challenges/voicemos-challenge-2024">VoiceMOS Challenge 2024</a> Track1!&emsp;&emsp;&emsp;&emsp;✨
</p>

<div align="center">
  <a target="_blank" href="https://www.python.org">
    <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python"/>
  </a>
</div>

<div  align="center">
  <a target="_blank" href="https://huggingface.co/spaces/sarulab-speech/UTMOSv2">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" alt="Hugging Face Spaces"/>
  </a>
  <a target="_blank" href="https://colab.research.google.com/github/sarulab-speech/UTMOSv2/blob/main/quickstart.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</div>

<div  align="center">
  <a target="_blank" href="http://arxiv.org/abs/2409.09305">
    <img src="https://img.shields.io/badge/arXiv-2409.09305-b31b1b.svg" alt="arXiv"/>
  </a>
  <a target="_blank" href="https://ieeexplore.ieee.org/document/10832315">
    <img src="https://img.shields.io/badge/IEEE%20Xplore-10832315-blue.svg" alt="poster"/>
  </a>
  <a target="_blank" href="https://github.com/sarulab-speech/UTMOSv2/blob/main/poster.pdf">
    <img src="https://img.shields.io/badge/IEEE%20SLT%202024-Poster-blue.svg" alt="poster"/>
  </a>
</div>

<br>

<h2 align="left">
  <div>🚀 Quick Prediction</div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/tree/main?tab=readme-ov-file#---quick-prediction--------">
    <img width="85%" height="6px" src="docs/image/line2.svg">
  </a>
</h2>

✨ You can easily use the pretrained UTMOSv2 model!

<h3 align="center">
  <div>🛠️ Using in your Python code 🛠️</div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/tree/doc-user-friendly-api?tab=readme-ov-file#--%EF%B8%8F-using-in-your-python-code-%EF%B8%8F--------">
    <img width="70%" height="6px" src="docs/image/line3.svg">
  </a>
</h3>

<div align="center">
✨⚡️&emsp;With the UTMOSv2 library, you can easily integrate it into your Python code,&emsp;⚡️✨<br>
✨&ensp;allowing you to quickly create models and make predictions with minimal effort!!&ensp;✨
</div>

<br>

If you want to make predictions using the UTMOSv2 library, follow these steps:

1. Install the UTMOSv2 library from GitHub

   ```bash
   uv add git+https://github.com/Scicom-AI-Enterprise-Organization/faster-UTMOSv2
   # If you're using pip:
   # pip install git+https://github.com/Scicom-AI-Enterprise-Organization/faster-UTMOSv2
   ```

2. Make predictions
   - To predict the MOS of a tensor or array already loaded in memory:

      ```python
      import utmosv2
      model = utmosv2.create_model(pretrained=True)
      # data: torch.Tensor or np.ndarray with shape (batch_size, sequence_length) or (sequence_length,)
      # sr: Sampling rate of the input audio data. If not provided, it defaults to 16000 Hz.
      mos = model.predict(data=data, sr=16000) # Returns a torch.Tensor or np.ndarray with shape (batch_size,) or (1,)
      ```

   - To predict the MOS of a single `.wav` or `.mp3` file:

      ```python
      import utmosv2
      model = utmosv2.create_model(pretrained=True)
      mos = model.predict(input_path="/path/to/file.wav")  # or .mp3 — returns a float
      ```

   - To predict the MOS of all `.wav` / `.mp3` files in a folder:

      ```python
      import utmosv2
      model = utmosv2.create_model(pretrained=True)
      mos = model.predict(input_dir="/path/to/audio/dir/")  # returns list of dicts with 'file_path' and 'predicted_mos'
      ```

> [!NOTE]
> When `data` is provided, `input_path` and `input_dir` are ignored.

> [!NOTE]
> Either `input_path` or `input_dir` must be specified when `data` is `None`, but not both.

<h3 align="center">
  <div>📜 Using the inference script 📜</div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/tree/doc-user-friendly-api?tab=readme-ov-file#---using-the-inference-script---------">
    <img width="70%" height="6px" src="docs/image/line3.svg">
  </a>
</h3>

If you want to make predictions using the inference script, follow these steps:

1. Clone this repository and navigate to UTMOSv2 folder

   ```bash
   git clone https://github.com/Scicom-AI-Enterprise-Organization/faster-UTMOSv2
   cd UTMOSv2
   ```

2. Install Package

   ```bash
   uv sync --extra optional
   # If you're using pip:
   # pip install --upgrade pip  # enable PEP 660 support
   # pip install -e .[optional] # install with optional dependencies
   ```

3. Make predictions
   - To predict the MOS of a single `.wav` file:

      ```bash
      python inference.py --input_path /path/to/wav/file.wav --out_path /path/to/output/file.csv
      ```

   - To predict the MOS of all `.wav` files in a folder:

      ```bash
      python inference.py --input_dir /path/to/wav/dir/ --out_path /path/to/output/file.csv
      ```

> [!NOTE]
> If you are using zsh, make sure to escape the square brackets like this:
>
> ```zsh
> pip install -e '.[optional]'
> ```

> [!TIP]
> If `--out_path` is not specified, the prediction results will be output to the standard output. This is particularly useful when the number of files to be predicted is small.

> [!NOTE]
> Either `--input_path` or `--input_dir` must be specified, but not both.

<br>

> [!NOTE]
> These methods provide quick and simple predictions. For more accurate predictions and detailed usage of the inference script, please refer to the [inference guide](docs/inference.md).

🤗 You can try a simple demonstration on Hugging Face Space:
<a href="https://huggingface.co/spaces/sarulab-speech/UTMOSv2">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" alt="Hugging Face Spaces" align="top">
</a>

<h2 align="left">
  <div>⚡ Performance &amp; Stress Test</div>
  <a href="#-performance--stress-test">
    <img width="85%" height="6px" src="docs/image/line2.svg">
  </a>
</h2>

### Benchmark Results

Measured on **1194 × 10 s audio clips** (Elise dataset, `fusion_stage3` config, fold 0, `num_repetitions=1`) using a single **NVIDIA GeForce RTX 3090 Ti**, PyTorch 2.1.2 + CUDA 11.8.

```bash
python3 stress_test.py --num_files 50 --compile --num_frames 1
```

```
========================== SUMMARY ===========================
  Test                                        Files   Elapsed  Files/s     RTF
  ----------------------------------------------------------------------------
  Single WAV                                      1     0.27s     3.70   37.0x
  Single MP3                                      1     0.27s     3.66   36.6x
  Batch WAV 50 files (per-file)                  50    13.54s     3.69   27.8x
  Batch MP3 50 files (per-file)                  50    13.99s     3.57   26.9x
  Batch WAV dir (input_dir API)                  50    10.64s     4.70   35.4x
  Batch MP3 dir (input_dir API)                  50     8.84s     5.66   42.6x
```

> [!TIP]
> Use `model.predict(input_dir=...)` instead of a per-file loop for best throughput — it amortises DataLoader startup across the whole batch.

> [!NOTE]
> Score variance between runs on the same file (std ≈ 0.22) is expected: with `num_repetitions=1` the model randomly crops a different window each time. Set `num_repetitions=5` for more stable scores at the cost of ~5× inference time.

#### Accuracy

1. Run UTMOSv2 using current implementation,

```bash
python3 benchmark_accuracy.py --audio_dir Elise_audio --num_files 100 \
--num_repetitions 20 --out ours.csv
```

2. Run UTMOSv2 using original implementation,

```bash
pip3 install git+https://github.com/sarulab-speech/UTMOSv2 --target /tmp/utmos_orig --no-deps
python3 benchmark_accuracy.py --audio_dir Elise_audio --num_files 100 \
--num_repetitions 20 --out orig.csv --use_upstream /tmp/utmos_orig
```

3. Compare,

```bash
python3 benchmark_accuracy.py --compare ours.csv orig.csv
```

```
Comparing A  vs  B
  Common files   : 50
  Pearson r      : 0.982328  (1.000 = perfect agreement)
  MAE            : 0.085449
  RMSE           : 0.105360
  Mean bias      : +0.004355  (A minus B)

  ✗ Notable differences — check which changes affect predictions.

  Largest discrepancies (top 10):
  File                                                      A        B         Δ
  Elise-data-train-00000-of-00001_1026                 1.5771   1.3418   +0.2354
  Elise-data-train-00000-of-00001_1002                 2.3203   2.0938   +0.2266
  Elise-data-train-00000-of-00001_1001                 1.1816   1.4004   -0.2188
  Elise-data-train-00000-of-00001_1030                 2.6836   2.9004   -0.2168
  Elise-data-train-00000-of-00001_1027                 2.6641   2.4785   +0.1855
  Elise-data-train-00000-of-00001_1036                 3.0879   2.9102   +0.1777
  Elise-data-train-00000-of-00001_1005                 1.8662   1.6943   +0.1719
  Elise-data-train-00000-of-00001_1038                 2.8242   2.9863   -0.1621
  Elise-data-train-00000-of-00001_102                  1.9141   2.0664   -0.1523
  Elise-data-train-00000-of-00001_1021                 1.1289   1.2637   -0.1348
```

### Score consistency (same WAV file, 5 independent runs)

```
Scores: [2.15, 1.52, 1.65, 1.73, 1.56]   std = 0.22
```

Use `num_repetitions=5` (or higher) to average out the random crop variance.

### WAV vs MP3 agreement

Both formats are supported. Because each run uses a random audio crop (TTA), scores for the same content in WAV and MP3 will differ slightly even on identical content:

```
Mean |WAV score − MP3 score| ≈ 0.36  (10 matched pairs, num_repetitions=1)
```

This gap closes to near zero when `num_repetitions` is increased.

### Running the stress test yourself

```bash
python3 stress_test.py                        # all 1194 files, default settings
python3 stress_test.py --num_files 50         # quick run with 50 files
python3 stress_test.py --num_files 50 --compile  # enable torch.compile (PyTorch >= 2.0)
python3 stress_test.py --num_files 50 --compile --num_frames 1
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--audio_dir` | `Elise_audio` | Directory of `.wav` / `.mp3` files |
| `--num_files` | all | Max files per format |
| `--num_workers` | 4 | DataLoader workers |
| `--batch_size` | 8 | Inference batch size |
| `--num_repetitions` | 1 | TTA repetitions |
| `--num_frames` | default (2) | Override spec frames per sample (e.g. `1` for ~4.7× GPU speedup) |
| `--compile` | off | Enable `torch.compile` |

### Speed vs. accuracy trade-off: `num_frames`

The `num_frames` parameter controls how many spectrogram frames EfficientNetV2 processes per audio sample. The default is 2 (from the config), giving the best accuracy. Setting `num_frames=1` halves the number of images from 64 to 8 per batch, reducing EfficientNetV2 GPU time by ~4.7× with a small accuracy trade-off:

```python
model = utmosv2.create_model()
# Fast mode: ~4.7x GPU speedup, slight accuracy drop
mos = model.predict(input_dir="...", num_frames=1)
# Default: full accuracy
mos = model.predict(input_dir="...")
```

<h2 align="left">
  <div>⚒️ Train UTMOSv2 Yourself</div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/tree/main?tab=readme-ov-file#--%EF%B8%8F-train-utmosv2-yourself--------">
    <img width="85%" height="6px" src="docs/image/line2.svg">
  </a>
</h2>

If you want to train UTMOSv2 yourself, please refer to the [training guide](docs/training.md). To reproduce the training as described in the paper or used in the competition, please refer to [this document](docs/reproduction.md).

<h2 align="left">
  <div>📂 Used Datasets</div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/tree/main?tab=readme-ov-file#---used-datasets--------">
    <img width="85%" height="6px" src="docs/image/line2.svg">
  </a>
</h2>

Details of the datasets used in this project can be found in the [datasets documentation](docs/datasets.md).

<h2 align="left">
  <div>🔖 Citation</div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/tree/main?tab=readme-ov-file#---citation--------">
    <img width="85%" height="6px" src="docs/image/line2.svg">
  </a>
</h2>

If you find UTMOSv2 useful in your research, please cite the following paper:

```bibtex
@inproceedings{baba2024utmosv2,
  title     = {The T05 System for The {V}oice{MOS} {C}hallenge 2024: Transfer Learning from Deep Image Classifier to Naturalness {MOS} Prediction of High-Quality Synthetic Speech},
  author    = {Baba, Kaito and Nakata, Wataru and Saito, Yuki and Saruwatari, Hiroshi},
  booktitle = {IEEE Spoken Language Technology Workshop (SLT)},
  year      = {2024},
  pages     = {818--824},
  doi       = {10.1109/SLT61566.2024.10832315},
}
```

<h2 align="left">
  <div>:octocat: GitHub Star History</div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/tree/main?tab=readme-ov-file#--octocat-github-star-history--------">
    <img width="85%" height="6px" src="docs/image/line2.svg">
  </a>
</h2>


<div align="center">
  <img width="90%" src="https://starchart.cc/sarulab-speech/UTMOSv2.svg?variant=adaptive" alt="GitHub Star History"/>
</div>
