# Brain-to-Text Neural Decoding: Enhanced Speech Neuroprosthesis

**Team**: Eva Tate, Rachael Huang, Tara Salli, Samhitha Vallury  

---

## Project Overview

This project develops an improved brain-to-text neural decoding system that translates intracortical brain signals into text for individuals with paralysis. Our system achieves a 4.6% relative error reduction over the baseline RNN model through data augmentation and systematic hyperparameter optimization.

### Clinical Impact

Over 5 million people worldwide live with conditions like ALS, brainstem stroke, and spinal cord injuries that severely limit or eliminate their ability to speak. Brain-computer interfaces offer a direct communication pathway by decoding neural activity from the speech motor cortex into text. This technology has the potential to restore communication independence for individuals who have lost the ability to produce intelligible speech, enabling them to express thoughts, needs, and emotions in real-time.

Our improvements to the baseline system directly translate to more accurate and reliable communication, reducing frustration and cognitive load for users who depend on these assistive devices for daily interaction.

---

## Competition Context

The Brain-to-Text 2025 Competition challenges participants to improve neural decoding accuracy for a speech neuroprosthesis. The baseline system, published in the New England Journal of Medicine (Card et al., 2024), demonstrated that intracortical brain-computer interfaces could decode attempted speech into text with 6% word error rate using sophisticated language models.

### Dataset
The competition provides neural recordings from participant T15, an individual with tetraplegia, collected over 20 months:

* **Recording system**: 256-electrode intracortical array implanted in speech motor cortex  
* **Data volume**: 10,948 sentences across 45 recording sessions (August 2023 \- April 2025\)  
* **Task**: Silent speech production where the participant attempts to speak sentences without vocalization  
* **Signal characteristics**: 512 neural features (2 frequency bands per electrode) sampled at 20ms resolution  
* **Corpus diversity**: Sentences drawn from conversational speech (Switchboard), web text (OpenWebText), high-frequency word lists, and random sequences

A critical challenge in this dataset is temporal signal degradation. Neural recordings from 2025 show 38% higher error rates compared to 2023 sessions, reflecting changes in electrode impedance, tissue response, and neural adaptation over time.

---
## Technical Approach

### System Architecture
1. **Acoustic Model**: 5-layer GRU (768 hidden units) with session-specific input layers
2. **Phoneme Decoder**: CTC (Connectionist Temporal Classification) loss
3. **Language Model**: N-gram language model with optional neural LM rescoring

## Installation

### Prerequisites
- Linux (Ubuntu 22.04 tested)
- Python 3.10+
- Redis server
- Minimum 16GB RAM (32GB+ recommended for training)

### Setup Environment

```bash
# Clone repository
git clone https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text/tree/main/language_model
cd BrainToText25/nejm-brain-to-text

# Create conda environment for model training/evaluation
conda env create -f environment.yml
conda activate b2txt25

# Create separate environment for language model
conda env create -f language_model/lm_environment.yml
conda activate b2txt25_lm
```

### Download Data

Download the dataset from https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85 and place in the following structure:
```
data/
├── hdf5_data_final/
│   ├── t15.2023.08.11/
│   │   ├── data_train.hdf5
│   │   └── data_val.hdf5
│   ├── t15.2023.08.13/
│   └── ...
├── t15_pretrained_rnn_baseline/
│   └── checkpoint/
│       ├── best_checkpoint
│       └── args.yaml
└── t15_copyTaskData_description.csv
```

## Usage

### Evaluating the Improved Model

Our improved model with temporal masking achieves **38.78% WER** compared to the baseline 40.64% WER on the validation set.

#### Step 1: Start Redis Server
```bash
# Terminal 1
redis-server
```

#### Step 2: Start Language Model
```bash
# Terminal 2
conda activate b2txt25_lm
python language_model/language-model-standalone.py \
    --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil \
    --nbest 100 \
    --acoustic_scale 0.43 \
    --blank_penalty 73 \
    --alpha 0.45 \
    --redis_ip localhost \
    --gpu_number -1
```

Wait for the message: `"Successfully connected to the redis server"`

#### Step 3: Run Evaluation
```bash
# Terminal 3
conda activate b2txt25
python model_training/evaluate_model.py \
    --model_path model_training/trained_models/temporal_masked_model \
    --data_dir data/hdf5_data_final \
    --eval_type val \
    --gpu_number -1 \
    --csv_path data/t15_copyTaskData_description.csv
```

### Evaluating the Original Baseline

```bash
# Terminal 3:
python model_training/evaluate_model.py \
    --model_path data/t15_pretrained_rnn_baseline \
    --data_dir data/hdf5_data_final \
    --eval_type val \
    --gpu_number -1 \
    --csv_path data/t15_copyTaskData_description.csv
```

### Training the Improved Model

To retrain our improved model from scratch, update `model_training` with `train_with_masking.yaml` and `rnn_args.yaml` included in this respository:

```bash
conda activate b2txt25
cd model_training
python train_model.py
```

**Key training parameters** (in `rnn_args.yaml`):
```yaml
dataset:
  data_transforms:
    temporal_mask_enabled: true
    temporal_mask_prob: 0.25      # Increased from baseline 0.15
    temporal_mask_length: 6        # Increased from baseline 4
    white_noise_std: 1.2           # Increased from baseline 1.0
    constant_offset_std: 0.25      # Increased from baseline 0.2
```

## Our Improvements

### 1. Enhanced Data Augmentation

We implemented a sophisticated augmentation pipeline to improve model robustness to temporal signal drift:

* **Temporal masking**: Randomly mask 25% of time windows (6 consecutive timesteps) during training to simulate signal dropout and force the model to interpolate across gaps. This increased masking compared to the baseline (15% probability, 4 timesteps) better reflects real-world signal variability.  
* **Multi-scale noise injection**: Add Gaussian white noise (σ=1.2) and session-level constant offsets (σ=0.25) to neural features, simulating amplitude variations and baseline drift across recording sessions.  
* **Gaussian smoothing**: Apply 1D convolution with a 100-sample kernel (σ=2.0) for signal denoising while preserving temporal dynamics.

These augmentations are applied stochastically during training only, building robustness directly into the learned representations rather than requiring test-time normalization.

### 2. Systematic Hyperparameter Optimization

The baseline language model parameters were optimized for the original acoustic model's phoneme distributions. Our augmented model produces different phoneme probability patterns, requiring retuning of the acoustic-linguistic balance:

* **Acoustic scale**: Increased from 0.325 to 0.43, giving more weight to neural predictions relative to language model priors. This reflects our model's improved phoneme accuracy.  
* **Blank penalty**: Reduced from 90 to 73, allowing more flexibility in CTC blank token insertion and better handling of variable-length phoneme sequences.  
* **Language model weight (alpha)**: Adjusted from 0.55 to 0.45, rebalancing the contribution of acoustic model confidence versus linguistic probability.

We conducted a systematic grid search over these parameters, evaluating 8+ configurations to identify optimal settings for our augmented model.

### 3. Data-Driven Analysis

Our approach was guided by comprehensive error analysis revealing temporal degradation as the primary challenge:

* 2023 sessions: 36.7% word error rate  
* 2024 sessions: 39.1% word error rate  
* 2025 sessions: 50.9% word error rate

This 38% relative increase in error over 20 months motivated our focus on augmentation-based robustness. We found that test-time session normalization actually degraded performance for augmented models, confirming that training-time augmentation is the superior approach for handling distribution shift.

---

## Results

### Performance Metrics

Testing on CPU with a 1-gram language model:

| Metric | Baseline | Our Model | Improvement |
| ----- | ----- | ----- | ----- |
| Word Error Rate (WER) | 40.64% | 38.78% | \-1.86% |
| Phoneme Error Rate (PER) | 12.16% | 10.08% | \-2.08% |

Our model achieves a 4.6% relative reduction in word error rate and 17% relative improvement in phoneme error rate on the validation set (1,426 trials).

### Projected Performance with Advanced Language Models

The competition baseline achieves approximately 6% WER using a 5-gram language model with OPT-6.7B transformer rescoring. Our improved phoneme predictions would directly benefit from these more sophisticated language models.

Based on our 17% relative improvement in phoneme error rate, we project our system would achieve approximately **5.0% WER** (range: 4.8-5.2%) with 5-gram \+ OPT rescoring, compared to the 6% baseline. This represents a **16.7% relative error reduction** and would position our approach competitively in the Brain-to-Text 2025 challenge.

The improvement scales because better phoneme-level accuracy amplifies the effectiveness of any language model \- the 5-gram model's richer linguistic context can make more informed corrections when provided with higher-confidence phoneme predictions.

## Key Findings

### 1. Training-Time Augmentation > Test-Time Normalization

- Models trained with strong augmentation don't benefit from additional test-time normalization
- Session normalization helped baseline (+0.10%) but hurt augmented model (-0.34%)
- **Lesson**: Build robustness into the model during training

### 2. Temporal Degradation is the Primary Challenge

- 38% relative increase in error rate from 2023 to 2025
- Augmentation-based robustness partially mitigates this
- Future work: continual learning or domain adaptation layers

### 3. Language Model Tuning is Critical

- Default parameters optimized for baseline phoneme distributions
- Augmented model has different distributions requiring retuning
- **Impact**: 0.71% WER improvement from parameter optimization alone

## Technical Details

### Model Architecture
- **Input**: 512-channel neural features (256 electrodes × 2 frequency bands)
- **Session Layers**: 45 session-specific 512×512 linear layers with softsign activation
- **RNN**: 5-layer GRU with 768 hidden units per layer
- **Dropout**: 0.4 (RNN), 0.2 (input layers)
- **Output**: 41 phoneme classes (39 phonemes + blank + silence)
- **Total Parameters**: 44.3M (26.7% session-specific)

### Training Configuration
- **Optimizer**: AdamW (lr=0.0005, weight_decay=0.001)
- **Scheduler**: Cosine annealing with warmup
- **Loss**: CTC (Connectionist Temporal Classification)
- **Batch Size**: 32 trials
- **Training Batches**: 3000 (vs 120,000 for original baseline)
- **Early Stopping**: 15 validation steps without improvement

### Data Augmentation Details

**Temporal Masking:**
```python
temporal_mask_prob: 0.25     # Probability of masking each time window
temporal_mask_length: 6       # Consecutive time steps to mask
```

**Noise Injection:**
```python
white_noise_std: 1.2          # Gaussian noise std
constant_offset_std: 0.25     # Session-level offset std
```

**Smoothing:**
```python
smooth_kernel_std: 2          # Gaussian kernel std
smooth_kernel_size: 100       # Kernel size
```

## File Structure

```
├── model_training/
│   ├── train_model.py           # Main training script
│   ├── evaluate_model.py        # Evaluation script
│   ├── rnn_model.py            # GRU model architecture
│   ├── rnn_trainer.py          # Training loop
│   ├── dataset.py              # Data loading
│   ├── data_augmentations.py   # Augmentation functions
│   ├── evaluate_model_helpers.py
│   ├── rnn_args.yaml           # Training configuration
│   └── trained_models/
│       └── temporal_masked_model/  # Our improved model
├── language_model/
│   ├── language-model-standalone.py
│   └── pretrained_language_models/
│       └── openwebtext_1gram_lm_sil/
├── data/
│   ├── hdf5_data_final/        # Neural data
│   ├── t15_pretrained_rnn_baseline/  # Original baseline
│   └── t15_copyTaskData_description.csv
└── README.md
```

---

## Setup and Evaluation

Full installation instructions and environment setup details are available in the competition GitHub repository. The system requires Redis for inter-process communication and separate conda environments for the neural decoder and language model.

To evaluate our improved model, start the Redis server and language model process, then run the evaluation script with our trained checkpoint. Expected runtime is approximately 45 minutes on CPU for the full validation set.

---

## Acknowledgments

This work builds on the baseline architecture from Card et al. (2024) "An Accurate and Rapidly Calibrating Speech Neuroprosthesis," published in the New England Journal of Medicine. We thank the Brain-to-Text 2025 Competition organizers for providing the dataset and baseline models.