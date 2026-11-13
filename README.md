
# Real vs AI-Generated Face Detection
### A CNN-Based Approach to Synthetic Face Image Classification

<p align="center">
  <b>Hiba Zormati</b> · Statistics & Data Science Engineering · ESSAI Tunis
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Deep_Learning-PyTorch-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Computer_Vision-CNN-16A085" />
  <img src="https://img.shields.io/badge/Explainability-Grad--CAM-8B5CF6" />
  <img src="https://img.shields.io/badge/Platform-Google_Colab-F9AB00?logo=googlecolab&logoColor=white" />
</p>

---

## Overview

The increasing realism of AI-generated human faces raises an important question:

**Can a convolutional neural network distinguish authentic photographs from synthetic face images?**

This project investigates binary face classification using a supervised CNN trained on real FFHQ photographs and StyleGAN-generated faces.

The work goes beyond model training by examining dataset integrity, baseline performance, threshold selection, prediction calibration, classification errors and visual explanations.

The central objective is to build a reproducible experimental pipeline and evaluate how well a CNN distinguishes real and synthetic faces within a controlled dataset distribution.

> **Research question:** To what extent can a CNN distinguish real FFHQ faces from StyleGAN-generated faces on held-out images from the same dataset distribution?

## Dataset

The project uses the [140k Real and Fake Faces dataset](https://huggingface.co/datasets/TheKernel01/140k-Real-and-Fake-Faces).

It contains 140,000 face images:

- 70,000 real photographs from FFHQ.
- 70,000 synthetic faces generated using StyleGAN.

### Official dataset splits

| Split | Published examples |
|---|---:|
| Training | 100,000 |
| Validation | 20,000 |
| Testing | 20,000 |
| **Total** | **140,000** |

The notebook performs additional dataset and index checks. The saved evaluation subsets used in the recorded experiment contain:

| Split | Examples used |
|---|---:|
| Training | 100,000 |
| Validation | 19,376 |
| Testing | 19,306 |

The validation and test subsets are balanced between real and fake classes.

### Labels

| Label | Class |
|---:|---|
| 0 | Real |
| 1 | Fake |

## Dataset Integrity and Duplicate Auditing

Before training, the dataset was examined to identify potential data leakage and duplicate images across the official splits.

Two complementary hashing approaches were used:

**1. Exact duplicate detection**

MD5 hashes were calculated from image bytes to identify identical files.

Result: No repeated MD5 hashes were detected across the 140,000 audited images.

**2. Perceptual duplicate candidates**

Difference hashing (dHash) was used to identify visually similar image candidates.

The audit identified:

- 2,912 rows belonging to repeated dHash groups.
- 645 candidate groups spanning multiple dataset splits.
- Four dHash groups containing conflicting labels.

A subsequent comparison using 64 × 64 grayscale images and mean squared error showed that only 0.31% of the cross-split candidate groups had MSE below 20.

These results suggest that perceptual-hash collisions and visually similar images require careful interpretation; a shared dHash alone does not establish that two images are duplicates.

## Methodology

### 1. Image preprocessing

Images are converted to RGB tensors and resized to 256 × 256 pixels.

Normalization statistics are estimated from a reproducible sample of training images.

Training augmentation includes horizontal flipping, while validation and test images use deterministic transformations.

### 2. Baseline models

Two simple baselines are considered before evaluating the CNN.

**Majority-class baseline**

A classifier predicts the most frequent class in the validation set.

**Pixel-based Logistic Regression**

Images are resized to 16 × 16 RGB pixels and flattened into feature vectors. Standardization and Logistic Regression are then applied.

Validation performance:

| Metric | Pixel Logistic Regression |
|---|---:|
| Accuracy | 0.7403 |
| ROC-AUC | 0.8081 |

This baseline helps assess whether coarse pixel-level information already provides useful class separation.

### 3. Convolutional Neural Network

A custom CNN was implemented using PyTorch.

Its architecture consists of five convolutional blocks, each containing:

- 3 × 3 convolution.
- Batch Normalization.
- ReLU activation.
- 2 × 2 Max Pooling.

The number of channels increases progressively:

```text
Input Image (3 × 256 × 256)
          |
          v
Conv Block 1: 3  → 16
          |
          v
Conv Block 2: 16 → 32
          |
          v
Conv Block 3: 32 → 64
          |
          v
Conv Block 4: 64 → 128
          |
          v
Conv Block 5: 128 → 256
          |
          v
Global Average Pooling
          |
          v
Dropout (0.3)
          |
          v
Linear Classification Head
          |
          v
Single Output Logit
          |
          v
Real / Fake
```

**Model characteristics**

| Property | Configuration |
|---|---|
| Framework | PyTorch |
| Architecture | Custom SmallCNN |
| Convolutional blocks | 5 |
| Input resolution | 256 × 256 |
| Dropout | 0.3 |
| Trainable parameters | 393,857 |
| Loss function | BCEWithLogitsLoss |
| Optimizer | Adam |

### 4. Training strategy

The model is trained using:

- Adam optimizer.
- Learning rate: 0.001.
- Batch size: 32.
- Maximum epochs: 15.
- Early stopping patience: 4.
- Validation-loss-based checkpoint selection.
- Checkpoint saving and training recovery.
- Automatic mixed precision when CUDA is available.

The recorded experiment used an NVIDIA Tesla T4 GPU.

The checkpoint with the lowest validation loss was selected for final testing.

## Experimental Results

The final model was evaluated using the held-out test subset of 19,306 images.

### Default threshold: 0.50

| Metric | Test performance |
|---|---:|
| Accuracy | **97.16%** |
| Precision (Fake) | 95.23% |
| Recall (Fake) | 99.29% |
| F1-score (Fake) | 97.22% |
| ROC-AUC | **99.85%** |
| Average Precision | 99.85% |

### Confusion Matrix

| | Predicted Real | Predicted Fake |
|---|---:|---:|
| Actual Real | 9,173 | 480 |
| Actual Fake | 69 | 9,584 |

The model achieves high recall for synthetic images while maintaining strong precision.

### Bootstrap confidence interval

A bootstrap procedure with 1,000 resamples was used to estimate uncertainty around test accuracy.

**95% percentile confidence interval:**

96.93% – 97.41%

This interval reflects sampling uncertainty under an IID test-sample assumption and does not account for dataset bias or distribution shift.

## Decision Threshold Optimization

The default classification threshold is 0.50.

An additional experiment selected a threshold by maximizing F1-score on the validation set only.

| Measurement | Result |
|---|---:|
| Default threshold | 0.500 |
| Validation-selected threshold | 0.805 |
| Validation F1 at 0.50 | 0.9736 |
| Validation F1 at 0.805 | 0.9822 |
| Test accuracy at 0.805 | **98.07%** |
| Test F1 at 0.805 | 0.9807 |

The threshold is selected using validation predictions and then applied to the test set.

This illustrates how decision thresholds influence the balance between false positives and false negatives.

The 0.805 threshold is an alternative operating point, not a replacement for the default-threshold result.

## Error Analysis

The notebook examines two types of classification errors.

**False positives:** Real faces incorrectly classified as fake.

**False negatives:** Synthetic faces incorrectly classified as real.

At the default threshold:

- False positives: 480.
- False negatives: 69.

Selected misclassified examples are inspected to understand potential visual patterns and model weaknesses.

## Calibration and Explainability

### Probability Calibration

A reliability diagram is used to examine the relationship between predicted probabilities and observed outcomes.

High ROC-AUC does not necessarily mean that predicted probabilities are well calibrated.

The notebook treats the test reliability diagram as descriptive analysis rather than using test data to fit a calibration correction.

### Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) is used to visualize image regions that contribute to the model's predictions.

The objective is to investigate whether the CNN focuses on meaningful facial regions or potentially relies on dataset-specific visual artifacts.

Grad-CAM provides a coarse, model-dependent explanation and should not be interpreted as causal evidence.

## Key Findings

**1. CNNs can learn discriminative visual patterns**

The custom CNN substantially outperformed the pixel-based Logistic Regression baseline in the recorded experiment.

**2. Ranking performance is strong**

The model achieved a test ROC-AUC of 0.9985, indicating strong separation between real and fake images within this dataset.

**3. Threshold selection changes classification behavior**

Validation-based threshold selection improved the reported test accuracy from 97.16% to 98.07%.

**4. Dataset integrity requires more than exact hashing**

Perceptual similarity can reveal candidates missed by exact hashes, but approximate matches require further investigation.

**5. Explainability and calibration matter**

Classification metrics alone do not explain model behavior or establish whether its probabilities are reliable.

## Limitations

The results should be interpreted within the scope of the experimental dataset.

1. Real images originate from FFHQ, while synthetic images originate from StyleGAN, creating potential source confounding.
2. The experiment does not establish generalization to other image generators.
3. Perceptual hashes can produce collisions and cannot conclusively establish duplicate images.
4. The CNN is relatively compact and is not intended to represent the strongest current detection architectures.
5. Hardware, runtime interruptions and saved evaluation indices can affect reproducibility.
6. Grad-CAM explanations are coarse and model-dependent.
7. The bootstrap confidence interval does not account for distribution shift.
8. The model's predicted probabilities may not remain calibrated under different image sources or class prevalence.

**Important:** High performance on this benchmark does not mean that the detector can reliably identify every AI-generated face.

## Future Work

Potential research extensions include:

- External testing on images from newer generative models.
- Evaluation under JPEG compression, resizing, blur and social-media processing.
- Comparing custom CNNs with transfer-learning architectures.
- Exploring frequency-domain features and alternative detection methods.
- Improving probability calibration using a separate calibration set.
- More extensive cross-dataset evaluation.
- Further investigation of source confounding and duplicate candidates.

## Technologies

| Category | Tools |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| Computer Vision | CNN, torchvision |
| Image Processing | Pillow, NumPy |
| Data Management | Hugging Face Datasets |
| Machine Learning | Logistic Regression, Scikit-learn |
| Evaluation | ROC-AUC, F1, Precision, Recall, Bootstrap |
| Explainability | Grad-CAM |
| Visualization | Matplotlib |
| Environment | Google Colab, NVIDIA T4 |

## Running the Project

The complete implementation is provided in the Jupyter notebook.

1. Open the notebook in Google Colab.
2. Install the required Python packages.
3. Load the dataset from Hugging Face.
4. Run the dataset integrity and preprocessing sections.
5. Execute the baseline experiments.
6. Train or resume the CNN using the saved checkpoints.
7. Evaluate the selected model and inspect the results.
8. Run the optional Grad-CAM visualizations.

The notebook includes experiment configuration, saved indices, checkpoint recovery and evaluation procedures.

## Dataset and References

**Dataset**

[TheKernel01 — 140k Real and Fake Faces](https://huggingface.co/datasets/TheKernel01/140k-Real-and-Fake-Faces)

**Research references**

1. Goodfellow et al. (2014). *Generative Adversarial Nets*. NeurIPS.
2. Karras et al. (2019). *A Style-Based Generator Architecture for Generative Adversarial Networks*. CVPR.
3. Wang et al. (2020). *CNN-Generated Images Are Surprisingly Easy to Spot... for Now*. CVPR.
4. Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. ICCV.
5. He et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.

---

## Disclaimer

This project is an academic machine learning experiment. Its predictions should not be treated as definitive proof that an image is authentic or AI-generated.

Performance on this dataset does not guarantee reliability on unseen generators, manipulated images or real-world social media content.

---

<p align="center">
  <b>Developed by Hiba Zormati</b><br/>
  Statistics & Data Science Engineering · ESSAI Tunis<br/>
  <a href="https://www.linkedin.com/in/hiba-zormati-624b13319/">LinkedIn</a> ·
  <a href="https://github.com/hibazormati">GitHub</a>
</p>
