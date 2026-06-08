# Classifying Martian Surface Features with a CNN

A convolutional neural network that classifies HiRISE (Mars Reconnaissance
Orbiter) grayscale images into **eight surface-feature categories**: other,
crater, dark dune, slope streak, bright dune, impact ejecta, swiss cheese, and
spider. Built with TensorFlow.

## Overview

The High Resolution Imaging Science Experiment (HiRISE) captures detailed images
of the Martian surface. This project trains a CNN on 10,815 labeled, grayscale
images (resized to 227×227) to recognise distinct geological formations,
supporting automated mapping for planetary-science investigations.

The dataset is heavily imbalanced — the "other" class alone accounts for ~81% of
samples, while the rarest class (impact ejecta) is under 1% — which is the
central challenge the project explores.

## Results

| Metric | Value |
|---|---|
| Test accuracy | ~85% |
| Majority-class baseline | ~81% |

Per-class behaviour is uneven, reflecting the class imbalance:

| Class | Approx. accuracy |
|---|---|
| other | 0.95 |
| crater | 0.60 |
| bright dune | 0.50 |
| swiss cheese | 0.40 |
| dark dune | 0.20 |
| slope streak | 0.05 |
| impact ejecta | 0.00 |
| spider | 0.00 |

The model beats the majority-class baseline and learns strong features for the
dominant classes, but fails on the rarest ones — a direct consequence of the
imbalance and the absence of augmentation.

## Architecture

Input (227 × 227 × 1) →
Conv2D(32, 3×3) ReLU → MaxPool(2×2) →
Conv2D(64, 3×3) ReLU → MaxPool(2×2) →
Conv2D(128, 3×3) ReLU → MaxPool(2×2) →
Flatten → Dense(128) ReLU → Dropout(0.5) →
Dense(8) Softmax

Optimiser: Adam (lr = 1e-3). Loss: sparse categorical cross-entropy.
Up to 50 epochs, batch size 32, early stopping on validation loss (patience 5).

## Data

Mars orbital image (HiRISE) labeled dataset, version 3
(Doran et al., 2020), from the NASA Open Data Portal. The dataset ships with
pre-split train/validation/test sets (~65% / 19% / 16%).

> The image data is not committed to this repo (see `.gitignore`). Download it
> from the NASA Open Data Portal and point the notebook at it.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Open `CNN_code.ipynb` and run the cells top to bottom.

## Repository contents

```
.
├── CNN_code.ipynb                  # full pipeline: data, model, training, evaluation
├── Classifying_Martian_Surface.pdf # project report
├── requirements.txt
├── .gitignore
└── README.md
```

## Known limitations & next steps

The main weakness is minority-class performance. Documented avenues for
improvement: targeted data augmentation (rotations, flips, zooms, brightness),
class weights in the loss, focal loss, resampling (over/undersampling),
ensemble methods, and transfer learning from pre-trained backbones
(e.g. ResNet/EfficientNet adapted to grayscale).

## References

Doran et al. (2020), NASA Open Data Portal; Palafox et al. (2017), *Icarus*
301:120–127; Wagstaff et al. (2018), *AAAI* 32(1).
