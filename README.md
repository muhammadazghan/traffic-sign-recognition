# Traffic Sign Recognition using CNN (GTSRB)

**Deep Learning Semester Project**

**Name:** Muhammad Azghan

A convolutional neural network that classifies German traffic signs into 43 classes.
It reaches **99.87% test accuracy** and runs fully in the browser on the project website.

## Project links

| | |
|---|---|
| Live website (Vercel) | _add Vercel URL_ |
| Project report | _add link_ |
| Presentation + YouTube video | _add link_ |
| LinkedIn post | _add link_ |

## Dataset

- **GTSRB** (German Traffic Sign Recognition Benchmark), secondary dataset from Kaggle
- 39,209 training images, 43 classes
- Kaggle link: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
- Details in [`dataset_link.md`](dataset_link.md)

## Model

- Input: 32x32 RGB image, pixels scaled to 0-1
- 3 blocks of `Conv -> BatchNorm -> Conv -> BatchNorm -> MaxPool -> Dropout(0.25)` with 32, 64, 128 filters
- `Flatten -> Dense(256, relu) -> Dropout(0.5) -> Dense(43, softmax)`
- Adam optimizer, sparse categorical cross-entropy, up to 25 epochs with early stopping
- Augmentation: small rotation, zoom and shift (no horizontal flip, since flipping changes a sign's meaning)
- Split: 80% train, 10% validation, 10% test (stratified)

## Results

- **Test accuracy: 99.87%** on 3,921 held-out images
- Plots and the full classification report are in [`results/`](results/)

## Repository structure

```
README.md
requirements.txt
train.ipynb              training notebook (run in Google Colab, T4 GPU)
predict.py               predict a sign from an image file
dataset_link.md          dataset description and Kaggle link
model/
  traffic_sign_model.keras   trained model (use this one)
  traffic_sign_model.h5      same model, HDF5 format
  traffic_sign.weights.h5    weights only
results/                 accuracy curves, confusion matrix, classification report
report/                  project report
website/                 static site deployed on Vercel (runs the model in the browser)
  model/                 weights.bin + model_config.json + class_names.json for the browser
```

## Run a prediction locally

```bash
pip install -r requirements.txt
python predict.py path/to/sign.jpg
```

## Retrain the model

1. Open `train.ipynb` in Google Colab and select a T4 GPU.
2. Create a Kaggle API token and paste it in the first code cell.
3. Run all cells. Weights and plots are saved in `outputs/`.

## Website

The site in `website/` is plain HTML, CSS and JavaScript. `cnn.js` loads the exported
weights and runs the network in the browser, so no server or Python is needed.
On Vercel, set the **Root Directory** to `website`.
