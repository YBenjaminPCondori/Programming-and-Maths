# INM702 Task 1 - FakeMusicCaps generator attribution

This folder contains a self-contained notebook suite for classifying five synthetic music generators: AudioLDM2, MusicGen, MusicLDM, Mustango, and Stable Audio Open.

The notebooks use the supplied FakeMusicCaps archive as data, the module labs and official lab solutions as coding-level references, and Model Variants only as a structural reference. The original archives are not modified. Lecture slides are out of scope. Task 2 is out of scope.

## Google Colab layout

The default configuration expects:

```text
/content/drive/MyDrive/ProgMathsAI/
├── Fake Music Caps.zip
├── Fake Music Caps/                 # created automatically on first extraction
├── Task 1 FakeMusicCaps/            # upload this notebook folder here
└── task1_outputs/                   # created by the notebooks
```

If your Drive layout differs, edit only `PROJECT_ROOT`, `ZIP_PATH`, or `DATA_DIR` in the first configuration cell.

## Run modes

- `smoke`: at most eight samples per generator in each split and short neural training. Use this first.
- `final`: every split is balanced to its smallest class; across the supplied archive this uses up to 1,257 examples per generator. NumPy CNN/LSTM runs can take a long time on Colab CPU.

Do not report smoke-mode scores as coursework results.

## Recommended order

1. `00_Data_Audit_and_NumPy_MFCC.ipynb`
2. `01_PCA_MFCC_Analysis.ipynb`
3. `02_Logistic_Regression.ipynb`
4. `03_Support_Vector_Machine.ipynb`
5. `04_Random_Forest.ipynb`
6. `05_XGBoost.ipynb`
7. `06_NumPy_MLP.ipynb`
8. `07_NumPy_CNN.ipynb`
9. `08_NumPy_LSTM.ipynb`
10. `09_Model_Comparison.ipynb`

Model notebooks do not depend on the data-audit or PCA notebook. Each can scan the dataset, recreate the deterministic split, extract its own NumPy MFCCs, train, evaluate, and save outputs without a custom helper module.

## Reproducibility and leakage controls

- Fixed seed: 42.
- Split: 70% train, 15% validation, 15% test, assigned by a stable hash of the shared 11-character track ID.
- All generator versions of a track ID stay in one split.
- Each split is balanced independently.
- Scaling and PCA are fitted on training data only.
- Hyperparameters and early stopping use validation data only.
- Test metrics are computed once after model selection.
- Cache filenames include the run mode, representation, and MFCC-parameter fingerprint.

## Feature pipeline

Audio is mono/resampled to 16 kHz, centre cropped or padded to 2.5 seconds, and converted to MFCCs with NumPy mathematics: 25 ms frames, 10 ms hop, 512-point FFT, 26 mel filters, and 20 coefficients. Statistical models and the MLP use 80 mean/std MFCC and delta statistics. The CNN uses coefficient-by-time maps; the LSTM uses time-by-coefficient sequences.

## Libraries

Scikit-learn is used only for PCA, statistical baselines, and evaluation metrics. XGBoost is used only in its named baseline. The MLP, CNN, and LSTM do not use TensorFlow, PyTorch, Keras, or scikit-learn neural networks.

## Outputs

Each model writes under `task1_outputs/<model>/`:

- `cache/`: model-local feature arrays
- `figures/`: confusion matrices and learning curves
- `metrics/`: stable JSON metric files
- `models/`: fitted estimators or NumPy weights
- `tables/`: validation experiments, predictions, and confusion matrices

The comparison notebook reads the stable `metrics/test_metrics_<run_mode>.json` files.

## Coursework reporting

Use the NumPy MLP as the canonical response to the fully connected-network requirements. CNN and LSTM are supplementary extensions. Report PCA as feature analysis, not as a classifier. Cite any external code or material used, and write conclusions only from executed final-mode outputs.
