# INM702 Task 1 NumPy MLP Trials

This folder contains a split-notebook workflow derived from `../MLP_v1.ipynb`.
The original notebook is unchanged.

## Run Order

1. `00_MLP_Baseline.ipynb`
2. `01_MLP_Activation_Trial.ipynb`
3. `02_MLP_Optimizer_Trial.ipynb`
4. `03_MLP_Dropout_Regularisation_Trial.ipynb`
5. `04_MLP_Architecture_Trial.ipynb`
6. `05_MLP_LearningRate_BatchSize_Trial.ipynb`
7. `06_MLP_Trial_Comparison.ipynb`
8. `07_MLP_Final_Selected_Model.ipynb`

The first six notebooks are validation-only hyperparameter trials. They prepare
the train and validation features, train NumPy MLP variants, and save validation
tables, histories, summaries, and training-curve plots. They do not evaluate the
held-out test set.

`06_MLP_Trial_Comparison.ipynb` reads the validation tables, ranks candidates by
validation macro F1, then balanced accuracy, then accuracy, and writes
`selected_hyperparameters_<RUN_MODE>.json`.

`07_MLP_Final_Selected_Model.ipynb` is the only notebook in this workflow that
prepares and evaluates the held-out test split. Run it only after notebook `06`
has selected hyperparameters for the same run mode.

## Run Modes

The notebooks default to smoke mode:

```powershell
$env:FAKEMUSICCAPS_RUN_MODE = "smoke"
```

Smoke mode uses fewer samples and fewer epochs so the workflow can be checked
quickly. For coursework-scale runs, use:

```powershell
$env:FAKEMUSICCAPS_RUN_MODE = "final"
```

Keep the same run mode across notebooks `00` to `07`. The comparison notebook
only reads validation files with the matching run-mode suffix.

## Data Paths

The notebooks keep the base notebook's environment-variable overrides. If you
run them locally instead of in Colab, set the project root and, if needed, the
dataset ZIP or extracted dataset folder before execution:

```powershell
$env:PROGMATHSAI_PROJECT_ROOT = "C:\Users\PC\Documents\GitHub\Programming-and-Maths"
$env:FAKEMUSICCAPS_ZIP_PATH = "C:\path\to\Fake Music Caps.zip"
$env:FAKEMUSICCAPS_DATA_DIR = "C:\path\to\Fake Music Caps"
```

You only need `FAKEMUSICCAPS_ZIP_PATH` if the ZIP is not under the default
project dataset path, and you only need `FAKEMUSICCAPS_DATA_DIR` if the dataset
has already been extracted somewhere else.

## Method Boundary

The MLP implementation remains from scratch in NumPy: dense layers, ReLU,
sigmoid, softmax cross-entropy, inverted dropout, SGD, momentum, early stopping,
and gradient checks are preserved from the base notebook. Scikit-learn is used
only for metrics. No scikit-learn, PyTorch, TensorFlow, or Keras model fitting is
introduced for Task 1 MLP.

The existing `CNN_v1.ipynb` and `lstm_V1.ipynb` notebooks remain unchanged as
optional extras outside this Task 1 MLP trial workflow.
