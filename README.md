Repository for the Scientific Initiation project that explores the construction of image similarity graphs using rank correlation functions, with subsequent classification via Graph Convolutional Networks (GCN).

## 📝 Project Summary

This work investigates an alternative to the traditional construction of neighbor graphs (like k-NN) for semi-supervised image classification tasks. The central hypothesis is that the similarity between two images can be more robustly defined by the correlation between their respective neighbor lists, rather than solely by the direct distance between them.

To achieve this, we implemented rank correlation methods, such as **RBO (Rank-Biased Overlap)** and **Jaccard**, to calculate the weight of the graph edges. Once constructed, the graph is used to train an **SGC (Simple Graph Convolution)** model, a GCN variant, to classify the images in the dataset.

## 📂 Project Structure

The code has been modularized to ensure clarity and maintainability, following the structure below:

```
/
├── data/                    # Folder for storing datasets 
├── main.py                  # Main script to run experiments
├── config.py                # Control panel with hyperparameters
├── data_loader.py           # Module for loading data
├── graph_builder.py         # Module with graph construction functions (BallTree, RBO, etc.)
├── gcn_model.py             # Module with the GCN model definition and class
├── utils.py                 # Helper functions (K-Fold, etc.)
└── README.md                # This file
```

## ⚙️ Installation and Setup

Follow the steps below to set up the environment and run the project.

**1. Clone the repository:**
```bash
git clone https://github.com/Bagiel1/SIBGRAPI-2025.git
```

**2. Create and activate a virtual environment (recommended):**
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install the dependencies:**
```bash
pip install -r requirements.txt
```

## 🚀 How to Run an Experiment

All experiments are controlled and executed from the `main.py` script. To configure a run, open `main.py` and edit the "Control Panel" at the beginning of the `main()` function.

**Example of the Control Panel in `main.py`:**
```python
def main():
    # --- EXPERIMENT CONTROL PANEL ---
    DATASET_TO_USE = 'flowers'
    FEATURES_EXTRACTOR = 'resnet'
    CORRELATION_FUNCTION_TO_USE = 'rbo'
    USE_AUTOMATIC_THRESHOLD = True
    MANUAL_THRESHOLD = 0.4
    # ------------------------------------
    
    # ... rest of the code ...
```

After setting the desired parameters, run the script from the terminal (in the project's root folder):
```bash
python main.py
```

## 📊 Results

Experiments were conducted on the **Flowers-102** dataset, using features extracted by different CNN architectures. The table below summarizes the main accuracy results (mean ± standard deviation over 10-fold cross-validation).

| Feature Extractor | Correlation Function | Threshold | Final Accuracy          |
|-------------------|----------------------|-----------|-------------------------|
| ResNet            | JaccardK             | Auto      | **85.41% ± 1.30%** |
| ResNet            | RBO                  | Auto      | **84.33% ± 2.37%** |
|                   |                      |           |                    |

## 🎓 Author and Acknowledgements

* **Author:** G. B. Maia
* **Advisor:** L. P. Valem
