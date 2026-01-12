# 🧠 Universal Neural Network (NumPy Only)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-blue?style=for-the-badge&logo=numpy)
![Manager](https://img.shields.io/badge/uv-Fastest_Installer-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## 📖 Project Brief
This project is a **professional-grade Deep Learning framework built entirely from scratch**, bypassing high-level libraries like PyTorch or TensorFlow to implement the core mathematics of neural networks using only **NumPy**. 

The goal was to deconstruct the "black box" of AI. By mathematically deriving and implementing every component—from **He Initialization** to **Adam Optimization** and **Inverted Dropout**—this framework serves as both a high-performance educational tool and a proof of concept for modular software engineering in Python. It is designed to be **universal**, seamlessly handling both tabular data (Breast Cancer diagnosis) and image data (MNIST digit recognition) through a unified pipeline.

---

## ⚙️ What is Working on It?
The "engine" behind this framework is fully operational and consists of four synchronized systems:

1.  **The Dynamic Computation Graph:** * A flexible **MLP (Multi-Layer Perceptron)** that supports variable depth, width, and activation functions.
    * **Forward Propagation** is matrix-vectorized for speed.
    * **Backward Propagation** manually calculates gradients for every layer, weight, and bias using the Chain Rule.

2.  **Advanced Optimization Suite:**
    * **Adam (Adaptive Moment Estimation):** Fully implemented with bias correction for stable convergence.
    * **Momentum & RMSProp:** Available for comparative analysis.
    * **Learning Rate Scheduling:** Dynamic adjustments via Cosine Annealing, Step Decay, or Exponential Decay.

3.  **Data Engineering Pipeline:**
    * **Universal Loader:** Automatically downloads, caches, and parses datasets (CSV for tabular, Gzip/IDX for MNIST).
    * **Smart Preprocessor:** Handles Normalization, One-Hot Encoding, and Flattening automatically based on data type.
    * **Memory-Efficient Generators:** Uses Python generators to serve data in batches, ensuring scalability.

4.  **Interactive Interface:**
    * A **Gradio-based GUI** that allows users to design architectures visually, train in real-time, and perform inference (e.g., drawing digits on a canvas).

---

## 🚀 Future Plan (Roadmap)
We are treating this as an evolving open-source framework. The next phases of development focus on expanding architecture support and deployment capabilities:

* **Phase 1: Convolutional Layers (CNNs):** Implement `Conv2D` and `MaxPooling` layers to move beyond flattened MLPs for state-of-the-art image recognition.
* **Phase 2: Recurrent Architectures (RNNs/LSTMs):** Add support for sequential data to handle time-series forecasting.
* **Phase 3: Model Export:** Implement an ONNX exporter to allow models trained here to run in production environments.
* **Phase 4: Cloud Deployment:** Dockerize the inference API for easy deployment on AWS/GCP.

---

## 📂 Project Structure

```text
NN_Project/
├── app_gradio.py             # Interactive Web Interface (Training & Eval)
├── run_pipeline.py           # CLI Entry Point for Headless Training
├── pyproject.toml            # Modern Dependency Configuration
├── config/
│   └── config.yml            # Central Configuration (Hyperparameters)
├── data_pipeline/
│   ├── data_loader.py        # Universal Data Downloader & Parser
│   ├── preprocessor.py       # Normalization & One-Hot Encoding
│   └── data_generator.py     # Memory-Efficient Batch Generator
├── model/
│   ├── mlp.py                # Multi-Layer Perceptron Orchestrator
│   ├── layer.py              # Dense Layer Logic (Forward/Backward)
│   ├── optimizers.py         # Adam, RMSProp, SGD, Momentum
│   ├── activations.py        # ReLU, Sigmoid, Tanh, Softmax (Math)
│   ├── loss.py               # BCE and MSE Loss Functions
│   └── scheduler.py          # Learning Rate Schedulers
└── pipeline/
    └── pipeline.py           # The "Brain" connecting Model, Data, and Optimizer


🛠️ Installation
You can choose between the standard pip method or the ultra-fast uv manager.
  Option A: Modern & Fast (Recommended with uv)
  We use uv for lightning-fast dependency resolution and virtual environment management.
                    
                   1- Install uv (if you haven't already):
                        # On macOS/Linux
                        curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
                        # On Windows
                        powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
                        
                   2- Initialize & Sync: This will automatically create the virtual environment and install dependencies from pyproject.toml.
                             - git clone [https://github.com/YOUR_USERNAME/Universal_Neural_Network.git](https://github.com/YOUR_USERNAME/Universal_Neural_Network.git)
                              - cd Universal_Neural_Network
                              -uv sync
                         
                  
                          Activate Environment:
                          Bash
                          source .venv/bin/activate
                          # On Windows: .venv\Scripts\activate

Option B: Standard (Legacy Pip)
        Clone the repository:
            Bash
            git clone [https://github.com/YOUR_USERNAME/Universal_Neural_Network.git](https://github.com/YOUR_USERNAME/Universal_Neural_Network.git)
            cd Universal_Neural_Network

        Install Dependencies:
          Bash
          pip install numpy pyyaml tqdm matplotlib seaborn gradio opencv-python-headless

🎮 Usage
  1. Interactive Web App (Gradio)
  Launch the dashboard to train models visually and draw digits for real-time prediction.
        Bash
        python3 app_gradio.py
        Open your browser at http://127.0.0.1:7860
  
  2. Command Line Interface (CLI)
  Train a model directly from your terminal. The system automatically downloads datasets.

          Train on MNIST (Digit Recognition):
                  Bash
                  python3 run_pipeline.py --dataset mnist

          Train on Breast Cancer Data (Medical Diagnosis):
                  Bash
                  python3 run_pipeline.py --dataset breast_cancer

🎛️ Full Control via config.yml
You do not need to touch the Python code to change the network behavior. Every aspect of the training lifecycle is controlled via NN_project/config/config.yml.

1. Architecture Control
Define the shape and depth of your network.

YAML
model:
  # Input -> Hidden 1 -> Hidden 2 -> Output
  architecture: [128, 64, 10]       
  # Activation function for each layer
  activations: ["relu", "relu", "sigmoid"]
  # Dropout probability (0.0 = Off, 0.5 = Drop 50% of neurons)
  dropout_rates: [0.2, 0.2]

       
2. Optimizer Tuning
Switch mathematical optimizers instantly.

YAML
optimizer:
  method: "adam"         # Options: "adam", "sgd", "rmsprop", "momentum"
  learning_rate: 0.001
  beta1: 0.9             # Momentum factor
  beta2: 0.999           # RMS factor


3. Training Strategy
Control how the model learns over time.

YAML
scheduler:
  method: "cosine_annealing"  # Options: "constant", "step_decay", "exponential"
  step_size: 10
  gamma: 0.5

4. Data Management
Manage dataset sources and splitting strategies.

YAML
data_config:
  test_split: 0.2        # 20% for Final Testing
  val_split: 0.1         # 10% for Validation during training
📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

Built with ❤️ by  abdelhadi osama
