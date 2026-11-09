
# Explainable Transformer-Based Malicious URL Detection Using DistilBERT

### Final Year Research Project | BSc (Hons) Computer Science with Artificial Intelligence

**Research Area:** Cybersecurity | Natural Language Processing | Deep Learning | Explainable Artificial Intelligence

---

## 1. Project Overview

This research investigates the effectiveness of **DistilBERT**, a lightweight transformer-based language model, for detecting malicious URLs and explaining its predictions using Explainable Artificial Intelligence (XAI).

The research evaluates both detection performance and explanation reliability through a comparative experimental approach.

Three explainability techniques are investigated:

- Integrated Gradients (IG)
- Local Interpretable Model-Agnostic Explanations (LIME)
- SHapley Additive exPlanations (SHAP)

Traditional machine-learning models, including Random Forest and XGBoost, are also implemented as baseline approaches.

A FastAPI-based web application demonstrates malicious URL prediction and explanation generation.

## 2. Research Problem

Malicious URLs are widely used in phishing attacks, malware distribution and other cybersecurity threats.

Traditional blacklist-based detection approaches may struggle to identify newly created malicious URLs.

Although transformer-based models can achieve high detection performance, their predictions are often difficult to interpret.

This research addresses the need to evaluate both **malicious URL detection accuracy and the reliability of model explanations**.

## 3. Research Objectives

1. Develop and fine-tune DistilBERT for malicious URL detection.
2. Implement Random Forest and XGBoost as baseline models.
3. Evaluate detection performance using standard classification metrics.
4. Integrate Integrated Gradients, LIME and SHAP for model explainability.
5. Compare explanation reliability, stability, faithfulness and computational efficiency.
6. Investigate model sensitivity to URL-format modifications.
7. Develop a web-based prototype for URL classification and explanation visualization.

## 4. Dataset

**Dataset:** ISCX-URL-2016

The dataset contains URL samples from four categories:

| Category        |           Samples |
| --------------- | ----------------: |
| Benign          |           428,103 |
| Defacement      |            96,457 |
| Phishing        |            94,111 |
| Malware         |            32,520 |
| **Total** | **651,191** |

The research includes URL preprocessing, duplicate removal, label encoding and training/test partitioning.

The primary DistilBERT evaluation uses a binary classification setting.

| Label | Class     |
| ----- | --------- |
| 0     | Benign    |
| 1     | Malicious |

**Final experiment:**

- Training samples: 448,779
- Held-out test samples: 96,167
- XAI evaluation subset: 200 URLs

The XAI evaluation subset contains 100 benign and 100 malicious samples.

## 5. Research Methodology

The research follows these main stages:

1. Dataset collection and preprocessing
2. Exploratory data analysis
3. Baseline model implementation
4. DistilBERT fine-tuning
5. Classification performance evaluation
6. XAI method integration
7. Explanation reliability evaluation
8. Robustness analysis
9. Web application development

### Models Implemented

| Model         | Purpose                                     |
| ------------- | ------------------------------------------- |
| Random Forest | Traditional machine-learning baseline       |
| XGBoost       | Gradient-boosting baseline                  |
| DistilBERT    | Main transformer-based classification model |

### Explainability Techniques

**Integrated Gradients:** Calculates feature attribution by integrating gradients between a baseline input and the actual input.

**LIME:** Generates local explanations by perturbing input features and fitting a simpler interpretable model.

**SHAP:** Estimates feature contributions using Shapley-value-based attribution.

## 6. Experimental Results

### DistilBERT Classification Performance

The fine-tuned DistilBERT model was evaluated on 96,167 held-out URLs.

| Evaluation Metric |           Result |
| ----------------- | ---------------: |
| Accuracy          | **98.98%** |
| Precision         |           98.68% |
| Recall            |           98.24% |
| F1-score          |           98.46% |
| ROC-AUC           |           99.92% |
| Test Samples      |           96,167 |

### Confusion Matrix

| Actual / Predicted | Benign | Malicious |
| ------------------ | -----: | --------: |
| Benign             | 63,791 |       420 |
| Malicious          |    562 |    31,394 |

These results demonstrate strong classification performance on the held-out evaluation dataset.

However, performance on different datasets, newly emerging attacks and real-world deployment environments requires further validation.

## 7. Explainable AI Evaluation

Three explanation techniques were evaluated using a balanced subset of 200 held-out URLs.

### Explanation Evaluation Results

| Metric                      | Integrated Gradients |   LIME |   SHAP |
| --------------------------- | -------------------: | -----: | -----: |
| Occlusion Ranking Agreement |                 0.95 |   1.00 |   1.00 |
| Mean Cosine Similarity      |               0.9712 | 0.9920 | 0.9930 |
| Top Component Stability     |               0.9744 | 1.0000 | 1.0000 |

These results summarize the observed explanation behavior under the implemented evaluation procedure.

The research also considers:

- Deletion probability drop
- Prediction flip rate
- Explanation generation time
- Evaluation overhead
- URL component-level attribution

The results should be interpreted within the limitations of the selected evaluation subset, perturbation strategy and metric definitions.

## 8. Robustness Evaluation

The research investigates how URL-format changes influence model predictions and explanation behavior.

Example modifications include:

- Brand-token insertion
- Digit modifications
- URL path reordering
- Benign-looking suffix insertion

Robustness is assessed using prediction changes and confidence differences under the selected perturbations.

## 9. Web Application (MVP)

A web-based prototype was developed using **FastAPI**.

### Main Features

- URL input interface
- Benign or malicious URL prediction
- Model confidence display
- Explainable AI integration
- Prediction explanation visualization

### Application Workflow

```text
User Enters URL
       |
       v
FastAPI Backend
       |
       v
URL Tokenization
       |
       v
Fine-Tuned DistilBERT
       |
       v
Classification Prediction
       |
       v
XAI Explanation Generation
       |
       v
Web Interface Results
```

The prototype is intended for research demonstration and should not be treated as a production-grade URL security scanner.

## 10. Project Structure

```text
PhishingDetection/
|
|-- app/
|   |-- main.py
|   |-- explain.py
|   |-- static/
|   |-- templates/
|
|-- src/
|   |-- Research implementation scripts
|
|-- notebooks/
|   |-- 01_Data_Preparation.ipynb
|   |-- 02_Random_Forest.ipynb
|   |-- 03_XGBoost.ipynb
|   |-- 04_DistilBERT.ipynb
|   |-- 05_Model_Comparison.ipynb
|   |-- 07_SHAP_Explainability.ipynb
|   |-- 08_DistilBERT_Full_Dataset_Finetuning.ipynb
|   |-- 09_XAI_200_Full_DistilBERT_Evaluation.ipynb
|   |-- 10_Results_Visualization_Fixed.ipynb
|
|-- results/
|-- dataset/          # Excluded from Git
|-- models/           # Excluded from Git
|
|-- pyproject.toml
|-- uv.lock
|-- README.md
|-- .gitignore
```

## 11. Installation and Setup

### Prerequisites

- Python version compatible with `pyproject.toml`
- Git
- uv package manager
- PyTorch
- Transformers
- FastAPI

GPU acceleration is recommended for transformer fine-tuning and some explanation computations.

### Clone Repository

```bash
git clone https://github.com/MohamedNajahi/Explainable-Malicious-URL-Detection-DistilBERT.git
cd Explainable-Malicious-URL-Detection-DistilBERT
```

### Install Dependencies

Install uv if it is not already available.

```bash
pip install uv
```

Synchronize the project dependencies:

```bash
uv sync --frozen
```

### Model Files

Trained model artifacts and research datasets are not included in this GitHub repository because of their size.

To reproduce inference, place the required trained model and tokenizer files in the directories expected by the application configuration.

### Run the Web Application

From the project root:

```bash
uv run uvicorn app.main:app --reload
```

The default development server address is:

```text
http://127.0.0.1:8000
```

The application requires the appropriate trained model files and dependencies to be available.

## 12. Technologies Used

| Area                 | Technology                |
| -------------------- | ------------------------- |
| Programming Language | Python                    |
| Transformer Model    | DistilBERT                |
| Deep Learning        | PyTorch                   |
| NLP Framework        | Hugging Face Transformers |
| Machine Learning     | Scikit-learn, XGBoost     |
| Explainable AI       | Captum, LIME, SHAP        |
| Backend              | FastAPI                   |
| Experiments          | Jupyter Notebook          |
| Version Control      | Git and GitHub            |

## 13. Research Contribution

The primary contribution of this research is an experimental investigation of transformer-based malicious URL detection with a particular focus on explainability.

The work combines:

- Fine-tuned DistilBERT classification
- Traditional machine-learning baseline comparisons
- Comparative evaluation of three XAI methods
- Explanation reliability analysis
- URL-format robustness investigation
- A functional research demonstration application

The research does not claim to introduce DistilBERT or the explainability methods themselves. Instead, it evaluates their effectiveness and behavior in the selected malicious URL detection setting.

## 14. Limitations and Future Work

Potential future research directions include:

- Cross-dataset evaluation
- Testing against newly emerging phishing URLs
- Adversarial robustness improvements
- More extensive XAI evaluation
- Model optimization for faster inference
- Real-time threat intelligence integration
- Production-level application security and deployment

## 15. Disclaimer

This project was developed for academic research and educational purposes.

The model may produce incorrect predictions, including false positives and false negatives. Results should not be used as the sole basis for cybersecurity decisions.

## 16. Author

**Mohamed Najahi**
BSc (Hons) Computer Science with Artificial Intelligence
National Institute of Business Management (NIBM)
Kandy Innovation Centre, Sri Lanka

**GitHub:** https://github.com/MohamedNajahi

---

*Final Year Research Project — 2026*
