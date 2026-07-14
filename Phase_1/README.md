# AI/ML Engineering Internship Tasks

This repository contains the projects and tasks completed during my **AI/ML Engineering Internship** at **Developers Hub Corporation**. The assignments cover a wide spectrum of machine learning and artificial intelligence workflows, ranging from Exploratory Data Analysis (EDA) and classical statistical forecasting to advanced Large Language Model (LLM) prompt engineering and custom conversational AI fine-tuning.

---

## 📌 Repository Overview

The repository is structured around 5 key implementation tasks designed to build and demonstrate practical skills in data science, predictive modeling, and engineering robust AI agents:

1. **Task 1: Iris Dataset Exploratory Data Analysis (EDA)**
2. **Task 2: Short-Term Stock Price Prediction (Regression & ARIMA)**
3. **Task 3: Heart Disease Diagnostic Prediction (Classification & Interpretability)**
4. **Task 4: General Health Query Chatbot (LangChain & OpenRouter)**
5. **Task 5: Empathetic Mental Health Support Chatbot (Fine-Tuning Causal LLM)**

---

## 🛠️ Detailed Task Breakdowns

### 📊 Task 1: Exploring and Visualizing a Simple Dataset

* **Objective:** Clean, inspect, and visualize the classic Iris dataset to understand multivariate feature distributions and class separability.
* **Tech Stack:** `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Key Implementations:**
* Statistical summary profiling (`.info()`, `.describe()`).
* Visual exploration using multi-dimensional scatter plots, feature distributions via histograms, and outlier identification via box plots.


* **Insights:** Identified that **petal length** and **petal width** act as linearly separable dimensions for the *Setosa* species, whereas *Versicolor* and *Virginica* share overlapping boundaries requiring higher-order classification models.

### 📈 Task 2: Short-Term Stock Price Prediction

* **Objective:** Forecast the next day's closing stock price using both features-driven machine learning models and pure autoregressive time series analysis.
* **Tech Stack:** `yfinance`, `scikit-learn`, `statsmodels`, `matplotlib`
* **Methodologies Applied:**
* **Regression Models:** Linear Regression and Random Forest Regressors utilizing same-day market technical indicators (`Open`, `High`, `Low`, `Volume`) to output the subsequent day's `Close` price.
* **Time Series Models:** A classical **ARIMA** (Autoregressive Integrated Moving Average) model forecasting exclusively based on historical close value sequences.


* **Key Outcome:** Regression variants showed tight alignment with daily variations by tracking intra-day variables, while ARIMA provided smoother, long-trend lines that act independently of real-time market data structures.

### 💔 Task 3: Heart Disease Diagnostic Prediction

* **Objective:** Predict binary clinical risk categories for cardiac illness based on multi-site diagnostic patient records while emphasizing feature importance and minimizing false negatives.
* **Tech Stack:** `scikit-learn`, `pandas`, `seaborn`
* **Methodologies Applied:** Trained and compared **Logistic Regression** and **Decision Tree Classifiers** evaluated over accuracy matrices, ROC-AUC, and clinical confusion charts.
* **Key Metrics:** Logistic Regression achieved a superior Area Under the Curve (**AUC = 0.921** vs 0.876) and significantly minimized high-risk False Negatives down to 11 compared to 16 for the baseline tree structure.
* **Interpretability:** Discovered through weight coefficients that physical angina definitions (`cp_atypical angina`) function as strong negative indicators, whereas regional demographics and calcium vessel metrics heavily bias positive diagnostic probability.

### 🤖 Task 4: General Health Query Chatbot

* **Objective:** Design an interactive health information agent capable of responding to everyday well-being queries while maintaining absolute safe clinical guardrails.
* **Tech Stack:** `LangChain`, `OpenRouter API`, `Python`
* **Architectural Safety Net:**
* Implemented an upstream, **deterministic Python rule-layer** to detect critical crisis/emergency terms before querying the model, guaranteeing consistent immediate redirect messages.
* Prompt-engineered system boundaries forbidding diagnostic pronouncements or precise pharmaceutical dosing.
* Configured explicit conversation-level memory buffers to retain semantic state across multiple interaction steps seamlessly.



### 🧠 Task 5: Empathetic Mental Health Support Chatbot

* **Objective:** Fine-tune a causal language model to respond to sensitive emotional disclosures using supportive, conversational patterns and serve it through an intuitive web UI.
* **Tech Stack:** `Hugging Face Transformers`, `PyTorch`, `Streamlit`, `Datasets`
* **Core Technical Workflow:**
* **Dataset Optimization:** Processed the `Ahren09/empathetic_dialogues` corpus (~25k samples) by accurately mapping consecutive conversation IDs (speaker $\rightarrow$ listener turns) instead of single utterances, preventing default model echo errors.
* **Training Protocol:** Fine-tuned `distilgpt2` using masked label padding via `DataCollatorForLanguageModeling` to track validation loss/perplexity trends across dedicated validation checkpoints.
* **Deployment Layout:** Built a custom **Streamlit** interface complete with visual safety disclaimers, real-time message state-handling, and built-in text search blocks for immediate emergency redirection.



---

## 🚀 Setup and Installation

Follow these instructions to clone and run the notebooks or application components locally:

1. **Clone the Repository:**
```bash
git clone https://github.com/umar-f001/dhc_internship.git

```


2. **Set Up a Virtual Environment:**
```bash
   python -m venv
   source venv\Scripts\activate

```

3. **Install Dependencies:**
```bash
pip install -r requirements.txt

```

   *(Ensure you install standard packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `yfinance`, `statsmodels`, `transformers`, `torch`, `streamlit`, `langchain`)*


## 📜 Disclaimer

*These projects are developed strictly as educational tasks under the Developers Hub Corporation AI/ML internship curriculum. None of the medical or financial components should be used as real sources of medical diagnostics or financial trading advice.*

---
