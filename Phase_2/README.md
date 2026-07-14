# ML Internship Projects

A collection of machine learning projects covering NLP, computer vision, multimodal learning, and applied ML engineering. Each task lives in its own notebook and is runnable independently.

---

## Task 1: News Topic Classifier Using BERT

Fine-tunes `bert-base-uncased` on the [AG News](https://huggingface.co/datasets/ag_news) dataset to classify news headlines into 4 topic categories (World, Sports, Business, Sci/Tech).

- **Approach:** Tokenization with the BERT tokenizer, fine-tuning via Hugging Face `Trainer`
- **Evaluation:** Accuracy and macro F1-score on a held-out test set
- **Deployment:** Streamlit app for live headline classification

**Notebook:** `Task1_News_Topic_Classifier_BERT.ipynb`

---

## Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API

Builds a production-ready churn prediction pipeline on the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset, with preprocessing and modeling bundled into a single reusable object.

- **Approach:** `ColumnTransformer` (scaling + one-hot encoding) chained into a `Pipeline`, with Logistic Regression and Random Forest as candidate models
- **Tuning:** `GridSearchCV` for hyperparameter search on both models
- **Output:** Final pipeline exported with `joblib` — takes raw customer data in, returns churn predictions out, no manual preprocessing required

**Notebook:** `Task5_Customer_Churn_Sklearn_Pipeline.ipynb`

---

## Task 3: Multimodal ML – Housing Price Prediction Using Images + Tabular Data

Predicts house prices by combining listing photos with structured data (beds, baths, sqft, location) on the [House Prices and Images (SoCal)](https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal) dataset.

- **Approach:** Pretrained ResNet18 as an image feature extractor, fused with an MLP branch on tabular features, trained end-to-end on log-transformed price
- **Evaluation:** MAE and RMSE (in real dollars) on a held-out test set, benchmarked against a tabular-only Random Forest baseline
- **Extra:** Includes a baseline comparison to verify the image features are actually contributing to performance

**Notebook:** `Task2_Multimodal_Housing_Price_Prediction.ipynb`

---

## Task 4: Context-Aware Chatbot Using LangChain / RAG

A conversational chatbot that retrieves answers from a vectorized knowledge base and remembers prior conversation turns.

- **Approach:** Documents chunked and embedded with `sentence-transformers/all-MiniLM-L6-v2`, indexed in FAISS, retrieved via LangChain's `ConversationalRetrievalChain`
- **Memory:** `ConversationBufferMemory` for multi-turn context (handles follow-up questions like "what about...")
- **LLM:** `google/flan-t5-base` run locally, no API key required
- **Deployment:** Streamlit chat interface with source citations per answer

**Notebook:** `Task3_Context_Aware_RAG_Chatbot.ipynb`

---

## Setup

Each notebook installs its own dependencies in the first cell. General requirements across the project:

```
transformers
datasets
torch / torchvision
scikit-learn
langchain, langchain-community, langchain-huggingface
faiss-cpu
streamlit
pandas, numpy, matplotlib
```

Notebooks were developed and run on Kaggle (GPU accelerator enabled where relevant). Dataset-specific setup instructions (e.g. adding a Kaggle dataset via "Add Data") are noted at the top of each notebook.

## Repo Structure

```
.
├── Task1_News_Topic_Classifier_BERT.ipynb
├── Task2_Multimodal_Housing_Price_Prediction.ipynb
├── Task3_Context_Aware_RAG_Chatbot.ipynb
├── Task5_Customer_Churn_Sklearn_Pipeline.ipynb
└── README.md
```
