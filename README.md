# SOAPGemma - Medical SOAP Note Generator

![SOAPGemma Banner](../DOCS/img.png)

## Table of Contents
1.  [Introduction](#introduction)
2.  [What is a SOAP Note?](#what-is-a-soap-note)
3.  [What is TxGemma?](#what-is-txgemma)
4.  [What is the OMI Dataset?](#what-is-the-omi-dataset)
5.  [Project Structure](#project-structure)
6.  [Model Details](#model-details)
7.  [Key Features](#key-features)
8.  [Setup and Installation](#setup-and-installation)
9.  [Usage](#usage)
    * [Data Processing](#data-processing)
    * [Model Fine-tuning](#model-fine-tuning)
    * [Inference](#inference)
    * [Evaluation](#evaluation)
10. [Notebooks Overview](#notebooks-overview)
11. [Scripts Overview](#scripts-overview)
12. [Evaluation Metrics](#evaluation-metrics)
13. [License](#license)
14. [Acknowledgements](#acknowledgements)

## Introduction
SOAPGemma is a project focused on fine-tuning the TxGemma model (a derivative of Google's Gemma 2 architecture) to automatically generate medical SOAP (Subjective, Objective, Assessment, Plan) notes from patient-doctor dialogues. The primary dataset used for this task is the OMI (Open Medical Insight) dataset, which consists of synthetic medical dialogues and their corresponding SOAP summaries.

This project aims to streamline the clinical documentation process by leveraging advanced natural language processing techniques.

## What is a SOAP Note?
A SOAP note is a structured method of documentation used by healthcare providers. The acronym SOAP stands for:

* **S - Subjective**: This section captures information reported by the patient, such as their feelings, concerns, and their description of symptoms (e.g., "stomach pain," "nauseated, fatigued"). It includes the patient's chief complaint and the history of their present illness.
* **O - Objective**: This part includes observable, measurable, and factual data collected by the clinician. This encompasses vital signs (temperature, heart rate, blood pressure, etc.), physical exam findings, laboratory results, and imaging data.
* **A - Assessment**: Here, the clinician provides their professional judgment and diagnosis based on the subjective and objective information gathered. It involves an analysis of the patient's condition, potential diagnoses, and the patient's progress.
* **P - Plan**: This section outlines the treatment plan, including any further tests, therapies, medications, referrals to specialists, and follow-up actions.

SOAP notes are a crucial tool for healthcare workers to organize patient information, guide clinical reasoning, and facilitate communication among health professionals.

## What is TxGemma?
TxGemma is a collection of open-source machine learning models designed to improve the efficiency of therapeutic development. These models are fine-tuned from Google DeepMind's Gemma 2 architecture using a large dataset (7 million training examples) from the Therapeutics Data Commons (TDC), which includes information on small molecules, proteins, nucleic acids, diseases, and cell lines. TxGemma models are built to predict therapeutic properties, perform classification, regression, and generation tasks, facilitate conversational AI, and support agentic orchestration.
*(Source: Google AI Blog)*

## What is the OMI Dataset?
The OMI dataset used in this project consists of 10,000 synthetic dialogues between a patient and clinician, created using the GPT-4 dataset from NoteChat, based on PubMed Central (PMC) case reports. Accompanying these dialogues are SOAP summaries generated through GPT-4. The dataset is split into training, validation, and test sets.
*(Source: Junxian Tang et al., "NoteChat: A Dataset of Synthetic Doctor-Patient Conversations Conditioned on Clinical Notes," arXiv)*

## Project Structure

medical-soap-note-generator/
├── .git/                     # Git directory (hidden)
├── .gitignore                # Specifies intentionally untracked files
├── data/
│   ├── raw/                  # Original, immutable data (e.g., conversations, reports)
│   │   └── .gitkeep          # Keeps the directory in git even if empty initially
│   ├── processed/            # Cleaned and preprocessed data
│   │   └── .gitkeep
│   └── annotated/            # Data with SOAP note annotations (if you create these)
│       └── .gitkeep
├── notebooks/                # Jupyter notebooks for exploration, experimentation
│   ├── data_processing.ipynb
│   ├── tx-gemma-demo.ipynb
│   ├── tx-gemma-soap-note-predict.ipynb
│   ├── tx-gemma-soap-notes.ipynb
│   └── tx_gemma_soap_note_eval.ipynb
├── src/                      # Main source code for the project
│   ├── init.py
│   ├── data_processing.py    # Scripts for cleaning, transforming data
│   ├── model.py              # Model definition, fine-tuning logic, and single inference
│   ├── inference.py          # Script for generating SOAP notes (callable)
│   ├── inference_batch.py    # Script for batch SOAP note generation
│   ├── train.py              # Script to run the training process (if separated from notebook)
│   ├── evaluate.py           # Script for model evaluation (if separated from notebook)
│   └── utils.py              # Utility functions
│   └── test_data/            # Sample data for testing inference
│       └── d1.txt
├── models_checkpoint/        # Saved model checkpoints
│   └── SOAPgemma_v1/         # Fine-tuned PEFT adapter
│       ├── README.md
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── tokenizer.model
│   └── .gitkeep
├── tests/                    # Unit tests and integration tests
│   ├── init.py
│   └── test_data_processing.py # Example test file
├── requirements.txt          # List of Python dependencies
├── environment.yml           # (Optional) For Conda environments
├── LICENSE                   # Your chosen license file
└── README.md                 # Project overview, setup, usage instructions


## Model Details
* **Base Model:** `google/txgemma-2b-predict`
* **Fine-tuned Adapter:** `SOAPgemma_v1` (LoRA-based PEFT adapter)
* **Quantization:** The model utilizes 4-bit quantization (bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16) to reduce memory usage.

## Key Features
* **Data Processing:** Scripts and notebooks for loading, cleaning, and preparing the OMI dataset.
* **LoRA Fine-tuning:** Efficient fine-tuning of the TxGemma model using Low-Rank Adaptation.
* **SOAP Note Generation:** Inference capabilities for generating structured SOAP notes from medical dialogues.
* **Batch Inference:** Support for generating SOAP notes for multiple dialogues in a batch.
* **Model Evaluation:** Scripts and notebooks for evaluating the generated SOAP notes using metrics like ROUGE, BERTScore, and BLEU.

## Setup and Installation
1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd medical-soap-note-generator
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```
3.  **Install dependencies:**
    The `requirements.txt` file lists the necessary Python packages.
    ```bash
    pip install -r requirements.txt
    # Ensure you have PyTorch installed with CUDA support if you have a GPU.
    # Example for CUDA 12.1:
    # pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    # pip install bitsandbytes datasets peft transformers trl rouge_score bert_score nltk
    ```
    (Note: The `requirements.txt` file in the provided project is currently empty. Users should populate it with necessary packages like `torch`, `transformers`, `datasets`, `peft`, `bitsandbytes`, `trl`, `pandas`, `rouge_score`, `bert_score`, `nltk` etc.)

## Usage

### Data Processing
* The `notebooks/data_processing.ipynb` notebook details the steps for loading the OMI dataset, extracting event tags, and splitting SOAP notes into their S, O, A, P components.
* The corresponding script `src/data_processing.py` can be used to automate these steps. Raw data is expected in `data/raw/omi-health/` and processed data will be saved to `data/processed/omi-health/`.

### Model Fine-tuning
* The primary notebook for fine-tuning is `notebooks/tx-gemma-soap-notes.ipynb`.
* This notebook covers:
    * Loading the base TxGemma model and tokenizer.
    * Applying 4-bit quantization.
    * Configuring LoRA.
    * Preparing the dataset and formatting it for training.
    * Initializing and running the `SFTTrainer` from the `trl` library.
    * Saving the fine-tuned LoRA adapter and tokenizer to `models_checkpoint/SOAPgemma_v1/`.

### Inference
There are multiple ways to perform inference:

1.  **Interactive Single Inference (using `src/model.py`):**
    This script allows you to load the fine-tuned model and provide a path to a dialogue text file interactively to generate a SOAP note.
    ```bash
    python src/model.py
    ```
    You can also provide arguments:
    ```bash
    python src/model.py --dialogue-file src/test_data/d1.txt --adapter_path models_checkpoint/SOAPgemma_v1
    ```
    The script includes a spinner animation during generation.

2.  **Batch Inference (using `notebooks/tx-gemma-soap-note-predict.ipynb`):**
    This notebook demonstrates how to load the test dataset and generate SOAP notes for all dialogues in a batch.
    * It uses the `src/inference_batch.py` script which contains the `generate_batch` function.
    * Generated notes are saved to a CSV file (e.g., `data/predicted/omi-health/test_v1.csv`).

3.  **Callable Function (using `src/inference.py`):**
    The `generate_soap_note` function in `src/inference.py` can be imported and used in other scripts or applications.

### Evaluation
* The `notebooks/tx_gemma_soap_note_eval.ipynb` notebook is used for evaluating the quality of the generated SOAP notes.
* It calculates:
    * **ROUGE scores** (ROUGE-1, ROUGE-2, ROUGE-L) for lexical overlap.
    * **BERTScore** for semantic similarity.
    * **BLEU score** for n-gram precision.
* Results are visualized using `matplotlib`.

## Notebooks Overview
* `notebooks/data_processing.ipynb`: Data loading, preprocessing, event tag extraction, and SOAP component splitting.
* `notebooks/tx-gemma-demo.ipynb`: Demonstrates basic capabilities of the base TxGemma model.
* `notebooks/tx-gemma-soap-notes.ipynb`: Core notebook for fine-tuning SOAPGemma using LoRA, including model quantization, data formatting, training, and saving the model.
* `notebooks/tx-gemma-soap-note-predict.ipynb`: Batch inference on a test set using the fine-tuned SOAPGemma model.
* `notebooks/tx_gemma_soap_note_eval.ipynb`: Evaluation of generated SOAP notes using ROUGE, BERTScore, and BLEU metrics.

## Scripts Overview (`src/`)
* `src/data_processing.py`: Contains functions for processing the medical dialogue data.
* `src/model.py`: Defines the model loading, interactive inference logic with spinner animation.
* `src/inference.py`: Contains the core `generate_soap_note` function for single dialogue inference.
* `src/inference_batch.py`: Contains the `generate_batch` function for processing multiple dialogues.
* `src/train.py`: (If created) Would contain the script to run the training process from the command line.
* `src/evaluate.py`: (If created) Would contain the script for model evaluation from the command line.
* `src/utils.py`: (If created) Would contain utility functions used across the project.

## Evaluation Metrics
The model's performance is evaluated using:
* **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** Measures overlap of n-grams, word sequences, and word pairs between the generated and reference summaries.
    * ROUGE-1: Unigram overlap.
    * ROUGE-2: Bigram overlap.
    * ROUGE-L: Longest common subsequence.
* **BERTScore:** Leverages pre-trained BERT embeddings to assess semantic similarity between the generated and reference texts.
* **BLEU (Bilingual Evaluation Understudy):** Measures precision of n-grams in the generated text compared to reference texts.

The fine-tuned TxGemma model (SOAPGemma) generates SOAP notes that are semantically accurate (high BERTScore) but may not always perfectly match the original phrasing (indicated by ROUGE and BLEU scores). For medical summarization, semantic accuracy is often prioritized.

## License
MIT License

Copyright (c) 2025 Shalinda Silva

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgements
* The OMI dataset creators (Junxian Tang et al., "NoteChat").
* Google for the Gemma and TxGemma models.
* The developers of Hugging Face Transformers, PEFT, TRL, and other open-source libraries used in this project.

