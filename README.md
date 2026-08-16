# T5 Text Summarizer

An AI-powered text summarization web application built with a fine-tuned **T5-base** Transformer model. Users can enter an article and generate a concise summary through a simple web interface.

## Features

* AI-based abstractive text summarization
* Fine-tuned T5-base Transformer model
* ROUGE-based evaluation
* Beam search text generation
* FastAPI backend
* HTML & CSS frontend
* Customizable summary length
* REST API for summarization

## Tech Stack

* **Python**
* **PyTorch / Hugging Face Transformers**
* **T5-base**
* **FastAPI**
* **HTML5 / CSS3**
* **Pandas**
* **ROUGE**
* **Google Colab**

## Model

The project uses a fine-tuned **T5-base** sequence-to-sequence Transformer model.

Training configuration includes:

* Learning rate: `3e-5`
* Batch size: `8`
* Epochs: `8`
* Maximum article length: `768`
* Maximum summary length: `128`
* FP16 mixed-precision training
* Early stopping
* ROUGE evaluation

## Dataset

The model was trained using the **BBC News Summary** dataset.

Preprocessing includes:

* Text cleaning
* Removing empty and short samples
* Duplicate removal
* Train/validation/test splitting
* T5 tokenization
* 
## API

The FastAPI backend receives article text and returns the generated summary.

## Evaluation

Model performance is evaluated using:

* ROUGE-1
* ROUGE-2
* ROUGE-L
* ROUGE-Lsum

## Project Goal

The goal of this project is to demonstrate an end-to-end **NLP Transformer application**, from dataset preprocessing and model fine-tuning to model deployment through a FastAPI backend and web frontend.

## Author

**Indunil Udayanga**

Computer Science Undergraduate | Aspiring AI & Machine Learning Engineer
