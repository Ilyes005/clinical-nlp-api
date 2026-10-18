
# Clinical NLP API

A Python-based Natural Language Processing (NLP) project
that classifies clinical-style text using a pretrained
Hugging Face Transformer model and exposes predictions
through a REST API built with FastAPI.

## Features

- Zero-shot text classification
- Four categories: symptoms, diagnosis, treatment, medication
- REST API with FastAPI
- Input validation using Pydantic
- Automated API tests using pytest
- Model evaluation using scikit-learn
- Interactive Swagger API documentation

## Technologies

- Python 3.12
- PyTorch
- Hugging Face Transformers
- facebook/bart-large-mnli
- FastAPI
- scikit-learn
- pytest

## Project Structure

    clinical-nlp-api/
    ├── model.py
    ├── api.py
    ├── evaluate.py
    ├── requirements.txt
    ├── tests/
    │   └── test_api.py
    └── README.md


## Installation

Clone the repository:

```bash
git clone https://github.com/Ilyes005/clinical-nlp-api.git
cd clinical-nlp-api
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```



## Run the API

Start the server:

```powershell
python -m uvicorn api:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs


## Example Request

POST /predict

    {
      "text": "The patient reports persistent headaches."
    }

## Example Response

    {
      "text": "The patient reports persistent headaches.",
      "label": "symptoms",
      "score": 0.89
    }

The response is illustrative. Scores depend on model output.

## Model Evaluation

The pretrained model was evaluated on two small
synthetic datasets.

| Dataset | Samples | Accuracy |
|---------|---------|----------|
| Basic examples | 12 | 100% |
| Extended examples | 16 | 100% |

These results are exploratory and do not establish
real-world performance. The examples are small,
synthetic, and some have ambiguous labels.

## Automated Testing

Run:

    python -m pytest -v

Four automated API tests cover:

- Homepage response
- Empty text validation
- Missing input validation
- Prediction endpoint using a mocked classifier

## Limitations

- The model is not specifically trained for clinical use.
- Classification scores are not calibrated probabilities.
- The classifier selects one label even when multiple
  categories could apply.
- Evaluation data is small and synthetic.
- The API is intended for local demonstration.

## Disclaimer

This is an educational Machine Learning project.
It is not a medical device and must not be used
for clinical decisions or real patient data.
