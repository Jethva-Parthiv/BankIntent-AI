# Models Directory

Trained weights and model artifacts (`model.safetensors`, `config.json`, tokenizer files) are saved locally during training in `best_model/` and published directly to the Hugging Face Hub.

In accordance with machine learning engineering best practices, large binary model files are excluded from Git version control (see `.gitignore`).

### Hugging Face Hub Download

You can download and run inference with the published model using:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "Parthiv-Jethva/bankintent-ai"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
```
