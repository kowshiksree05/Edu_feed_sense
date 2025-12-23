from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "cardiffnlp/twitter-roberta-base-sentiment"

# Force PyTorch format
model = AutoModelForSequenceClassification.from_pretrained(model_name, from_tf=False, force_download=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)

print("Model downloaded successfully in PyTorch format!")
