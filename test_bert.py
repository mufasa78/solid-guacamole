from transformer_models import BERTClassifier

# Create a BERT model
model = BERTClassifier(num_classes=8)
print(f"BERT model created with {sum(p.numel() for p in model.parameters())} parameters")

# Get tokenizer
tokenizer = model.get_tokenizer()
print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

# Test tokenization
text = "这是一个测试文本，用于测试中文分词器。"
tokens = tokenizer.tokenize(text)
print(f"BERT tokens: {tokens}")
