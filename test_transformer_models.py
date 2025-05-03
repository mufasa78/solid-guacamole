import torch
import pandas as pd
import numpy as np
from transformer_models import BERTClassifier, RoBERTaClassifier, preprocess_for_transformer
from transformer_utils import load_transformer_model, classify_text_with_transformer, classify_and_explain

def test_model_creation():
    """Test creating BERT and RoBERTa models"""
    print("Testing model creation...")
    
    # Create BERT model
    bert_model = BERTClassifier(num_classes=8)
    print(f"BERT model created with {sum(p.numel() for p in bert_model.parameters())} parameters")
    
    # Create RoBERTa model
    roberta_model = RoBERTaClassifier(num_classes=8)
    print(f"RoBERTa model created with {sum(p.numel() for p in roberta_model.parameters())} parameters")
    
    print("Model creation test passed!")

def test_tokenization():
    """Test tokenization with BERT and RoBERTa tokenizers"""
    print("\nTesting tokenization...")
    
    # Sample Chinese text
    text = "这是一个测试文本，用于测试中文分词器。"
    
    # Get tokenizers
    bert_model = BERTClassifier(num_classes=8)
    roberta_model = RoBERTaClassifier(num_classes=8)
    
    bert_tokenizer = bert_model.get_tokenizer()
    roberta_tokenizer = roberta_model.get_tokenizer()
    
    # Tokenize with BERT
    bert_tokens = bert_tokenizer.tokenize(text)
    print(f"BERT tokens: {bert_tokens}")
    
    # Tokenize with RoBERTa
    roberta_tokens = roberta_tokenizer.tokenize(text)
    print(f"RoBERTa tokens: {roberta_tokens}")
    
    # Test preprocessing
    bert_encoding = preprocess_for_transformer([text], bert_tokenizer)
    print(f"BERT encoding shape: {bert_encoding['input_ids'].shape}")
    
    roberta_encoding = preprocess_for_transformer([text], roberta_tokenizer)
    print(f"RoBERTa encoding shape: {roberta_encoding['input_ids'].shape}")
    
    print("Tokenization test passed!")

def test_forward_pass():
    """Test forward pass with BERT and RoBERTa models"""
    print("\nTesting forward pass...")
    
    # Sample Chinese text
    text = "这是一个测试文本，用于测试中文分词器。"
    
    # Create models
    bert_model = BERTClassifier(num_classes=8)
    roberta_model = RoBERTaClassifier(num_classes=8)
    
    # Get tokenizers
    bert_tokenizer = bert_model.get_tokenizer()
    roberta_tokenizer = roberta_model.get_tokenizer()
    
    # Preprocess text
    bert_encoding = preprocess_for_transformer([text], bert_tokenizer)
    roberta_encoding = preprocess_for_transformer([text], roberta_tokenizer)
    
    # Forward pass with BERT
    bert_outputs = bert_model(
        input_ids=bert_encoding['input_ids'],
        attention_mask=bert_encoding['attention_mask'],
        token_type_ids=bert_encoding['token_type_ids']
    )
    print(f"BERT output shape: {bert_outputs.shape}")
    
    # Forward pass with RoBERTa
    roberta_outputs = roberta_model(
        input_ids=roberta_encoding['input_ids'],
        attention_mask=roberta_encoding['attention_mask']
    )
    print(f"RoBERTa output shape: {roberta_outputs.shape}")
    
    print("Forward pass test passed!")

def test_classification():
    """Test text classification with sample texts"""
    print("\nTesting text classification...")
    
    # Create models
    bert_model = BERTClassifier(num_classes=8)
    bert_tokenizer = bert_model.get_tokenizer()
    
    # Sample texts for different genres
    sample_texts = {
        "玄幻": "龙王大人挥动神剑，一道金光划破天际，整个修真界都为之震动。他体内的灵力如江河奔涌，修为已达到前所未有的境界。",
        "武侠": "剑客独立山巅，长剑出鞘，剑气纵横三千里。他的武功已达到了出神入化的境界，整个江湖都在传颂他的名号。",
        "科幻": "宇宙飞船穿越虫洞，来到了一个全新的星系。这里的科技水平远超地球，人类正在与外星文明建立第一次接触。"
    }
    
    for genre, text in sample_texts.items():
        # Classify text
        predicted_class, probabilities = classify_text_with_transformer(text, bert_model, bert_tokenizer)
        
        # Get result with explanation
        result = classify_and_explain(text, bert_model, bert_tokenizer)
        
        print(f"\nGenre: {genre}")
        print(f"Text: {text[:50]}...")
        print(f"Prediction: {result['predicted_class_name']} (Confidence: {result['confidence']*100:.2f}%)")
        print("Top 3 predictions:")
        for class_name, prob in result['top_classes']:
            print(f"- {class_name}: {prob:.2f}%")
    
    print("\nClassification test passed!")

def main():
    """Run all tests"""
    print("=== Testing Transformer Models ===\n")
    
    # Run tests
    test_model_creation()
    test_tokenization()
    test_forward_pass()
    test_classification()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()
