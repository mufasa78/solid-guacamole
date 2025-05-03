import torch
import pickle
import os
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer
from transformer_models import BERTClassifier, RoBERTaClassifier, AutoTransformerClassifier

def load_transformer_model(model_path):
    """
    Load a transformer model and its tokenizer
    
    Parameters:
    model_path: Path to the saved model file
    
    Returns:
    model: The loaded model
    tokenizer: The appropriate tokenizer
    """
    # Load tokenizer info
    tokenizer_info_path = model_path.replace('.pt', '_tokenizer_info.pkl')
    
    try:
        # Load tokenizer info
        with open(tokenizer_info_path, 'rb') as f:
            tokenizer_info = pickle.load(f)
        
        model_name = tokenizer_info['model_name']
        
        # Determine model class based on model path
        if 'BERT' in model_path and 'RoBERTa' not in model_path:
            model = BERTClassifier(
                pretrained_model_name=model_name,
                num_classes=8  # Default to 8 classes for novel classification
            )
            tokenizer = BertTokenizer.from_pretrained(model_name)
        elif 'RoBERTa' in model_path:
            model = RoBERTaClassifier(
                pretrained_model_name=model_name,
                num_classes=8
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            # Default to AutoTransformerClassifier
            model = AutoTransformerClassifier(
                pretrained_model_name=model_name,
                num_classes=8
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model parameters
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading transformer model: {str(e)}")
        return None, None

def classify_text_with_transformer(text, model, tokenizer, max_length=128):
    """
    Classify text using a transformer model
    
    Parameters:
    text: Text to classify
    model: Transformer model
    tokenizer: Tokenizer for the model
    max_length: Maximum sequence length
    
    Returns:
    predicted_class: The predicted class index
    probabilities: Probability distribution over classes
    """
    # Tokenize the text
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move to the same device as the model
    device = next(model.parameters()).device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Forward pass
    with torch.no_grad():
        if 'token_type_ids' in encoding:
            token_type_ids = encoding['token_type_ids'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get predicted class
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item(), probs[0].cpu().numpy()

def get_class_names():
    """
    Get the mapping of class indices to class names
    
    Returns:
    Dictionary mapping class indices to names
    """
    return {
        0: "玄幻",
        1: "武侠",
        2: "都市",
        3: "言情",
        4: "科幻",
        5: "历史",
        6: "游戏",
        7: "悬疑"
    }

def classify_and_explain(text, model, tokenizer, max_length=128):
    """
    Classify text and provide explanation of the prediction
    
    Parameters:
    text: Text to classify
    model: Transformer model
    tokenizer: Tokenizer for the model
    max_length: Maximum sequence length
    
    Returns:
    result: Dictionary with prediction and explanation
    """
    # Get class names
    class_names = get_class_names()
    
    # Classify the text
    predicted_class, probabilities = classify_text_with_transformer(text, model, tokenizer, max_length)
    
    # Get the predicted class name
    predicted_class_name = class_names[predicted_class]
    
    # Get the top 3 classes with their probabilities
    top_indices = probabilities.argsort()[-3:][::-1]
    top_classes = [(class_names[idx], probabilities[idx] * 100) for idx in top_indices]
    
    # Create explanation
    explanation = f"预测类别: {predicted_class_name} (置信度: {probabilities[predicted_class]*100:.2f}%)\n\n"
    explanation += "Top 3 预测:\n"
    for class_name, prob in top_classes:
        explanation += f"- {class_name}: {prob:.2f}%\n"
    
    result = {
        'predicted_class': predicted_class,
        'predicted_class_name': predicted_class_name,
        'confidence': probabilities[predicted_class],
        'top_classes': top_classes,
        'explanation': explanation
    }
    
    return result
