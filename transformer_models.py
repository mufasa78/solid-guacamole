import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, AutoModel, AutoTokenizer
import numpy as np

class BERTClassifier(nn.Module):
    """
    BERT-based text classification model
    """
    def __init__(self, pretrained_model_name="bert-base-chinese", num_classes=8, dropout=0.1):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.model_name = pretrained_model_name
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # BERT forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use the [CLS] token representation for classification
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_tokenizer(self):
        """
        Returns the appropriate tokenizer for this model
        """
        return BertTokenizer.from_pretrained(self.model_name)

class RoBERTaClassifier(nn.Module):
    """
    RoBERTa-based text classification model
    """
    def __init__(self, pretrained_model_name="hfl/chinese-roberta-wwm-ext", num_classes=8, dropout=0.1):
        super(RoBERTaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)
        self.model_name = pretrained_model_name
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # RoBERTa forward pass
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None
        )
        
        # Use the [CLS] token representation for classification
        # For RoBERTa, we use the last hidden state of the first token
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]  # Take the [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_tokenizer(self):
        """
        Returns the appropriate tokenizer for this model
        """
        return AutoTokenizer.from_pretrained(self.model_name)

class AutoTransformerClassifier(nn.Module):
    """
    Generic transformer-based text classification model that can use any pretrained model
    """
    def __init__(self, pretrained_model_name, num_classes=8, dropout=0.1):
        super(AutoTransformerClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        self.model_name = pretrained_model_name
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Transformer forward pass
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if hasattr(self.transformer.config, 'type_vocab_size') and self.transformer.config.type_vocab_size > 0 else None
        )
        
        # Use the [CLS] token representation for classification
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            # BERT-like models have pooler_output
            pooled_output = outputs.pooler_output
        else:
            # For models without pooler_output, use the last hidden state of the first token
            sequence_output = outputs.last_hidden_state
            pooled_output = sequence_output[:, 0, :]
            
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_tokenizer(self):
        """
        Returns the appropriate tokenizer for this model
        """
        return AutoTokenizer.from_pretrained(self.model_name)

def preprocess_for_transformer(texts, tokenizer, max_length=128):
    """
    Preprocess text data for transformer models
    
    Parameters:
    texts: List of text strings
    tokenizer: Transformer tokenizer
    max_length: Maximum sequence length
    
    Returns:
    Dictionary with input_ids, attention_mask, and token_type_ids (if applicable)
    """
    # Tokenize all texts
    encoding = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    return encoding

def batch_encode_for_transformer(texts, tokenizer, max_length=128, batch_size=32):
    """
    Encode texts in batches to avoid memory issues with large datasets
    
    Parameters:
    texts: List of text strings
    tokenizer: Transformer tokenizer
    max_length: Maximum sequence length
    batch_size: Batch size for encoding
    
    Returns:
    Dictionary with input_ids, attention_mask, and token_type_ids (if applicable)
    """
    total_samples = len(texts)
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    all_input_ids = []
    all_attention_masks = []
    all_token_type_ids = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)
        batch_texts = texts[start_idx:end_idx]
        
        encoding = tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        all_input_ids.append(encoding['input_ids'])
        all_attention_masks.append(encoding['attention_mask'])
        
        if 'token_type_ids' in encoding:
            all_token_type_ids.append(encoding['token_type_ids'])
    
    # Concatenate all batches
    result = {
        'input_ids': torch.cat(all_input_ids, dim=0),
        'attention_mask': torch.cat(all_attention_masks, dim=0)
    }
    
    if all_token_type_ids:
        result['token_type_ids'] = torch.cat(all_token_type_ids, dim=0)
    
    return result
