#!/usr/bin/env python3
"""
Load a pre-trained RoBERTa model for Masked Language Modeling (MLM).
"""
from transformers import RobertaForMaskedLM

def load_mlm(model_name):
    """Load a pre-trained RoBERTa model for masked language modeling."""
    model = RobertaForMaskedLM.from_pretrained(model_name)
    model.eval()  # Disable dropout for inference
    return model
