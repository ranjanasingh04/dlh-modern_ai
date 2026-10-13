#!/usr/bin/env python3
"""Load a pretrained DistilBERT tokenizer and classification model."""
import transformers


def load_distilbert(model_name, num_classes, id2label, label2id):
    """Load a tokenizer and a DistilBERT sequence classification model."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id
    )

    return tokenizer, model
