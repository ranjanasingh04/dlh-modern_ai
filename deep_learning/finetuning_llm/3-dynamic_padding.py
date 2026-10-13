#!/usr/bin/env python3
"""Create a data collator for dynamic padding."""
import transformers


def create_data_collator(tokenizer):
    """Dynamically pad each batch to its longest sequence."""
    return transformers.DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        return_tensors="pt"
    )
