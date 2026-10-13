#!/usr/bin/env python3
"""Train, evaluate and save a Hugging Face model."""
import transformers


def train_model(
    model,
    training_args,
    train_dataset,
    eval_dataset,
    test_dataset,
    tokenizer,
    data_collator,
    compute_metrics,
    model_save_name
):
    """Fine-tune, evaluate on the test set, and save the model and tokenizer."""
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    train_results = trainer.train()

    test_results = trainer.evaluate(
        eval_dataset=test_dataset,
        metric_key_prefix="test"
    )

    trainer.save_model(model_save_name)
    tokenizer.save_pretrained(model_save_name)

    return trainer, train_results, test_results
