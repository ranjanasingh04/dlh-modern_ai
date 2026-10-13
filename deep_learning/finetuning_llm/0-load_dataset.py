from datasets import load_dataset


def load_emotion_dataset():
    """Load the Emotion dataset with train, validation, and test splits."""
    return load_dataset("dair-ai/emotion", "split")
