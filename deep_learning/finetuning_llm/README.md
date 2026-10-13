# Emotion Classification with DistilBERT

Fine-tune DistilBERT to identify the emotion expressed in English text using the Hugging Face Emotion dataset.

```text
Input:  "I feel really disappointed."
Output: sadness (example; actual predictions depend on the trained model)
```

The six classes are **sadness, joy, love, anger, fear, and surprise**. This is multi-class emotion classification, rather than only positive/negative sentiment analysis.

## What this project achieves

1. **Train:** teach a pretrained model to classify labelled emotion examples.
2. **Evaluate:** measure performance on held-out examples.
3. **Predict:** classify new sentences using the saved model.

Training changes the model weights. Saving preserves those weights. Uploading makes them available online; uploading alone does not establish that training succeeded.

## Project files

| File | Purpose |
| --- | --- |
| `DistilBERT_Emotion_VSCode.ipynb` | Self-contained guided workflow for VS Code |
| `0-load_dataset.py` | Load the Emotion dataset |
| `1-load_distilbert.py` | Load tokenizer and classification model |
| `2-tokenize_and_map.py` | Tokenize and return train, validation, and test datasets |
| `3-dynamic_padding.py` | Pad each batch to its longest sequence |
| `4-compute_metrics.py` | Calculate accuracy and weighted precision, recall, and F1 |
| `5-configure_training_arguments.py` | Configure training and checkpoint selection |
| `6-training_mode.py` | Train, evaluate, and save |
| `7-inference_mode.py` | Create a text classification pipeline |

The notebook defines its own functions, so it does not require the separate exercise scripts to run. Script filenames should match the imports in each main file.

## Local setup: VS Code and Miniconda

Install the VS Code **Python** and **Jupyter** extensions. Activate your environment in the terminal:

```powershell
conda activate dlh-ai
```

If creating a new environment instead:

```powershell
conda create -n dlh-ai python=3.11 -y
conda activate dlh-ai
```

Install the pinned packages:

```powershell
python -m pip install --no-cache-dir torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers==4.56.1 datasets==4.0.0 accelerate==1.12.0 sentence-transformers==5.1.0 tiktoken==0.11.0 sentencepiece==0.2.1 tabulate==0.9.0 scikit-learn pandas matplotlib ipykernel
python -m pip check
```

| Package | Version |
| --- | --- |
| torch (CPU) | 2.8.0+cpu |
| transformers | 4.56.1 |
| datasets | 4.0.0 |
| accelerate | 1.12.0 |
| sentence-transformers | 5.1.0 |
| tiktoken | 0.11.0 |
| sentencepiece | 0.2.1 |
| tabulate | 0.9.0 |

The notebook also needs scikit-learn, pandas, matplotlib, and ipykernel. The first run requires internet access to download the dataset and pretrained model.

## Run the notebook

1. Save `DistilBERT_Emotion_Analysis.ipynb` in `deep_learning/finetuning_llm/` and open it in VS Code.
2. Choose **Select Kernel → Python Environments → dlh-ai**.
3. Run the cells from top to bottom. The interpreter check should point to your `dlh-ai` environment.
4. Leave the commented installation commands alone if packages are already installed.

For the small trial run, set these values in the configuration cell:

```python
SMOKE_RUN = True
RUN_TRAINING = True
```

Rerun the configuration cell, then the training cell. The trial uses 256 training examples, 128 validation examples, and one epoch. It verifies the workflow; its scores are not final model performance.

For full training, set:

```python
SMOKE_RUN = False
RUN_TRAINING = True
```

Restart the kernel and run cells in order again to start from the original pretrained model. Full training uses five epochs and can take considerable time on CPU. Keep optional upload disabled unless you intend to upload:

```python
UPLOAD_TO_HUB = False
```

## Dataset and preprocessing

The dataset is [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion).

```python
import datasets

dataset = datasets.load_dataset("dair-ai/emotion", "split")
```

Here, `"split"` is the configuration name. It is different from `split="train"`, which selects only the training partition. The configuration provides train, validation, and test sets.

Label mappings are read from the dataset metadata. Text is tokenized with a maximum length of 128 tokens and truncation enabled. Batched mapping processes multiple texts together. Dynamic padding pads only to the longest sequence in each batch.

## Training configuration

The base checkpoint is `distilbert-base-uncased`, with a classification head configured for six labels. A warning about newly initialized classification weights is expected before fine-tuning.

| Setting | Full training value |
| --- | --- |
| Epochs | 5 |
| Training batch size | 16 |
| Evaluation batch size | 32 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Evaluation and saving | Every epoch |
| Best checkpoint metric | Weighted F1 |
| Seed | 0 |

The exercise configuration function enables Hub uploads. The notebook overrides `push_to_hub=False` for local training and provides a separate optional upload cell. It uses `processing_class=tokenizer` with Trainer and `eval_strategy="epoch"` with TrainingArguments.

## Evaluation

Validation data selects the best checkpoint during training. Test data measures performance after training.

The project reports:

- **Accuracy:** fraction of correct predictions.
- **Precision:** how often a predicted emotion is correct.
- **Recall:** how many true examples of an emotion are identified.
- **F1:** balance between precision and recall.

Weighted metrics account for class frequency but can hide poor results on rare classes. Inspect the per-class classification report and confusion matrix too. Matrix rows are actual labels; columns are predictions. Diagonal entries are correct predictions.

No measured accuracy is claimed in this README. Hardware, versions, training settings, and random effects can change results.

## Save and use the model

The notebook saves the model and tokenizer beneath its printed working directory in:

```text
hbtn_emotion_classifier_best_model/
```

A complete saved model includes `config.json`, model weights, and tokenizer files. Check the directory before loading it:

```python
print("Model folder:", SAVE_DIR)
print("Config exists:", (SAVE_DIR / "config.json").is_file())
```

Use the local model:

```python
MODEL_SOURCE = str(SAVE_DIR)
```

Or use the intended Hugging Face repository, if it contains your uploaded trained model:

```python
MODEL_SOURCE = "ranjanasingh/hbtn_emotion_classifier"
```

Repository IDs must be quoted strings. Training in Colab does not automatically save model files to your Windows laptop.

Example inference:

```python
import transformers

classifier = transformers.pipeline(
    "text-classification",
    model=MODEL_SOURCE,
    tokenizer=MODEL_SOURCE,
    top_k=3,
    truncation=True,
    device=-1
)

print(classifier("I love spending time with my family."))
```

Each result contains the highest-scoring labels and their model scores. Scores are not guaranteed calibrated probabilities. Sarcasm, ambiguous wording, and unfamiliar text may produce incorrect predictions.

## Optional Hugging Face upload

Authenticate interactively using a token with write permission:

```python
import huggingface_hub

huggingface_hub.login(add_to_git_credential=False)
```

In the notebook upload cell, set:

```python
UPLOAD_TO_HUB = True
REPO_ID = "ranjanasingh/hbtn_emotion_classifier"
```

Keep the placeholder guard as `if "YOUR_USERNAME" in REPO_ID:`. Do not change the guard to your real username, as that would deliberately raise an error.

Uploading requires a complete local model in `SAVE_DIR`. If the trained model is already on the Hub and you only want predictions, skip uploading and use its repository ID. A private repository requires authentication for download. The notebook requests private visibility when creating its upload repository; check the repository settings for actual visibility.

## GitHub versus Hugging Face

| Location | What to store |
| --- | --- |
| GitHub | Notebook, Python scripts, and README |
| Local model folder | Trained weights and tokenizer files for local use |
| Hugging Face Hub | Uploaded model weights and tokenizer for later loading |


## References

- [Emotion dataset](https://huggingface.co/datasets/dair-ai/emotion)
- [DistilBERT base checkpoint](https://huggingface.co/distilbert/distilbert-base-uncased)
- [Transformers 4.56.1 Trainer documentation](https://huggingface.co/docs/transformers/v4.56.1/en/main_classes/trainer)
- [Hugging Face token settings](https://huggingface.co/settings/tokens)
- [Project model repository](https://huggingface.co/ranjanasingh/hbtn_emotion_classifier) — 