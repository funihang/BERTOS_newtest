import os
import random
import numpy as np
import torch

from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizerFast,
    AutoConfig,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    get_scheduler,
)
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

# ============================================================
# Configuration
# ============================================================

DATA_ROOT = "./dataset/ICSD_CN"
TRAIN_FILE = "train.txt"
VALIDATION_FILE = "validation.txt"
TEST_FILE = "test.txt"

TOKENIZER_PATH = "./tokenizer"
CONFIG_PATH="./random_config"
OUTPUT_DIR = "./trained_model"

MAX_LENGTH = 100
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
NUM_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 20

SEED = 42


# ============================================================
# Labels
# ============================================================

LABEL_LIST = [
    "-5",
    "-4",
    "-3",
    "-2",
    "-1",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
]

LABEL_TO_ID = {
    label: i
    for i, label in enumerate(LABEL_LIST)
}

ID_TO_LABEL = {
    i: label
    for i, label in enumerate(LABEL_LIST)
}


# ============================================================
# Random seed
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        print("CUDA available")
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Read BERTOS txt dataset
# ============================================================

def read_materials_file(filepath):

    examples = []

    tokens = []
    ner_tags = []

    guid = 0

    with open(filepath, encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            # Empty line -> end of one material
            if not line:

                if tokens:

                    examples.append(
                        {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                    )

                    guid += 1
                    tokens = []
                    ner_tags = []

                continue

            # Original format:
            # token oxidation_state
            splits = line.split()

            tokens.append(splits[0])
            ner_tags.append(splits[1])

    # Last example
    if tokens:

        examples.append(
            {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": ner_tags,
            }
        )

    return examples


# ============================================================
# Load dataset
# ============================================================

def load_materials_dataset():

    train_data = read_materials_file(
        os.path.join(DATA_ROOT, TRAIN_FILE)
    )

    validation_data = read_materials_file(
        os.path.join(DATA_ROOT, VALIDATION_FILE)
    )

    test_data = read_materials_file(
        os.path.join(DATA_ROOT, TEST_FILE)
    )

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(validation_data),
            "test": Dataset.from_list(test_data),
        }
    )

    return dataset


# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, "./runs/"))

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    raw_datasets = load_materials_dataset()

    # print(raw_datasets)

    print(
        "Training samples:",
        len(raw_datasets["train"])
    )

    print(
        "Validation samples:",
        len(raw_datasets["validation"])
    )

    print(
        "Test samples:",
        len(raw_datasets["test"])
    )


    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = BertTokenizerFast.from_pretrained(
        TOKENIZER_PATH,
        do_lower_case=False,
    )

    print(
        "Vocabulary size:",
        len(tokenizer)
    )


    # --------------------------------------------------------
    # Model configuration
    # --------------------------------------------------------

    config = AutoConfig.from_pretrained(
        CONFIG_PATH,
        num_labels=len(LABEL_LIST),
        label2id=LABEL_TO_ID,
        id2label=ID_TO_LABEL,
    )


    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print("\nCreating BERTOS model...")

    model = AutoModelForTokenClassification.from_config(
        config
    )

    model.resize_token_embeddings(
        len(tokenizer)
    )

    model.to(device)


    # --------------------------------------------------------
    # Tokenization and label alignment
    # --------------------------------------------------------

    def tokenize_and_align_labels(examples):

        tokenized_inputs = tokenizer(
            examples["tokens"],
            max_length=MAX_LENGTH,
            padding=False,
            truncation=True,
            is_split_into_words=True,
        )

        labels = []

        for i, label in enumerate(
            examples["ner_tags"]
        ):

            word_ids = tokenized_inputs.word_ids(
                batch_index=i
            )

            previous_word_idx = None

            label_ids = []

            for word_idx in word_ids:

                # Special token
                if word_idx is None:

                    label_ids.append(-100)

                # First sub-token
                elif word_idx != previous_word_idx:

                    label_ids.append(
                        LABEL_TO_ID[
                            str(label[word_idx])
                        ]
                    )

                # Other sub-token
                else:

                    label_ids.append(-100)

                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels

        return tokenized_inputs


    # --------------------------------------------------------
    # Process datasets
    # --------------------------------------------------------

    print("\nTokenizing dataset...")

    processed_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=[
            "id",
            "tokens",
            "ner_tags",
        ],
    )


    # --------------------------------------------------------
    # Data collator
    # --------------------------------------------------------

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )


    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_dataloader = DataLoader(
        processed_datasets["train"],
        shuffle=True,
        batch_size=TRAIN_BATCH_SIZE,
        collate_fn=data_collator,
    )

    validation_dataloader = DataLoader(
        processed_datasets["validation"],
        shuffle=False,
        batch_size=EVAL_BATCH_SIZE,
        collate_fn=data_collator,
    )


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    total_steps = (
        len(train_dataloader)
        * NUM_EPOCHS
    )

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )


    # ========================================================
    # Training
    # ========================================================

    print("\n" + "=" * 60)
    print("Start training")
    print("=" * 60)
    
    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):

        model.train()

        total_loss = 0.0

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
        )

        for batch in progress_bar:

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            loss = outputs.loss

            total_loss += loss.item()

            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        average_loss = (
            total_loss /
            len(train_dataloader)
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        validation_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for batch in validation_dataloader:

                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }

                outputs = model(**batch)

                validation_loss += (
                    outputs.loss.item()
                )

                predictions = (
                    outputs.logits.argmax(
                        dim=-1
                    )
                )

                labels = batch["labels"]

                mask = labels != -100

                correct += (
                    predictions[mask]
                    == labels[mask]
                ).sum().item()

                total += mask.sum().item()


        average_validation_loss = (
            validation_loss
            / len(validation_dataloader)
        )

        
        if average_validation_loss < best_val_loss:
            best_val_loss = average_validation_loss
            patience_counter = 0

            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
        else:

            patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(
                    f"\nEarly stopping at epoch {epoch + 1}. "
                    f"Best validation loss: {best_val_loss:.4f}"
                )
                break

        accuracy = (
            correct / total
            if total > 0
            else 0.0
        )

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )


        writer.add_scalar(
            "Learning_Rate",
            optimizer.param_groups[0]["lr"],
            epoch + 1
        )

        writer.add_scalar(
            "Loss/train",
            average_loss,
            epoch + 1
        )

        writer.add_scalar(
            "Loss/validation",
            average_validation_loss,
            epoch + 1
        )

        writer.add_scalar(
            "Accuracy/validation",
            accuracy,
            epoch + 1
        )

    writer.close()


if __name__ == "__main__":
    main()
