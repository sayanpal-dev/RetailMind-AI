from pathlib import Path
import copy
import csv
import time

import pandas as pd
import torch

from torch.utils.data import (
    Dataset,
    DataLoader
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from day13_data import load_review_data


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

RANDOM_STATE = 42

MODEL_NAME = (
    "distilbert-base-uncased"
)

MAX_LENGTH = 128

BATCH_SIZE = 8

EPOCHS = 2

LEARNING_RATE = 2e-5

MAXIMUM_REVIEWS = 3000


MODEL_DIRECTORY = Path(
    "models/day13_bert_sentiment"
)

RESULTS_PATH = Path(
    "models/day13_sentiment_results.csv"
)

COMPARISON_PATH = Path(
    "models/day13_baseline_comparison.csv"
)

DAY12_RESULTS_PATH = Path(
    "models/day12_results.csv"
)


MODEL_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


torch.manual_seed(
    RANDOM_STATE
)


# --------------------------------------------------
# 2. Device
# --------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)

print("DAY 13: BERT SENTIMENT CLASSIFICATION")

print("=" * 60)

print("Device:", device)


# --------------------------------------------------
# 3. Load data
# --------------------------------------------------

reviews_df = load_review_data(
    maximum_rows=MAXIMUM_REVIEWS
)


train_df, test_df = train_test_split(
    reviews_df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=reviews_df["label"]
)


train_df, validation_df = (
    train_test_split(
        train_df,
        test_size=0.10,
        random_state=RANDOM_STATE,
        stratify=train_df["label"]
    )
)


print("Training reviews  :", len(train_df))

print("Validation reviews:", len(validation_df))

print("Testing reviews   :", len(test_df))


# --------------------------------------------------
# 4. Tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# --------------------------------------------------
# 5. Dataset class
# --------------------------------------------------

class ReviewDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        maximum_length
    ):

        self.texts = (
            dataframe["text"]
            .tolist()
        )

        self.labels = (
            dataframe["label"]
            .astype(int)
            .tolist()
        )

        self.tokenizer = tokenizer

        self.maximum_length = (
            maximum_length
        )


    def __len__(self):

        return len(
            self.texts
        )


    def __getitem__(
        self,
        index
    ):

        encoded = self.tokenizer(
            self.texts[index],
            padding="max_length",
            truncation=True,
            max_length=self.maximum_length,
            return_tensors="pt"
        )

        item = {
            "input_ids": (
                encoded["input_ids"]
                .squeeze(0)
            ),

            "attention_mask": (
                encoded["attention_mask"]
                .squeeze(0)
            ),

            "labels": torch.tensor(
                self.labels[index],
                dtype=torch.long
            )
        }

        return item


# --------------------------------------------------
# 6. Data loaders
# --------------------------------------------------

train_dataset = ReviewDataset(
    train_df,
    tokenizer,
    MAX_LENGTH
)

validation_dataset = ReviewDataset(
    validation_df,
    tokenizer,
    MAX_LENGTH
)

test_dataset = ReviewDataset(
    test_df,
    tokenizer,
    MAX_LENGTH
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# 7. Load pretrained model
# --------------------------------------------------

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={
            0: "Negative",
            1: "Positive"
        },
        label2id={
            "Negative": 0,
            "Positive": 1
        }
    )
)

model.to(
    device
)


# --------------------------------------------------
# 8. Optimizer and scheduler
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


total_steps = (
    len(train_loader) *
    EPOCHS
)


scheduler = (
    get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(
            1,
            total_steps // 10
        ),
        num_training_steps=total_steps
    )
)


# --------------------------------------------------
# 9. Evaluation helper
# --------------------------------------------------

def evaluate_model(
    model,
    data_loader
):

    model.eval()

    total_loss = 0.0

    actual_labels = []

    predicted_labels = []


    with torch.no_grad():

        for batch in data_loader:

            batch = {
                name: tensor.to(device)
                for name, tensor
                in batch.items()
            }

            output = model(
                **batch
            )

            total_loss += float(
                output.loss.item()
            )

            predictions = torch.argmax(
                output.logits,
                dim=1
            )

            actual_labels.extend(
                batch["labels"]
                .cpu()
                .tolist()
            )

            predicted_labels.extend(
                predictions
                .cpu()
                .tolist()
            )


    average_loss = (
        total_loss /
        max(
            len(data_loader),
            1
        )
    )

    return (
        average_loss,
        actual_labels,
        predicted_labels
    )


# --------------------------------------------------
# 10. Fine-tune
# --------------------------------------------------

best_validation_loss = float(
    "inf"
)

best_model_state = None

training_start = time.time()


for epoch in range(
    EPOCHS
):

    model.train()

    epoch_loss = 0.0


    for batch_number, batch in enumerate(
        train_loader,
        start=1
    ):

        batch = {
            name: tensor.to(device)
            for name, tensor
            in batch.items()
        }

        optimizer.zero_grad()

        output = model(
            **batch
        )

        loss = output.loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        scheduler.step()

        epoch_loss += float(
            loss.item()
        )


        if batch_number % 50 == 0:

            print(
                f"Epoch {epoch + 1} | "
                f"Batch {batch_number}/"
                f"{len(train_loader)}"
            )


    average_train_loss = (
        epoch_loss /
        len(train_loader)
    )


    validation_loss, _, _ = (
        evaluate_model(
            model,
            validation_loader
        )
    )


    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Training loss  : "
        f"{average_train_loss:.4f}"
    )

    print(
        f"Validation loss: "
        f"{validation_loss:.4f}"
    )


    if (
        validation_loss <
        best_validation_loss
    ):

        best_validation_loss = (
            validation_loss
        )

        best_model_state = copy.deepcopy(
            model.state_dict()
        )


training_seconds = (
    time.time() -
    training_start
)


# --------------------------------------------------
# 11. Restore best model
# --------------------------------------------------

if best_model_state is not None:

    model.load_state_dict(
        best_model_state
    )


# --------------------------------------------------
# 12. Test results
# --------------------------------------------------

test_loss, actual, predicted = (
    evaluate_model(
        model,
        test_loader
    )
)


accuracy = accuracy_score(
    actual,
    predicted
)

precision = precision_score(
    actual,
    predicted,
    zero_division=0
)

recall = recall_score(
    actual,
    predicted,
    zero_division=0
)

f1 = f1_score(
    actual,
    predicted,
    zero_division=0
)

matrix = confusion_matrix(
    actual,
    predicted
)


print("\n" + "=" * 60)

print("DAY 13 BERT SENTIMENT RESULTS")

print("=" * 60)

print(f"Test loss : {test_loss:.4f}")

print(f"Accuracy  : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1-score  : {f1:.4f}")

print(
    f"Time      : "
    f"{training_seconds:.2f} seconds"
)

print("\nConfusion matrix:")

print(matrix)

print("\nClassification report:")

print(
    classification_report(
        actual,
        predicted,
        target_names=[
            "Negative",
            "Positive"
        ],
        zero_division=0
    )
)


# --------------------------------------------------
# 13. Save model and tokenizer
# --------------------------------------------------

model.save_pretrained(
    MODEL_DIRECTORY
)

tokenizer.save_pretrained(
    MODEL_DIRECTORY
)


# --------------------------------------------------
# 14. Save results
# --------------------------------------------------

result_row = {
    "Model":
        "DistilBERT Sentiment",

    "Training Reviews":
        len(train_df),

    "Testing Reviews":
        len(test_df),

    "Accuracy":
        round(accuracy, 4),

    "Precision":
        round(precision, 4),

    "Recall":
        round(recall, 4),

    "F1":
        round(f1, 4),

    "Training Seconds":
        round(training_seconds, 2)
}


pd.DataFrame([
    result_row
]).to_csv(
    RESULTS_PATH,
    index=False
)


# --------------------------------------------------
# 15. Compare with Day 12
# --------------------------------------------------

comparison_rows = []


if DAY12_RESULTS_PATH.exists():

    day12_results = pd.read_csv(
        DAY12_RESULTS_PATH
    )

    day12_row = (
        day12_results.iloc[0]
    )

    comparison_rows.append({
        "Model":
            "Day 12 TF-IDF Baseline",

        "Accuracy":
            day12_row.get(
                "Accuracy",
                ""
            ),

        "Precision":
            day12_row.get(
                "Precision",
                ""
            ),

        "Recall":
            day12_row.get(
                "Recall",
                ""
            ),

        "F1":
            day12_row.get(
                "F1",
                ""
            )
    })


comparison_rows.append({
    "Model":
        "Day 13 DistilBERT",

    "Accuracy":
        round(accuracy, 4),

    "Precision":
        round(precision, 4),

    "Recall":
        round(recall, 4),

    "F1":
        round(f1, 4)
})


pd.DataFrame(
    comparison_rows
).to_csv(
    COMPARISON_PATH,
    index=False
)


print("\nGenerated files:")

print("BERT model :", MODEL_DIRECTORY)

print("Results    :", RESULTS_PATH)

print("Comparison :", COMPARISON_PATH)