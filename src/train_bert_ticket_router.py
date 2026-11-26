from pathlib import Path
import copy

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
    precision_recall_fscore_support,
    classification_report
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from day13_data import load_review_data


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

MODEL_NAME = (
    "distilbert-base-uncased"
)

MAX_LENGTH = 128

BATCH_SIZE = 8

EPOCHS = 2

LEARNING_RATE = 2e-5

MAXIMUM_REVIEWS = 4000


OUTPUT_DIRECTORY = Path(
    "models/day13_ticket_router"
)

RESULTS_PATH = Path(
    "models/day13_ticket_results.csv"
)


OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


LABEL_NAMES = {
    0: "Delivery",
    1: "Refund",
    2: "Product Issue",
    3: "General Feedback"
}


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# --------------------------------------------------
# 2. Weak ticket-label rules
# --------------------------------------------------

DELIVERY_WORDS = {
    "delivery",
    "delivered",
    "shipping",
    "shipment",
    "arrived",
    "arrival",
    "package",
    "courier",
    "late"
}


REFUND_WORDS = {
    "refund",
    "return",
    "returned",
    "money back",
    "replacement",
    "replace",
    "exchange"
}


PRODUCT_WORDS = {
    "broken",
    "defective",
    "damaged",
    "stopped working",
    "not working",
    "faulty",
    "poor quality",
    "missing part"
}


def assign_ticket_category(
    text
):

    text = str(
        text
    ).lower()


    if any(
        keyword in text
        for keyword in DELIVERY_WORDS
    ):
        return 0


    if any(
        keyword in text
        for keyword in REFUND_WORDS
    ):
        return 1


    if any(
        keyword in text
        for keyword in PRODUCT_WORDS
    ):
        return 2


    return 3


# --------------------------------------------------
# 3. Load reviews and create labels
# --------------------------------------------------

reviews_df = load_review_data(
    maximum_rows=MAXIMUM_REVIEWS
)


tickets_df = pd.DataFrame({
    "text":
        reviews_df["text"],

    "label":
        reviews_df["text"]
        .apply(
            assign_ticket_category
        )
})


print("=" * 60)

print("DAY 13: BERT SUPPORT-TICKET ROUTER")

print("=" * 60)

print("\nWeak-label distribution:")

print(
    tickets_df["label"]
    .map(LABEL_NAMES)
    .value_counts()
)


# Keep at most the same number per category
smallest_count = int(
    tickets_df["label"]
    .value_counts()
    .min()
)


samples_per_class = min(
    smallest_count,
    500
)


balanced_parts = []


for label_value in LABEL_NAMES:

    category_data = tickets_df[
        tickets_df["label"] ==
        label_value
    ]

    balanced_parts.append(
        category_data.sample(
            n=samples_per_class,
            random_state=42
        )
    )


tickets_df = pd.concat(
    balanced_parts,
    ignore_index=True
).sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)


train_df, test_df = train_test_split(
    tickets_df,
    test_size=0.20,
    random_state=42,
    stratify=tickets_df["label"]
)


# --------------------------------------------------
# 4. Tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# --------------------------------------------------
# 5. Dataset
# --------------------------------------------------

class TicketDataset(Dataset):

    def __init__(
        self,
        dataframe
    ):

        self.texts = dataframe[
            "text"
        ].tolist()

        self.labels = dataframe[
            "label"
        ].astype(
            int
        ).tolist()


    def __len__(self):

        return len(
            self.texts
        )


    def __getitem__(
        self,
        index
    ):

        encoded = tokenizer(
            self.texts[index],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {
            "input_ids":
                encoded["input_ids"]
                .squeeze(0),

            "attention_mask":
                encoded["attention_mask"]
                .squeeze(0),

            "labels":
                torch.tensor(
                    self.labels[index],
                    dtype=torch.long
                )
        }


train_loader = DataLoader(
    TicketDataset(
        train_df
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)


test_loader = DataLoader(
    TicketDataset(
        test_df
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# 6. Model
# --------------------------------------------------

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_NAME,
        num_labels=4,
        id2label=LABEL_NAMES,
        label2id={
            value: key
            for key, value
            in LABEL_NAMES.items()
        }
    )
)


model.to(
    device
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# 7. Train
# --------------------------------------------------

for epoch in range(
    EPOCHS
):

    model.train()

    total_loss = 0.0


    for batch in train_loader:

        batch = {
            name: tensor.to(device)
            for name, tensor
            in batch.items()
        }

        optimizer.zero_grad()

        output = model(
            **batch
        )

        output.loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += float(
            output.loss.item()
        )


    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: "
        f"{total_loss / len(train_loader):.4f}"
    )


# --------------------------------------------------
# 8. Test
# --------------------------------------------------

model.eval()

actual = []

predicted = []


with torch.no_grad():

    for batch in test_loader:

        labels = batch[
            "labels"
        ].tolist()

        model_batch = {
            name: tensor.to(device)
            for name, tensor
            in batch.items()
        }

        output = model(
            **model_batch
        )

        predictions = torch.argmax(
            output.logits,
            dim=1
        ).cpu().tolist()

        actual.extend(
            labels
        )

        predicted.extend(
            predictions
        )


accuracy = accuracy_score(
    actual,
    predicted
)


print("\nClassification report:")

print(
    classification_report(
        actual,
        predicted,
        labels=[
            0,
            1,
            2,
            3
        ],
        target_names=[
            LABEL_NAMES[index]
            for index in range(4)
        ],
        zero_division=0
    )
)


# --------------------------------------------------
# 9. Per-class metrics
# --------------------------------------------------

precision, recall, f1, support = (
    precision_recall_fscore_support(
        actual,
        predicted,
        labels=[
            0,
            1,
            2,
            3
        ],
        zero_division=0
    )
)


result_rows = []


for label_index in range(4):

    result_rows.append({
        "Category":
            LABEL_NAMES[
                label_index
            ],

        "Precision":
            round(
                float(
                    precision[
                        label_index
                    ]
                ),
                4
            ),

        "Recall":
            round(
                float(
                    recall[
                        label_index
                    ]
                ),
                4
            ),

        "F1":
            round(
                float(
                    f1[
                        label_index
                    ]
                ),
                4
            ),

        "Support":
            int(
                support[
                    label_index
                ]
            ),

        "Overall Accuracy":
            round(
                accuracy,
                4
            )
    })


pd.DataFrame(
    result_rows
).to_csv(
    RESULTS_PATH,
    index=False
)


# --------------------------------------------------
# 10. Save model
# --------------------------------------------------

model.save_pretrained(
    OUTPUT_DIRECTORY
)

tokenizer.save_pretrained(
    OUTPUT_DIRECTORY
)


print("\nGenerated files:")

print("Ticket model:", OUTPUT_DIRECTORY)

print("Results     :", RESULTS_PATH)