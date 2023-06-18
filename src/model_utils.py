import torch
from transformers import TrainingArguments, Trainer
import config
import numpy as np
from src.plot_utils import plot_losses
from sklearn.metrics import classification_report


class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


def train_model(model, train_set, eval_set):
    training_args = TrainingArguments(
        output_dir=config.results_path,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        logging_dir=config.logs_path,
        logging_steps=config.steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )
    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
    )
    trainer.train()
    trainer.save_model()
    plot_losses(trainer.state.log_history)
    return trainer, model


def evaluate_dataset(dataset, trainer):
    predictions = trainer.predict(dataset)
    predicted_labels = predictions.label_ids
    true_labels = np.array(dataset['labels'])
    report = classification_report(true_labels, predicted_labels, output_dict=True, zero_division=0)
    print(report)


def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.sigmoid(outputs.logits)
