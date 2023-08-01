import os
import gc
import pandas as pd
from tqdm import tqdm
import shutil
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from utils import check_and_create_directory
from classification_dataset import classification_dataset, _data_types
from models import DebertaV2ForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support

def _save(model,
          output_dir: str,
          tokenizer=None,
          state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(model, PreTrainedModel):
            if isinstance(unwrap_model(model), PreTrainedModel):
                if state_dict is None:
                    state_dict = model.state_dict()
                unwrap_model(model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                print("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, 'WEIGHTS_NAME'))
        else:
            model.save_pretrained(output_dir, state_dict=state_dict)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        # torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

def save_model(model,
               output_dir: str,
               tokenizer=None,
               state_dict=None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.
        Will only save from the main process.
        """
        _save(model,output_dir, tokenizer=tokenizer, state_dict=state_dict)

class classification_trainer:
    def __init__(self, config: dict):
        super(classification_trainer, self).__init__()

        self.config = config

        self.model_checkpoint = config["MODEL_CHECKPOINT"]
        self.path_to_model_output_dir = config["PATH_TO_MODEL_OUTPUT_DIR"]
        self.path_to_result_output_dir = config["PATH_TO_RESULT_OUTPUT_DIR"]
        self.early_stopping_threshold = float(config["EARLY_STOPPING_THRESHOLD"])
        self.learning_rate = float(config["LEARNING_RATE"])
        self.weight_decay = float(config["WEIGHT_DECAY"])
        self.num_epochs = int(config["EPOCHS"])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # initialize classification_dataset
        self.dataset = classification_dataset(self.config)
        self.train_data_loader = self.dataset.set_up_data_loader("train")
        self.validation_data_loader = self.dataset.set_up_data_loader("validation")
        self.test_data_loader = self.dataset.set_up_data_loader("test")

        # intialize model
        self.model = DebertaV2ForSequenceClassification.from_pretrained(self.model_checkpoint, num_labels=len(self.dataset.class_labels))
        self.model.to(self.device)

    def train(self, **gen_kwargs):

        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        train_losses, val_f1 = [], []
        patience = 1
        epoch_val_loss_tracker = []
        for epoch in range(1, self.num_epochs):
            train_loss = self.train_epoch()
            train_losses.append(train_loss)

            val_loss = self.val_epoch()
            epoch_val_loss_tracker.append(val_loss)

            val_results = self.get_val_scores(data_type="validation", desc="Validation Generation Iteration",
                                              epoch=epoch, **gen_kwargs)
            val_f1.append(val_results["f1"])

            print("Epoch: {:0.2f}\ttrain_loss: {:0.2f}\tval_loss: {:0.2f}\tmin_validation_loss: {:0.2f}".format(
                epoch+1, train_loss, val_loss, min(epoch_val_loss_tracker)))

            print("val_precision: {:0.2f}\tval_recall: {:0.2f}\tval_f1: {:0.2f}\tval_accuracy: {:0.2f}".format(
                val_results["precision"], val_results["recall"],val_results["f1"]))

            path = self.path_to_model_output_dir + f"{self.model_checkpoint.split('/')[-1]}_epoch_" + str(epoch+1) + "_" + datetime.now().strftime("%d-%m-%Y-%H:%M")

            save_model(self.model,
                       path,
                       self.dataset.tokenizer)
            print("Model saved at path: ", path)

            print("---------------------------------------------------------------")

            if val_results["f1"] < max(val_f1):
                patience = patience + 1
                if patience == self.early_stopping_threshold:
                    break
            else:
                patience = 1

            # keep top-3 models
            model_foldernames = os.listdir(self.path_to_model_output_dir)
            model_foldernames = [os.path.join(self.path_to_model_output_dir, foldername) for foldername in model_foldernames]
            if len(model_foldernames) > 3:
                oldest_folderpath = min(model_foldernames, key=os.path.getctime)
                shutil.rmtree(oldest_folderpath)
                print(f"Deleted previously saved model: {oldest_folderpath}")

            del train_loss
            del val_loss
            del path
            gc.collect()
            torch.cuda.empty_cache()

    def train_epoch(self):

        self.model.train()
        epoch_train_loss = 0.0
        pbar = tqdm(self.train_data_loader, desc="Training Iteration")
        for step, batch in enumerate(pbar):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, attention_mask, labels = batch
            self.optimizer.zero_grad()

            outputs = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
            loss = outputs["loss"]
            epoch_train_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            pbar.set_description('train_loss={0:.3f}'.format(loss.item()))

        del batch
        del input_ids
        del attention_mask
        del labels
        del outputs
        del loss
        gc.collect()
        torch.cuda.empty_cache()

        return epoch_train_loss / step

    def val_epoch(self):

        self.model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.validation_data_loader, desc="Validation Loss Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels = batch

                outputs = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)
                loss = outputs["loss"]
                epoch_val_loss += loss.item()

        del batch
        del input_ids
        del attention_mask
        del labels
        del outputs
        del loss
        gc.collect()
        torch.cuda.empty_cache()

        return epoch_val_loss / step

    def test_epoch(self, data_type: _data_types, **gen_kwargs):
        self.model.eval()
        out_predictions = []
        gold = []

        data_loader = self.validation_data_loader if data_type == "validation" else self.test_data_loader

        with torch.no_grad():
            for step, batch in enumerate(tqdm(data_loader, desc="Prediction Procedure")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels = batch

                outputs = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    **gen_kwargs)

                # check for the model outputs {loss, logits, hidden_states, attentions}
                # print(outputs.logits.shape) ideally should be (batch_size x sequence_length x num_classes)

                probabilities = F.softmax(outputs.logits, dim=-1)
                predictions = probabilities.argmax(dim=-1).cpu().tolist()

                out_predictions.extend(predictions)
                gold.extend(labels.cpu().tolist())

        del batch
        del input_ids
        del attention_mask
        del labels
        del outputs
        del probabilities
        del predictions
        gc.collect()
        torch.cuda.empty_cache()

        return out_predictions, gold

    def get_val_scores(self, data_type: _data_types, epoch, desc="Validation Loss Iteration", **gen_kwargs):

        predictions, gold = self.test_epoch(data_type, **gen_kwargs)
        result = self.get_scores((predictions, gold))

        if "Validation" in desc:
            val_df = pd.DataFrame(list(zip(gold, predictions)), columns=["ground_truth", "prediction"])
            file_name = check_and_create_directory(self.path_to_result_output_dir + "val/") + f"./{self.model_checkpoint.split('/')[-1]}_epoch_" + str(epoch+1) + "_val_results.csv"
            val_df.to_csv(file_name, index=False)
            print("Validation File Saved")
        elif "Test" in desc:
            test_df = pd.DataFrame(list(zip(gold, predictions)), columns=["ground_truth", "prediction"])
            file_name = check_and_create_directory(self.path_to_result_output_dir + "test/") + f"./{self.model_checkpoint.split('/')[-1]}_epoch_" + str(epoch+1) + "_test_results.csv"
            test_df.to_csv(file_name, index=False)
            print("Test File Saved")

        del predictions
        del gold
        gc.collect()
        torch.cuda.empty_cache()

        return result

    def get_scores(self, p, full_rep: bool=False):
        true_predictions, true_labels = p

        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_predictions, average='micro')

        if full_rep:
            return {
                "predictions": true_predictions,
                "labels": true_labels,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        else:
            return {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
