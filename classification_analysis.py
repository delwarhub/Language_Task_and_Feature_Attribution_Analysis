import os
import gc
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F
from utils import check_and_create_directory
from classification_dataset import classification_dataset
from models import DebertaV2ForSequenceClassification

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

class classification_analysis:
    def __init__(self, config: dict,
                 path_to_saved_model_dir: None):

        self.config = config
        self.path_to_saved_model_dir = path_to_saved_model_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.source_column = config["SOURCE_COLUMN"]
        self.target_column = config["TARGET_COLUMN"]
        self.path_to_result_output_dir = config["PATH_TO_RESULT_OUTPUT_DIR"]
        self.dataset_label = config["DATASET_LABEL"].split("/")[-1]
        self.model_checkpoint = config["MODEL_CHECKPOINT"]

        self.dataset = classification_dataset(self.config)
        self.model = DebertaV2ForSequenceClassification.from_pretrained(self.path_to_saved_model_dir, num_labels=len(self.dataset.class_labels))
        self.model.to(self.device)
        self.test_df = self.dataset.retrieve_dataframe("test")
        self.data_loader = self.dataset.set_up_data_loader("test")

         # update model configurations
        class_labels = self.dataset.class_labels
        label2id = dict(zip(class_labels, range(len(class_labels))))
        id2label = dict((idx, label) for label, idx in label2id.items())
        self.model.config.label2id = label2id
        self.model.config.id2label = id2label

    def prediction_procedure(self):
        """ prediction procedure, & evaluation """
        
        predictions = []
        ground_truth = []
        for batch in tqdm(self.data_loader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, attention_mask, labels = batch
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions.extend(F.softmax(outputs.logits, dim=-1).argmax(dim=-1).detach().cpu().tolist())
            ground_truth.extend(labels.tolist())
            del batch
            del input_ids
            del attention_mask
            del labels
            del outputs
            gc.collect()    

        report = classification_report(ground_truth, predictions, target_names=self.dataset.class_labels, zero_division=0)
        print(report)    

    def plot_barchart(self):
        """ plot bar-chart """

        train_df = self.dataset.retrieve_dataframe("train")
        label_counts = train_df[self.target_column].value_counts().to_dict()
        label_counts = {self.dataset.class_labels[key]: value for key, value in label_counts.items()}
        labels = list(label_counts.keys())
        counts = list(label_counts.values())
        label_2_count_df = pd.DataFrame({"label": labels, "count": counts})
        sns.barplot(data=label_2_count_df, x="count", y="label")
        plt.xticks(rotation=90)
        plt.tight_layout()

        check_and_create_directory(os.path.join(self.path_to_result_output_dir, "./barplot/"))
        plt.savefig(os.path.join(self.path_to_result_output_dir + "./barplot/" + self.model_checkpoint.split("/")[-1] + "-finetuned-on-" + self.dataset_label + '_barplot.jpg'), dpi=300, bbox_inches='tight')
        print("saved bar-chart @ ", os.path.join(self.path_to_result_output_dir + "./barplot/" + self.model_checkpoint.split("/")[-1] + "-finetuned-on-" + self.dataset_label + '_barplot.jpg'))
        plt.show()

    def generate_confusion_matrix(self):
        """ plot confusion-matrix """

        predictions = []
        ground_truth = []
        for batch in tqdm(self.data_loader):
            input_ids, attention_mask, labels = batch
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = F.softmax(outputs.logits, dim=-1)
            batch_predictions = probabilities.argmax(dim=-1).to("cpu").tolist()
            predictions.extend(batch_predictions)
            ground_truth.extend(labels.to("cpu").tolist())

            del batch
            del input_ids
            del attention_mask
            del labels
            del outputs
            del probabilities
            del batch_predictions
            gc.collect()

        true_predictions = [self.dataset.class_labels[pred] for pred in predictions]
        true_labels = [self.dataset.class_labels[gt] for gt in ground_truth]

        cm = confusion_matrix(true_predictions, true_labels, labels=self.dataset.class_labels) # generate confusion_matrix

        sns.set()
        sns.heatmap(cm, annot=True, cmap='PuRd', fmt='d', xticklabels=self.dataset.class_labels, yticklabels=self.dataset.class_labels)

        # set plot labels
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # save heatmap
        check_and_create_directory(os.path.join(self.path_to_result_output_dir, "./confusion_matrix/"))
        plt.savefig(os.path.join(self.path_to_result_output_dir + "./confusion_matrix/" + self.model_checkpoint.split("/")[-1] + "-finetuned-on-" + self.dataset_label + '_confusion_matrix.jpg'), dpi=300, bbox_inches='tight')
        print("saved confusion matrix @ ", os.path.join(self.path_to_result_output_dir + "./confusion_matrix/" + self.model_checkpoint.split("/")[-1] + "-finetuned-on-" + self.dataset_label + '_confusion_matrix.jpg'))
        plt.show()