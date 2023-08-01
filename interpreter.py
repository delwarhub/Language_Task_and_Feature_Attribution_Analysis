import pandas as pd
import torch
import shap
import transformers
from transformers import pipeline

class shap_interpretability:
    def __init__(self, model: transformers.AutoModelForSequenceClassification,
                 data: pd.DataFrame,
                 dataset, config,):
        super(shap_interpretability, self).__init__()

        self.model = model
        self.dataset = dataset
        self.tokenizer = dataset.tokenizer
        self.data = data
        self.labels = dataset.class_labels
        self.config = config
        self.sample_size= self.config["SAMPLE_SIZE"]
        self.source_column = config["SOURCE_COLUMN"]
        self.target_column = config["TARGET_COLUMN"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clf_pipeline = pipeline("text-classification", self.model, tokenizer=self.tokenizer, device=0, top_k=None)

        self.explainer = shap.Explainer(self.clf_pipeline) # initialize shap-explainer

        # compute shapley-values for text instances in the self.data
        self.sample_data = self.data.sample(n=self.sample_size)
        self.shap_values = self.explainer(self.sample_data[self.source_column])

    def plot_top_word_impact_class_k(self, class_label: str):
        """ visualize the top-words impacting a specific class """

        label_id = self.labels.index(class_label)
        plt = shap.plots.bar(self.shap_values[:, :, label_id].mean(0))
        return plt

    def plot_impact_on_single_class_k(self, class_label: str, instance_idx: int):
        """ visualize the impact on single-class """

        label_id = self.labels.index(class_label)
        plt = shap.plots.text(self.shap_values[instance_idx, :, label_id])
        return plt
    
    def plot_interpretation(self):
        
        def predict_labels(text):
            output = self.clf_pipeline(text)
            return self.dataset.class_labels.index(output[0][0]["label"])

        self.sample_data["pred_label"] = self.sample_data.apply(lambda row: predict_labels(row[self.source_column]), axis=1)

        # positve example
        # 1. sports & gaming
        sample = self.sample_data[self.sample_data["pred_label"] == self.dataset.class_labels.index("sports_&_gaming")].sample(1)
        shap_values = self.clf_pipeline(sample["text"])
        shap.plots.text(shap_values[:, :, self.labels.index("sports_&_gaming")])

        # 2. arts & culture
        sample = self.sample_data[self.sample_data["pred_label"] == self.dataset.class_labels.index("arts_&_culture")].sample(1)
        shap_values = self.clf_pipeline(sample["text"])
        shap.plots.text(shap_values[:, :, self.labels.index("arts_&_culture")])

        # negative example
        # 1. arts & culture
        sample = self.sample_data[(self.sample_data["pred_label"] == self.dataset.class_labels.index("daily_life")) & (self.sample_data[self.target_column] == self.dataset.class_labels.index("arts_&_culture"))].sample(1)
        shap_values = self.clf_pipeline(sample["text"])
        shap.plots.text(shap_values)

        # 2. science & technology
        sample = self.sample_data[(self.sample_data["pred_label"] == self.dataset.class_labels.index("business_&_entrepreneurs")) & (self.sample_data[self.target_column] == self.dataset.class_labels.index("science_&_technology"))].sample(1)
        shap_values = self.clf_pipeline(sample["text"])
        shap.plots.text(shap_values)

        text1 = "This book provides the perfect quick read…” Don’t miss Truth and Grace Homeschool Academy’s review of Ghosted at the Altar by {{USERNAME}} #book & enter the #free #giveaway for a $25 Amazon gift card! #books #amreading {{URL}}" # actual text
        text2 = "This book provides the perfect quick read…” Don’t miss Truth and Grace Homeschool Academy’s review of Ghosted at the Altar by {{USERNAME}} #book & enter the #books #amreading {{URL}}" # removing add related hastags
        text3 = "This book provides the perfect quick read…” Don’t miss Truth and Grace Homeschool Academy’s review of Ghosted at the Altar by {{USERNAME}}" # discard all the hastags
        shap_values = self.clf_pipeline([text1, text2, text3])
        shap.plots.text(shap_values)
        
