from utils import config
from trainer import classification_trainer
from classification_analysis import classification_analysis
from interpreter import shap_interpretability

if __name__ == "__main__":
    
    # initialize trainer 
    trainer = classification_trainer(config)

    if config.train:
        # train & evaluate model
        trainer.train()
    else:
        # analysis
        clf_analysis = classification_analysis(config=config, path_to_saved_model_dir="./models/deberta-v3-base_epoch_10_18-06-2023-1455/")    
        clf_analysis.prediction_procedure() # print classification-report
        clf_analysis.plot_barchart() # plot bar-chart
        clf_analysis.generate_confusion_matrix() # plot confusion-graph
        
        # shap-interpretability
        model = clf_analysis.model
        tokenizer = clf_analysis.dataset.tokenizer
        test_data = clf_analysis.test_df
        labels = clf_analysis.dataset.class_labels

        interpreter = shap_interpretability(model=model, 
                                            tokenizer=tokenizer,
                                            data=test_data,
                                            labels=labels,
                                            config=config)
             
