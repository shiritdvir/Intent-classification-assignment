from transformers import AutoTokenizer, AutoModelForSequenceClassification
import config
from src.data_utils import load_df, analyze_data, convert_df_to_train_eval_datasets, save_labelled_test
from src.model_utils import train_model, evaluate_dataset, predict


if __name__ == '__main__':

    # Load data
    train_df = load_df(config.train_path)
    test_df = load_df(config.test_path)

    # Analyze data
    analyze_data(train_df, config.labels)

    # Convert dfs to datasets
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_dataset, validation_dataset = convert_df_to_train_eval_datasets(train_df, tokenizer)

    # Train model
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_labels)
    model.config.problem_type = 'multi_label_classification'
    trainer, model = train_model(model, train_dataset, validation_dataset)

    # Evaluate model
    evaluate_dataset(train_dataset, trainer)
    evaluate_dataset(validation_dataset, trainer)

    # Inference (Test set)
    predictions = predict(list(test_df['original_text']), tokenizer, model)
    save_labelled_test(test_df, predictions)

