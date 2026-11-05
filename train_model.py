from naive_bayes_model import train_and_save_model


if __name__ == "__main__":
    metadata = train_and_save_model()
    print("Model training complete.")
    print(f"Dataset: {metadata['dataset_name']}")
    print(f"Training samples: {metadata['train_samples']}")
    print(f"Validation accuracy: {metadata['validation_accuracy']:.4f}")
    print(f"Test accuracy: {metadata['test_accuracy']:.4f}")
