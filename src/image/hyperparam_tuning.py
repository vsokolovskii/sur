import optuna
from functools import partial
import torch

from cnn import CNN, PngsDataset, train, validate, make


# Define objective function for Optuna
def objective(trial, config):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.7)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])

    # Update config with suggested hyperparameters
    config.update({
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'dropout': dropout,
        'batch_size': batch_size
    })
    
    # Create model, device, train_loader, val_loader, and optimizer
    model, device, train_loader, val_loader, optimizer = make(config)
    validations = []
    # Train and validate model
    for epoch in range(1, config['epochs'] + 1):
        train(model, device, train_loader, optimizer, epoch, config['wandb_mode'])
        val_loss, val_acc = validate(model, device, val_loader, epoch, config['wandb_mode'])
        validations.append(val_loss)

    # calculate average validation loss of last 50 epochs
    avg_loss = sum(validations[-50:]) / 50

    # Return the validation loss as the objective for Optuna to minimize
    return avg_loss


def main():
    # Load your config here
    config = dict(
    epochs=2000,
    batch_size=4,
    learning_rate=1e-3,
    weight_decay=1e-2,
    dropout=0.4,
    wandb_run_desc="deeper_cnn",
    train_data_desc_file="../pngs-train.csv",
    test_data_desc_file="../pngs-dev.csv",
    optimizer=torch.optim.RMSprop,
    optimizer_name='RMS',
    momentum=0,
    wandb_mode='disabled')

    # Create a study and optimize the objective
    study = optuna.create_study(direction='minimize')
    study.optimize(partial(objective, config=config), n_trials=50)

    # Print the best hyperparameters found
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == '__main__':
    main()
