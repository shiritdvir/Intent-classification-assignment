import os
import matplotlib.pyplot as plt
import plotly.express as px
import config


def plot_ratios(data, labels, save_path=config.results_path):
    fig = px.imshow(data,
                    text_auto=True,
                    color_continuous_scale='Sunset',
                    labels=dict(color="Ratio"),
                    x=labels,
                    y=labels
                    )
    fig.update_xaxes(side="top")
    fig.write_html(os.path.join(save_path, 'labels_ratios.html'))


def plot_losses(log_history, save_path=config.results_path):
    train_losses = [entry['loss'] for entry in log_history if 'loss' in entry]
    eval_losses = [entry['eval_loss'] for entry in log_history if 'eval_loss' in entry]
    train_steps = [entry['step'] for entry in log_history if 'loss' in entry]
    eval_steps = [entry['step'] for entry in log_history if 'eval_loss' in entry]
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_losses, label='Training Loss')
    plt.plot(eval_steps, eval_losses, label='Evaluation Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'losses.png'))