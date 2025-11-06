import os
import torch.optim as optim
import torch.nn as nn
from .datasets import get_dataset, get_drawn_data
from torch.utils.data import DataLoader, TensorDataset
from .utils.video import generate_video
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

def train_model(
    model,
    dataset_name: str,
    learning_rate=0.01,
    batch_size=32,
    num_epochs=100,
    num_data_points=200,
    dataset_noise=0.0,
    test_data_fraction=0.3,
    visualize_every_nth_step=1,
    video_frames_folder=None,
    save_video_as = None,
    video_fps=60,
    device = "mps:0",
    ylim = 1.1,
) -> float:
    """
    Trains a model on the given dataset.

    Parameters:
    - model: PyTorch model to be trained
    - dataset: Dictionary containing training and test data (keys: 'train' and 'test')
    - learning_rate: Learning rate for the optimizer
    - batch_size: Batch size for training
    - num_epochs: Number of epochs to train
    - visualize_every_nth_step: Frequency of visualization
    - video_frames_folder: Directory to save visualizations (if None, visualizations are not saved)
    - save_video_as: Path to save the video (if None, video is not saved)
    - video_fps: Frames per second for the video

    Returns:
    - Final test loss (float)
    """
    if video_frames_folder is not None:
        os.system(f"rm -rf {video_frames_folder} && mkdir -p {video_frames_folder}")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    if dataset_name != "drawn":
        dataset = get_dataset(
            name=dataset_name,
            num_points=num_data_points,
            test_data_fraction=test_data_fraction,
            noise=dataset_noise
        )
    else:
        dataset = get_drawn_data(
            test_data_fraction=test_data_fraction
        )
    
    train_x, train_y = dataset["train"]["x"].to(device), dataset["train"]["y"].to(device)
    test_x, test_y = dataset["test"]["x"].to(device), dataset["test"]["y"].to(device)

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    step = 0
    video_frame_filenames = []
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            if video_frames_folder is not None and step % visualize_every_nth_step == 0:
                with torch.no_grad():
                    test_preds = model(test_x)
                    fig = plt.figure(figsize=(8, 6))
                    plt.scatter(train_x.cpu().numpy(), train_y.cpu().numpy(), label="Train data", color="gray", alpha=0.3)
                    plt.scatter(test_x.cpu().numpy(), test_y.cpu().numpy(), label="Test data", color="red", linewidth=2)
                    # Sort test_x and test_preds for line plot
                    sorted_indices = torch.argsort(test_x.cpu(), dim=0)
                    sorted_x = test_x.cpu()[sorted_indices].squeeze()
                    sorted_preds = test_preds.cpu()[sorted_indices].squeeze()
                    
                    # Plot scatter points
                    plt.scatter(test_x.cpu().numpy(), test_preds.cpu().numpy(), label="Predicted", color="blue", linewidth=2)
                    
                    # Plot line with low alpha
                    plt.plot(sorted_x.numpy(), sorted_preds.numpy(), color="blue", alpha=0.2)
                    plt.xlabel("Input")
                    plt.ylabel("Output")

                    if "train_without_noise" in dataset:
                        plt.scatter(
                            dataset["train_without_noise"]["x"].cpu().numpy(), 
                            dataset["train_without_noise"]["y"].cpu().numpy(), 
                            label="Train data (without noise)", color="green", alpha=0.3
                        )
                    plt.legend()

                    # plt.tight_layout()
                    ## remove upper and right spine
                    plt.gca().spines['top'].set_visible(False)
                    plt.gca().spines['right'].set_visible(False)
                    plt.title(f"Epoch {step}: Loss = {loss.item():.4f}")
                    
                    if ylim is not None:
                        plt.ylim(-ylim, ylim)
                    filename = os.path.join(video_frames_folder, f"epoch_{step:04d}.png")
                    fig.savefig(filename)
                    plt.close()
                    video_frame_filenames.append(filename)
            
            step += 1
    
    # Compute final test loss
    with torch.no_grad():
        final_test_preds = model(test_x)
        final_test_loss = criterion(final_test_preds, test_y).item()

    if video_frames_folder is not None and save_video_as is not None:
        generate_video(
            list_of_pil_images=[Image.open(filename) for filename in video_frame_filenames],
            framerate=video_fps,
            filename=save_video_as
        )
    
    return final_test_loss