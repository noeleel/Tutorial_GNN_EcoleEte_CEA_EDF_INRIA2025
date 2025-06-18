import torch
import matplotlib.pyplot as plt

def plot_sequential_intermediate_outputs(model, input_tensor, max_filters=64):
    """
    Pass input through each layer of a Sequential model and plot intermediate outputs.

    Args:
        model: PyTorch Sequential model
        input_tensor: input tensor with batch dimension (e.g. shape [1, C, H, W])
        max_filters: max number of feature maps to plot per layer
    """
    x = input_tensor
    for i, layer in enumerate(model):
        x = layer(x)
        print(f"Layer {i} ({layer.__class__.__name__}) output shape: {x.shape}")

        # Remove batch dimension
        feature_maps = x.squeeze(0)

        # If output is 1D (e.g. from Linear), reshape to (features, 1, 1) to plot as heatmap
        if feature_maps.dim() == 1:
            feature_maps = feature_maps.unsqueeze(-1).unsqueeze(-1)

        num_filters = feature_maps.shape[0] if feature_maps.dim() == 3 else 1

        plt.figure(figsize=(15, 15))
        plt.suptitle(f"Layer {i}: {layer.__class__.__name__} output")

        for f in range(min(num_filters, max_filters)):
            plt.subplot(8, 8, f + 1)

            # Plot each feature map
            if feature_maps.dim() == 3:
                # Shape: [Channels, Height, Width]
                plt.imshow(feature_maps[f].cpu(), cmap='viridis')
            elif feature_maps.dim() == 2:
                plt.imshow(feature_maps.cpu(), cmap='viridis')
            else:
                # Flatten and plot as 1xN heatmap
                plt.imshow(feature_maps.view(1, -1).cpu(), cmap='viridis')

            plt.axis('off')

        plt.show()

# Example usage:
# model = torch.nn.Sequential(...)
# input_tensor = torch.randn(1, 3, 64, 64)  # example input with batch dim
# plot_sequential_intermediate_outputs(model, input_tensor)