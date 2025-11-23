#!/usr/bin/env python3
"""
Example usages for the remote-sensing PCA visualization script.

Demonstrates multiple entry points exposed by remote_sensing_pca_visualization.py.
"""

import os
import sys
from pathlib import Path

import torch

# Add project root to PYTHONPATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from remote_sensing_pca_visualization import RemoteSensingPCAVisualizer


def example_basic_usage():
    """Basic usage example."""
    print("=== Basic Usage ===")

    # Instantiate with default settings
    visualizer = RemoteSensingPCAVisualizer()

    # Example placeholder for an image path
    # image_path = "path/to/your/satellite_image.jpg"

    # Basic visualization
    # result = visualizer.visualize(image_path)
    print("Basic call: visualizer.visualize('image.jpg')")


def example_advanced_usage():
    """Advanced usage example."""
    print("\n=== Advanced Usage ===")

    # Switch to the ViT-7B backbone
    visualizer_large = RemoteSensingPCAVisualizer(model_name="vit7b16")

    # Provide foreground points
    points = [(100, 200), (300, 400), (500, 600)]

    # Advanced visualization call
    # result = visualizer_large.visualize(
    #     image_path="satellite_image.jpg",
    #     output_path="output_advanced.png",
    #     foreground_method="points",
    #     points=points,
    #     show_plot=True
    # )

    print("Advanced usage snippet:")
    print("""
    visualizer = RemoteSensingPCAVisualizer(model_name="vit7b16")
    points = [(100, 200), (300, 400)]
    result = visualizer.visualize(
        image_path="satellite_image.jpg",
        output_path="output.png",
        foreground_method="points",
        points=points
    )
    """)


def example_batch_processing():
    """Batch processing example."""
    print("\n=== Batch Processing ===")

    visualizer = RemoteSensingPCAVisualizer()

    # Pseudo-code for looping over a directory
    # image_dir = "path/to/satellite_images/"
    # output_dir = "path/to/output/"

    # for image_file in os.listdir(image_dir):
    #     if image_file.endswith(('.jpg', '.png', '.tif')):
    #         input_path = os.path.join(image_dir, image_file)
    #         output_path = os.path.join(output_dir, f"vis_{image_file}")
    #
    #         try:
    #             visualizer.visualize(input_path, output_path, show_plot=False)
    #             print(f"Finished: {image_file}")
    #         except Exception as e:
    #             print(f"Failed {image_file}: {e}")

    print("Batch processing snippet:")
    print("""
    for image_file in os.listdir('satellite_images/'):
        if image_file.endswith(('.jpg', '.png')):
            input_path = f'satellite_images/{image_file}'
            output_path = f'output/vis_{image_file}'
            visualizer.visualize(input_path, output_path, show_plot=False)
    """)


def example_custom_foreground_detection():
    """Custom foreground detector example."""
    print("\n=== Custom Foreground Detection ===")

    class CustomRemoteSensingVisualizer(RemoteSensingPCAVisualizer):
        def detect_foreground(self, image, method="custom", threshold=0.5, **kwargs):
            """Custom detection logic."""
            import numpy as np
            from scipy import ndimage

            # PIL image -> numpy array
            image_array = np.array(image)

            # Example: NDVI-like heuristic (adapt as needed for your data)
            if len(image_array.shape) == 3:
                # Assume RGB
                r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]

                # Simplified vegetation mask
                vegetation_mask = (g > r) & (g > b) & (g > threshold * 255)

                # Morphological cleanup
                vegetation_mask = ndimage.binary_opening(vegetation_mask, iterations=2)
                vegetation_mask = ndimage.binary_closing(vegetation_mask, iterations=2)

                return torch.from_numpy(vegetation_mask.astype(float))

            # Default to full-foreground mask
            return torch.ones(image.size[1], image.size[0])

    # custom_visualizer = CustomRemoteSensingVisualizer()
    # custom_visualizer = CustomRemoteSensingVisualizer()
    # result = custom_visualizer.visualize("satellite_image.jpg")

    print("Custom detector snippet:")
    print("""
    class CustomVisualizer(RemoteSensingPCAVisualizer):
        def detect_foreground(self, image, method="custom", **kwargs):
            # Implement your detection logic here
            # e.g., NDVI-based vegetation detection
            pass
    """)


def show_command_line_examples():
    """Command-line examples."""
    print("\n=== Command-line Examples ===")

    examples = [
        "# Basic usage",
        "python remote_sensing_pca_visualization.py satellite_image.jpg",

        "# Custom output path",
        "python remote_sensing_pca_visualization.py image.jpg -o result.png",

        "# Use ViT-7B",
        "python remote_sensing_pca_visualization.py image.jpg -m vit7b16",

        "# Provide foreground points",
        "python remote_sensing_pca_visualization.py image.jpg -p 100 200 300 400",

        "# Run on CPU",
        "python remote_sensing_pca_visualization.py image.jpg -d cpu",

        "# Save only",
        "python remote_sensing_pca_visualization.py image.jpg --no-plot",
    ]

    for example in examples:
        print(f"  {example}")


def main():
    """Entry point."""
    print("Remote-sensing DINOv3 PCA Visualizer - Examples")
    print("=" * 50)

    # Display sample usages
    example_basic_usage()
    example_advanced_usage()
    example_batch_processing()
    example_custom_foreground_detection()
    show_command_line_examples()

    print("\n" + "=" * 50)
    print("💡 Tips:")
    print("1. Ensure checkpoints exist under pretrained_weights/.")
    print("2. Tune foreground parameters for best results.")
    print("3. Use ViT-7B for higher accuracy (requires more resources).")
    print("4. Prefer --no-plot for large batch jobs.")


if __name__ == "__main__":
    main()
