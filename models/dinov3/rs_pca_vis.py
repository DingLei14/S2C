#!/usr/bin/env python3
"""
DINOv3 PCA rainbow visualization utility for remote sensing imagery.

Built on top of the DINOv3 SAT-493M pretrained model to reproduce the PCA-style
visualizations from pca.ipynb. Includes optional foreground detection and
point-based interaction helpers.

Author: AI Assistant
Date: 2025-09-19
"""

import argparse
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA
from scipy import signal, ndimage
import cv2

# Add project root to the import path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from dinov3.hub.backbones import dinov3_vitl16, dinov3_vit7b16
except ImportError as e:
    print(f"Failed to import DINOv3 modules: {e}")
    print("Please ensure the DINOv3 project is installed or PYTHONPATH is configured.")
    sys.exit(1)


class RemoteSensingPCAVisualizer:
    """DINOv3 PCA visualizer for remote sensing images."""

    def __init__(self, model_name="vitl16", device="cuda"):
        """
        Initialize the visualizer.

        Args:
            model_name: DINOv3 backbone ('vitl16' or 'vit7b16').
            device: Torch device ('cuda' or 'cpu').
        """
        self.model_name = model_name
        self.device = device
        self.patch_size = 16

        # normalization parameters for satellite imagery (SAT-493M)
        self.SATELLITE_MEAN = (0.430, 0.411, 0.296)
        self.SATELLITE_STD = (0.213, 0.156, 0.143)

        # number of layers per model variant
        self.MODEL_TO_NUM_LAYERS = {
            "vits16": 12,
            "vits16plus": 12,
            "vitb16": 12,
            "vitl16": 24,
            "vith16plus": 32,
            "vit7b16": 40,
        }

        # load backbone
        self._load_model()

    def _load_model(self):
        """Load the requested DINOv3 backbone."""
        print(f"Loading DINOv3 {self.model_name} backbone...")

        # resolve pretrained checkpoint path
        weights_dir = project_root / "pretrained_weights" 
        if self.model_name == "vitl16":
            weights_file = "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
        elif self.model_name == "vit7b16":
            weights_file = "dinov3_vit7b16_pretrain_sat493m-a6675841.pth"
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        weights_path = weights_dir / weights_file

        if not weights_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

        # load backbone
        if self.model_name == "vitl16":
            self.model = dinov3_vitl16(weights=str(weights_path))
        elif self.model_name == "vit7b16":
            self.model = dinov3_vit7b16(weights=str(weights_path))

        self.model.to(self.device)
        self.model.eval()

        print(f"Model ready on device: {self.device}")

    def load_image(self, image_path):
        """
        Load and preprocess the input image.

        Args:
            image_path: path to the input image.

        Returns:
            Tuple of (preprocessed tensor, resized PIL image, original PIL image).
        """
        # read image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")

        # resize so that both sides are multiples of patch size
        w, h = image.size
        h_patches = int(768 / self.patch_size)  # keep a fixed number of patches vertically
        w_patches = int((w * 768) / (h * self.patch_size))

        new_h = h_patches * self.patch_size
        new_w = w_patches * self.patch_size

        # resize
        image_resized = TF.resize(image, (new_h, new_w))

        # convert to tensor and normalize
        image_tensor = TF.to_tensor(image_resized)
        image_tensor = TF.normalize(image_tensor, mean=self.SATELLITE_MEAN, std=self.SATELLITE_STD)

        return image_tensor, image_resized, image

    def extract_features(self, image_tensor):
        """
        Extract patch features from the image tensor.

        Args:
            image_tensor: preprocessed image tensor.

        Returns:
            Patch features tensor.
        """
        n_layers = self.MODEL_TO_NUM_LAYERS[self.model_name]

        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.float32):
                feats = self.model.get_intermediate_layers(
                    image_tensor.unsqueeze(0).to(self.device),
                    n=range(n_layers),
                    reshape=True,
                    norm=True
                )

                # use the last layer representation
                x = feats[-1].squeeze().detach().cpu()
                dim = x.shape[0]

                # reshape to [H*W, D]
                x = x.view(dim, -1).permute(1, 0)

        return x

    def detect_foreground(self, image, method="auto", threshold=0.5, points=None):
        """
        Detect a foreground mask.

        Args:
            image: PIL image.
            method: detection strategy ('auto', 'manual', 'points', 'none').
            threshold: threshold for the auto/manual branches.
            points: optional list of (x, y) coordinates when method == 'points'.

        Returns:
            Foreground mask tensor.
        """
        if method == "none":
            # no foreground detection: use the full image
            return torch.ones(image.size[1], image.size[0])

        elif method == "manual":
            # manually approximate foreground via color heuristics
            image_array = np.array(image)

            # hsv separation helps differentiate foreground/background
            hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)

            # threshold by brightness & saturation (tweakable per dataset)
            brightness = hsv[:, :, 2]
            saturation = hsv[:, :, 1]

            # adaptive thresholds
            brightness_thresh = np.percentile(brightness, 70)
            saturation_thresh = np.percentile(saturation, 30)

            foreground_mask = (brightness > brightness_thresh) & (saturation > saturation_thresh)

        elif method == "points" and points is not None:
            # grow small regions around user-specified points
            h, w = image.size[1], image.size[0]
            foreground_mask = np.zeros((h, w), dtype=bool)

            # build a square region around each point
            for point in points:
                x, y = point
                y_min = max(0, y - 50)
                y_max = min(h, y + 50)
                x_min = max(0, x - 50)
                x_max = min(w, x + 50)
                foreground_mask[y_min:y_max, x_min:x_max] = True

        else:  # method == "auto"
            # simple automatic edge-based detector
            image_array = np.array(image)

            # edges -> dilation -> mask
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)

            kernel = np.ones((5, 5), np.uint8)
            foreground_mask = cv2.dilate(edges, kernel, iterations=2).astype(bool)

        # median filter to smooth the mask
        foreground_mask = signal.medfilt2d(foreground_mask.astype(float), kernel_size=5) > threshold

        return torch.from_numpy(foreground_mask.astype(float))

    def create_pca_visualization(self, features, foreground_mask, image_size):
        """
        Build the PCA rainbow visualization.

        Args:
            features: patch features shaped [H*W, D].
            foreground_mask: image-level mask [H, W].
            image_size: (H, W) tuple.

        Returns:
            Tensor representing the visualization.
        """
        h_patches, w_patches = image_size[0] // self.patch_size, image_size[1] // self.patch_size

        # downsample the image-level mask to the patch grid via avg pooling
        from torch.nn.functional import avg_pool2d
        patch_mask = avg_pool2d(
            foreground_mask.unsqueeze(0).unsqueeze(0),
            kernel_size=self.patch_size,
            stride=self.patch_size
        ).squeeze()

        # flatten mask to pick patch features
        foreground_selection = patch_mask.view(-1) > 0.5

        # foreground patches only
        fg_patches = features[foreground_selection]

        if len(fg_patches) == 0:
            print("Warning: no foreground detected, falling back to all patches.")
            fg_patches = features

        # PCA
        print(f"Running PCA on {len(fg_patches)} patch tokens...")
        pca = PCA(n_components=3, whiten=True)
        pca.fit(fg_patches.numpy())

        # apply PCA to every patch
        projected_image = torch.from_numpy(
            pca.transform(features.numpy())
        ).view(h_patches, w_patches, 3)

        # sigmoid for vivid colors
        projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)

        # optionally mask background patches (skip if mask is all ones)
        if not torch.all(patch_mask == 1.0):
            projected_image *= (patch_mask.unsqueeze(0) > 0.5)

        return projected_image

    def visualize(self, image_path, output_path=None, foreground_method="auto",
                  foreground_threshold=0.5, points=None, show_plot=True):
        """
        Run the full visualization pipeline.

        Args:
            image_path: path to the input image.
            output_path: optional output path.
            foreground_method: foreground detection strategy.
            foreground_threshold: threshold for the detector.
            points: optional point list for the 'points' mode.
            show_plot: whether to display matplotlib figures.

        Returns:
            Output path as a string.
        """
        print(f"Processing image: {image_path}")

        # 1. load & preprocess
        image_tensor, image_resized, original_image = self.load_image(image_path)
        print(f"Image sizes: original={original_image.size}, resized={image_resized.size}")

        # 2. feature extraction
        features = self.extract_features(image_tensor)
        print(f"Feature tensor shape: {features.shape}")

        # 3. foreground detection
        foreground_mask = self.detect_foreground(
            image_resized,
            method=foreground_method,
            threshold=foreground_threshold,
            points=points
        )
        print(f"Foreground coverage: {foreground_mask.mean().item():.2%}")

        # 4. PCA visualization
        pca_visualization = self.create_pca_visualization(
            features,
            foreground_mask,
            image_resized.size[::-1]  # (H, W)
        )

        # 5. save/show
        if output_path is None:
            output_path = Path(image_path).stem + "_pca_visualization.png"

        if show_plot:
            plt.figure(figsize=(12, 5), dpi=150)

            # original image
            plt.subplot(1, 3, 1)
            plt.imshow(original_image)
            plt.title("Original Image", fontsize=12)
            plt.axis('off')

            # foreground mask
            plt.subplot(1, 3, 2)
            plt.imshow(foreground_mask.numpy(), cmap='gray')
            plt.title("Foreground Mask", fontsize=12)
            plt.axis('off')

            # PCA visualization
            plt.subplot(1, 3, 3)
            plt.imshow(pca_visualization.permute(1, 2, 0).numpy())
            plt.title("PCA Visualization", fontsize=12)
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.show()
        else:
            # save PCA result only
            pca_image = pca_visualization.permute(1, 2, 0).numpy()
            pca_image = (pca_image * 255).astype(np.uint8)
            Image.fromarray(pca_image).save(output_path)

        print(f"PCA visualization saved to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Remote sensing DINOv3 PCA visualization")
    parser.add_argument("--image_path", default="0.jpg", help="Path to the input image")
    parser.add_argument("-o", "--output", help="Destination path for the visualization")
    parser.add_argument("-m", "--model", choices=["vitl16", "vit7b16"],
                       default="vitl16", help="Backbone to use")
    parser.add_argument("-d", "--device", choices=["cuda", "cpu"],
                       default="cuda", help="Device to run on")
    parser.add_argument("--enable-foreground", action="store_true",
                       help="Enable foreground detection (disabled uses the full image)")
    parser.add_argument("-f", "--foreground_method", choices=["auto", "manual", "points", "none"],
                       default="auto", help="Foreground detection strategy")
    parser.add_argument("-t", "--threshold", type=float, default=0.5,
                       help="Threshold for the auto/manual detectors")
    parser.add_argument("-p", "--points", nargs='+', type=int,
                       default=[524, 686], help="Point coordinates (x1 y1 x2 y2 ...)")
    parser.add_argument("--no-plot", action="store_true",
                       help="Skip matplotlib display and save only")

    args = parser.parse_args()

    # convert point list
    points = None
    if args.points:
        if len(args.points) % 2 != 0:
            print("Error: point coordinates must be pairs (x y).")
            return
        points = [(args.points[i], args.points[i+1])
                 for i in range(0, len(args.points), 2)]

    # derive foreground settings
    if not args.enable_foreground:
        args.foreground_method = "none"
    else:
        if points and len(points) == 1 and points[0] == (524, 686):
            args.foreground_method = "points"

    try:
        visualizer = RemoteSensingPCAVisualizer(
            model_name=args.model,
            device=args.device
        )

        output_path = visualizer.visualize(
            image_path=args.image_path,
            output_path=args.output,
            foreground_method=args.foreground_method,
            foreground_threshold=args.threshold,
            points=points,
            show_plot=not args.no_plot
        )

        print("\n✅ Visualization complete!")
        print(f"📁 Output file: {output_path}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())