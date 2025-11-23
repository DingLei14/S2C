# Remote-Sensing DINOv3 PCA Visualization

Generate the rainbow-style PCA visualizations from `pca.ipynb` using the SAT-493M pretrained DINOv3 checkpoints for remote-sensing imagery.

## Highlights

- ✅ Uses the SAT-493M DINOv3 weights tailored for satellite data
- ✅ Supports both ViT-L and ViT-7B variants
- ✅ Provides automatic foreground detection plus manual point selection
- ✅ Produces vibrant PCA rainbow maps
- ✅ Offers multiple output/display options
- ✅ Ships with a full CLI interface

## Installation

```bash
# Required Python packages
pip install torch torchvision torchaudio
pip install numpy matplotlib scikit-learn scipy opencv-python pillow tqdm

# CUDA build (match the desired CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Basic usage

```bash
# Process an image with default settings
python remote_sensing_pca_visualization.py your_image.jpg

# Specify an explicit output path
python remote_sensing_pca_visualization.py your_image.jpg -o output.png
```

### Advanced usage

```bash
# Use the ViT-7B backbone (more accurate, slower)
python remote_sensing_pca_visualization.py your_image.jpg -m vit7b16

# Run on CPU
python remote_sensing_pca_visualization.py your_image.jpg -d cpu

# Manually provide foreground points
python remote_sensing_pca_visualization.py your_image.jpg -p 100 200 300 400

# Use the manual HSV-based foreground detector
python remote_sensing_pca_visualization.py your_image.jpg -f manual -t 0.7

# Skip matplotlib windows, just save the result
python remote_sensing_pca_visualization.py your_image.jpg --no-plot
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `image_path` | Path to the input image | Required |
| `-o, --output` | Output path | Auto-generated |
| `-m, --model` | Backbone type (`vitl16` / `vit7b16`) | `vitl16` |
| `-d, --device` | Device (`cuda` / `cpu`) | `cuda` |
| `-f, --foreground_method` | Foreground detector (`auto`/`manual`/`points`/`none`) | `auto` |
| `-t, --threshold` | Foreground threshold | `0.5` |
| `-p, --points` | Point coordinates (`x1 y1 x2 y2 ...`) | None |
| `--no-plot` | Save results without displaying | `False` |

## Foreground Detection Modes

### Auto
Edge detection + dilation to approximate foreground.

### Manual
HSV brightness/saturation thresholds tuned for satellite imagery.

### Points
Create ROIs using explicit coordinates:
```bash
python remote_sensing_pca_visualization.py image.jpg -p 100 200 300 400
```

## Output

Each run produces a three-panel figure:
1. **Original Image** – input RGB image.
2. **Foreground Mask** – detected foreground (grayscale).
3. **PCA Rainbow Map** – DINOv3 PCA visualization.

## Technical Notes

### Backbones
- **ViT-L**: 24 layers, ~300M params, good balance.
- **ViT-7B**: 40 layers, ~6.7B params, higher quality at higher cost.

### Preprocessing
- Resize to multiples of 16×16 patches.
- Normalize with: mean (0.430, 0.411, 0.296), std (0.213, 0.156, 0.143).

### PCA Visualization
- Run 3D PCA on foreground patch features.
- Apply sigmoid for vibrant colors.
- Whitening keeps component variance balanced.

## Sample Output

```
Loading DINOv3 vitl16 backbone...
Model ready on device: cuda
Processing image: satellite_image.jpg
Image sizes: original=(1920, 1080), resized=(768, 1366)
Feature tensor shape: torch.Size([3072, 1024])
Foreground coverage: 45.2%
Running PCA on 1389 patch tokens...
PCA visualization saved to: satellite_image_pca_visualization.png

✅ Visualization complete!
📁 Output file: satellite_image_pca_visualization.png
```

## Troubleshooting

### Common issues

1. **CUDA errors**
   ```bash
   # Fall back to CPU
   python remote_sensing_pca_visualization.py image.jpg -d cpu
   ```

2. **Out of memory**
   ```bash
   # Use the smaller ViT-L backbone
   python remote_sensing_pca_visualization.py image.jpg -m vitl16
   ```

3. **Missing checkpoint**
   - Confirm `pretrained_weights/` contains the SAT-493M weights
   - Or adjust the script to point to your checkpoint

4. **Import errors**
   ```bash
   # Update PYTHONPATH
   export PYTHONPATH=/path/to/dinov3-main:$PYTHONPATH
   ```

### Performance tips

- ViT-L offers a good trade-off between quality and speed.
- For large inputs, `--no-plot` skips matplotlib and saves time.
- Use ViT-7B when GPU memory allows for the best quality.

## Extending

### Use as a Python module

```python
from remote_sensing_pca_visualization import RemoteSensingPCAVisualizer

visualizer = RemoteSensingPCAVisualizer(model_name="vitl16", device="cuda")

result_path = visualizer.visualize(
    image_path="your_image.jpg",
    output_path="output.png",
    foreground_method="auto",
    show_plot=True
)
```

### Custom foreground detection

```python
class CustomVisualizer(RemoteSensingPCAVisualizer):
    def detect_foreground(self, image, method="custom", **kwargs):
        # Implement custom logic here
        pass
```

## Citation

Based on the DINOv3 paper:
```
@article{simeoni2025dinov3,
  title={DINOv3},
  author={Siméoni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Michaël and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timothée and Moutakanni, Théo and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and Jégou, Hervé and Labatut, Patrick and Bojanowski, Piotr},
  year={2025}
}
```

## License

Complies with the license terms of the DINOv3 project.
