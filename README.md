# Modified version of [PointSSM](https://github.com/HQU-3DCV/PointSSM). Updated for DALES LiDAR dataset.


## Usage:
### train: 
sh scripts/train.sh -p python -d dales -c semseg-pointssm-base -n semseg-dales-12 -g 1

### predict:
python3 predict.py --folder /home/fractal01/PointSSM/data/demo_tennet --model_path exp/dales/semseg-dales-12/model/model_best.pth --config_file /home/fractal01/PointSSM/exp/dales/semseg-dales-12/config.py

### args:
    parser.add_argument("--folder", type=str, required=True, help="Folder containing .las files")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint (e.g., model_best.pth)")
    parser.add_argument("--config_file", type=str, default="configs/dales/semseg-dales-12.py", help="Model config file")
    parser.add_argument("--noise_filter", choices=["yes", "no", "interactive"], default="interactive", help="Enable noise filtering")
    parser.add_argument("--smoothing", choices=["yes", "no"], default="yes", help="Enable majority voting smoothing")
    parser.add_argument("--options", nargs="+", action="append", help="override some settings in the used config")

### interactive noise viewer:

Quick Start Guide for the New GUI
When you run python predict.py ... --noise_filter interactive, the new Open3D window will open. Here is how to use it:

Visualizing Noise: Valid points appear in a blue/cyan color ramp (by elevation). As you enable filters and move sliders, points identified as noise will immediately turn White.
Killing Underground Noise: In the Ground Elevation Filter tab, enable the filter and adjust the Floor Percentile (usually 1%) and Buffer. Anything in White is now "underground" noise that won't affect the model.
Preserving Power Lines: Use the Radius Outlier (ROR) section. If wires are turning white, increase the Radius. I've set the default Min Points to 2 so as long as a wire has one neighbor, it stays.
Cleaning Air Noise: Use the DBSCAN section. Increase the Min Cluster Size to delete larger clumps of birds or sensor artifacts.
Finishing: Once the point cloud looks clean (only real features are colored, noise is white), click Accept & Continue Prediction.



-------------------------------------------------------------------------------------------------------------------------------------------


# PointSSM: State Space Model for Large-Scale LiDAR Point Cloud Semantic Segmentation
![poster](./PointSSM-poster.png)
## Results
### Indoor semantic segmentation

|  Model   |   Benchmark   | Num GPUs | Val mIoU |                                       Tensorboard                                       |              Exp Record              |
|:--------:| :-----------: |:--------:|:--------:|:---------------------------------------------------------------------------------------:|:------------------------------------:|
| PointSSM |    ScanNet    |    2     |  78.1%   | [link](exp/scannet/semseg-default/events.out.tfevents.1728354922.cv-Z690-GAMING-X-DDR4) |  [link](exp/scannet/semseg-default)  |
|   PointSSM   |   ScanNet200  |    2     |  35.7%   |         [link](exp/scannet200/semseg-default/events.out.tfevents.1730197260.cv-Z690-GAMING-X-DDR4)         | [link](exp/scanne200/semseg-default) |
|   PointSSM   | S3DIS (Area5) |    2     |  72.8%   |         [link](exp/s3dis/semseg-default/events.out.tfevents.1728446362.cv-Z690-GAMING-X-DDR4)         |   [link](exp/s3dis/semseg-default)   |

### Outdoor semantic segmentation  
|    Model   |   Benchmark   | Num GPUs | Val mIoU |                                       Tensorboard                                        |             Exp Record              |
| :--------: |:-------------:|:--------:|:--------:|:----------------------------------------------------------------------------------------:|:-----------------------------------:|
|    PointSSM    |   nuScenes    |    2     |  80.7%   | [link](exp/nuscenes/semseg-default/events.out.tfevents.1728970636.cv-Z690-GAMING-X-DDR4) | [link](exp/nuscenes/semseg-default) |
|    PointSSM    | SemanticKITTI |    2     |    70.8%     |                                            –                                             |                  –                  |
|    PointSSM    |     DALES     |    2     |  82.3%   |                                            -                                             |                  -                  |

## Data
* Scannet,S3DIS datasets can be downloaded following by (https://huggingface.co/Pointcept/datasets)
* DALES can be downloaded at (https://drive.google.com/file/d/1Ta5Hg7e1dyUCSaDRCfAt_sDyisBuoikw/view?usp=sharing)


## Environment
Our database builds on [Pointcept](https://github.com/Pointcept/Pointcept>) codebase.
### Requirements
- Ubuntu: 18.04 and above.
- *CUDA: 11.8 and above.*
- PyTorch: 1.10.0 and above.
### Conda Environment

```bash
conda create -n pointssm python=3.8 -y
conda activate pointssm
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu113

# PPT (clip)
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# PTv1 & PTv2 or precise eval
cd libs/pointops
# usual
python setup.py install
# docker & multi GPU arch
TORCH_CUDA_ARCH_LIST="ARCH LIST" python  setup.py install
# e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
TORCH_CUDA_ARCH_LIST="7.5 8.0" python  setup.py install
cd ../..

# Open3D (visualization, optional)
pip install open3d

# Mamba-ssm
pip install mamba-ssm==1.0.1
```

## Training and testing
Please follow the Pointcept codebase.

## Acknowledgements
We thank the authors of [Point Transformer V3](https://github.com/Pointcept/Pointcept>). Our implementation is heavily built upon their codes.
