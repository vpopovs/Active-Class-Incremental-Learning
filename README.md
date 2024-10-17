# WIP
# Novelty detection with Active Class-Incremental Learning on Long-tailed datasets

This repository contains the research framework used in the paper: 
"Novelty detection with Active Class-Incremental Learning on Long-tailed datasets."

## Results
WIP

# Codebase
This repository contains the scripts required to run the experiments from the corresponding paper. The code is licensed under the ["**CC BY-NC 4.0**" license](https://creativecommons.org/licenses/by-nc/4.0/); feel free to adapt the code as long as appropriate credit is provided. The code cannot be used for commercial purposes.

## Configurations with Hydra
The code base runs using [Hydra](https://hydra.cc/docs/intro/) configurations, which may have a learning curve to use.
In short, there are three command line functions you need to know:
- **Overwriting a single variable:** Access the variable like this: `trainer.epochs=10`, where `trainer` is the module, `epochs` is the variable, and `10` is the value it should be changed to.
- **Adding yaml config (`+`):** By default, no modules are added, therefore `+trainer=steppedtrainer` is required to include a trainer for the run. `trainer` is the module and `steppedtrainer` refers to the `steppedtrainer.yaml` file. Its contents will be placed under `trainer`.

    If the yaml is nested in another folder, you may need to use `+model/ResNet@model.network=ResNet50`, where the contents of `model/ResNet/ResNet50.yaml` will be placed under `model.network`.
- **Removing a module (`~`)**: `~data.dataset.val` will remove the module `data.dataset.val` and all its contents from the run.

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data preperation
- **Places365-LT** and **ImageNet-LT**: We reuse the data preparation of [OLTR](https://github.com/zhmiao/OpenLongTailRecognition-OLTR?tab=readme-ov-file#data-preparation). We also use their annotations and their open-set dataset.
> **Note**: The ImageNet version is **2012**, not 2014.
- **iNaturalist2018(-Plantae)**: Download the dataset from this repository [iNaturalist2018](https://github.com/visipedia/inat_comp/tree/master/2018#Data).

Your `data/` directory should look like this, with the .txt files inside the corresponding folders (e.g., `data/Places365/Places_LT_train.txt`):

```
data
  ├──ImageNet2012
  │  ├──ImageNet_LT_open
  │  ├──test
  │  ├──train
  │  └──val
  ├──Places365
  │  ├──data_256
  │  ├──Places_LT_open
  │  ├──test_256
  │  └──val_256
  └──iNaturalist2018
     └──train_val2018
```

### Other datasets
Any dataset can be adapted to work within this research framework. We have included `torchvision`'s MNIST dataset in `ACIL/data/dataset/MNIST.yaml` to demonstrate the configurations. 

## Running experiments
These examples use the Places365-LT dataset, however, any torchvision compatible dataset will work in this framework.
The option used in the paper were:
```bash
+data/dataset=Places365_LT data.query.n_samples=1500
+data/dataset=ImageNet_LT data.query.n_samples=2000
+data/dataset=iNaturalist2018_Partial data.dataset.data.families=[Plantae] data.query.n_samples=3000
```
### Single step Open-set Recognition

##### Frozen
``` bash
python3 ACIL/main.py --config-name=osr \
    +trainer=trainer +model=queryaftertrain \
    +data/dataset=Places365_LT \
    +data/query/strategy=energy \
    +model/ResNet@model.network=ResNet50 \
    model.network.freeze=True
```

##### Both
``` bash
python3 ACIL/main.py --config-name=osr \
    +trainer=finetuned +model=steppedmodelAL \
    +data/dataset=Places365_LT \
    +data/query/strategy=energy \
    +model/ResNet@model.network=ResNet50 
```

### Active Class-Incremental Learning

##### Frozen
``` bash
python3 ACIL/main.py --config-name=ACIL \
    +trainer=steppedtrainer \
    +data/dataset=Places365_LT \
    ~data.dataset.val \
    +data/query/strategy=energy \
    +model/ResNet@model.network=ResNet50 \
    model.steps=[0,None] trainer.steps=1
```

##### Both Continuous
``` bash
python3 ACIL/main.py --config-name=ACIL \
    +trainer=finetuned +model=steppedmodelAL \
    +data/dataset=Places365_LT \
    ~data.dataset.val \
    +data/query/strategy=energy \
    +model/ResNet@model.network=ResNet50 \
    model.reload_model_on_new_classes=False \
    model.fc.transfer_weights=True
```

##### Both Reload
``` bash
python3 ACIL/main.py --config-name=ACIL \
    +trainer=finetuned +model=steppedmodelAL \ 
    +data/dataset=Places365_LT \
    ~data.dataset.val \
    +data/query/strategy=energy \
    +model/ResNet@model.network=ResNet50 \
    model.reload_model_on_new_classes=True
```

For ACIL, `~data.dataset.val` is required, otherwise the predefined validation set is used.

### Configurations
- **Datasets**:
    - Places365-LT: `+data/dataset=Places365_LT`
    - ImageNet-LT: `+data/dataset=ImageNet_LT`
    - iNaturalist2018-Plantae: `+iNaturalist2018_Partial data.dataset.data.families=[Plantae]`
- **Pretrained models**: `model.network.pretrained=<PATH_TO_MODEL>`
- **Trainable parameters**:
    Frozen and Both are configured through the trainer (`steppedtrainer` and `finetuned`, respectively).
    - **Reload**: `model.reload_model_on_new_classes=True`
    - **Continuous**: `model.reload_model_on_new_classes=False model.fc.transfer_weights=True`
- **Querying**:
    - Initial size: `data.initial.size=<INT>`
    - Query size: `data.query.n_samples=<INT>` (by default, this is the same as `data.initial.size`)
    - Iterations: `trainer.iterations=<INT>`
- **Detectors**:
    `DETECTOR_NAME`: one of `Random, Uncertainty, Margin, KLMatching, Energy, Entropy, KNN, ViM, ReAct, DICE, SHE, ASH-s, ASH-b, ASH-p, RMD, ODIN`
    - Single detector: `+data/query/strategy=<DETECTOR_NAME>`
    - Single detector, but check multiple: `data/query=query_checkall data.query.strategy.name=<DETECTOR_NAME>`
    - Hybrid detectors: # TODO
- **Seed**: `seed=<INT>`

### Pretraining with MoCo
If you are using MoCo, please refer to [their repository](https://github.com/facebookresearch/moco.git) for more information. 

To include it, please use:
```bash
    git submodule add https://github.com/facebookresearch/moco.git external/moco/moco
```

Depending on your available resources, you might need to adapt their `main.py`. I included my adaptation, which was run on a slurm cluster, implying that multiple nodes were connected through a local network. The model used in the paper was trained with the following command line, where `<RANK>` is `[0-3]` (as `world_size=4`), `MASTER_HOSTNAME` is the address to `RANK=0`, and `PORT` is an available port in the network.
``` bash
    python3 -u external/moco/main.py --config-name=MoCo_distributed \
    +data/dataset=Places365_LT  \
    trainer.arch=resnet50 \
    trainer.epochs=600 \
    trainer.workers=16 \
    seed=1 \
    trainer.dist_url=tcp://<MASTER_HOSTNAME:PORT> \
    trainer.world_size=4 \
    trainer.rank=<RANK> \
    trainer.batch_size=256
```
