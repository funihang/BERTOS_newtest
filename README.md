# BERTOS New Test

BERTOS new version written to adapt to the new version of some packages.

[Original Version](https://github.com/usccolumbia/BERTOS)

[Online Toolbox](https://www.materialsatlas.org/apps/oxidation-states)

## Table of Contents
- [Installations](#Installations)

- [Datasets](#Datasets)

- [Usage](#Usage)

- [Pretrained Models](#Pretrained-models)

- [Performance](#Performance)

- [Acknowledgement](#Acknowledgement)

## Installations

0. Set up a virtual environment
```
conda create -n bertos  python=3.12
conda activate bertos
```

1. PyTorch for computers with NVIDIA GPU.
Go to pytorch history installation and Select the version according to your CUDA version (recommend `pip install`).

3. Other packages
```
pip install -r requirements.txt
```  

## Datasets  
Release an updated version of `ICSD_CN` dataset.

Pending: `ICSD`, `ICSD_oxide`, `ICSD_oxide_CN`

## Usage
### Training
Train a BERTOS model (on the `ICSD_CN` dataset).  
```
python train_BERTOS.py
```
If you want to change the model setting, please check `./random_config/config.json`

If you want to change the training setting, please check the hyperparameter setting in the `train_bertos.py` (since this is just a test instead of the official code, so Nihang have not  added arguments)

### Predict Oxidation States
Run `getOS.py` file to get predicted oxidation states for an input formula or input formulas.csv file containing multiple formulas. <br>
Using default pretrained model (trained on ICSD_CN):
```
python getOS.py --i SrTiO3 --model_name_or_path ./trained_models/ICSD_CN
python getOS.py --f formulas.csv --model_name_or_path ./trained_models/ICSD_CN
```
Or using your model:
```
python getOS.py --i SrTiO3 --model_name_or_path ./model_directory
python getOS.py --f formulas.csv --model_name_or_path ./model_directory

```

### Check Charge Neutrality
Run `checkCN.py` file to check charge neutrality for an input formula or input formulas.csv file containing multiple formulas. <br>
Using default pretrained model (trained on ICSD_CN):
```
python checkCN.py --i SrTiO3 
python checkCN.py --f formulas.csv 
```
Or using your model:
```
python checkCN.py --i SrTiO3 --model_name_or_path ./model_directory
python checkCN.py --f formulas.csv --model_name_or_path ./model_directory
```

## Acknowledgement
We use the transformer model as implemented in Huggingface.
```
@article{wolf2019huggingface,  
  title={Huggingface's transformers: State-of-the-art natural language processing},  
  author={Wolf, Thomas and Debut, Lysandre and Sanh, Victor and Chaumond, Julien and Delangue, Clement and Moi, Anthony and Cistac, Pierric and Rault, Tim and Louf, R{\'e}mi and Funtowicz, Morgan and others},  
  journal={arXiv preprint arXiv:1910.03771},  
  year={2019}  
}
```

## Cite our work
```
@article{fu2023composition,
  title={Composition based oxidation state prediction of materials using deep learning language models},
  author={Fu, Nihang and Hu, Jeffrey and Feng, Ying and Morrison, Gregory and Loye, Hans-Conrad zur and Hu, Jianjun},
  journal={Advanced Science},
  volume={10},
  number={28},
  pages={2301011},
  year={2023},
  publisher={Wiley Online Library}
}

```

# Contact
If you have any problem using BERTOS, feel free to contact via [our email](mailto:funihang@gmail.com).
