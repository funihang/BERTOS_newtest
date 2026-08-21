# BERTOS New Test

BERTOS new version written to adapt to the new version of some packages.

[Original Version](https://github.com/usccolumbia/BERTOS)
[Online Toolbox](http://www.materialsatlas.org/bertos)

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

1. PyTorch for computers with Nvidia GPU.
Go to pytorch histort installation and Select the version according to your CUDA version (recommend `pip install`).

3. Other packagess
```
pip install -r requirements.txt
```  

## Datasets  
Our training process is carried out on our BERTOS datasets. Now we train our model on the updated `ICSD_CN` dataset.

## Usage
### Training
The command is to train a BERTOS model.  
```
python train_BERTOS.py
```
If you want to change the model setting, please check `./random_config/config.json`

If you want to change the training setting, please check the hyperparameter setting in the `train_bertos.py` (since this is just a test instead of the official code, so Nihang have not  added the arguments)

### Predict
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

### Check charge neutrality for hypothetical formulas
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
Fu, Nihang, Jeffrey Hu, Ying Feng, Gregory Morrison, Hans‐Conrad zur Loye, and Jianjun Hu. "Composition Based Oxidation State Prediction of Materials Using Deep Learning Language Models." Advanced Science (2023): 2301011. [PDF](https://arxiv.org/pdf/2211.15895)

```

# Contact
If you have any problem using BERTOS, feel free to contact via [our email](mailto:funihang@gmail.com).
