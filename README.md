### Nemo NER 

This repo uses [NeMo 1.4.0](https://github.com/NVIDIA/NeMo/tree/r1.4.0). To specialize existing Bert models to Named 
entity recognition tasks. 

#### Installation 

##### Conda 
   
- Create conda env with python 3.8.10 

```shell
conda create --name nemo-1.4.0 python=3.8.10
conda activate nemo-1.4.0 
``` 

- Install requirements 

```shell
pip install -r pre-requirements.txt
pip install -r requirements.txt
```