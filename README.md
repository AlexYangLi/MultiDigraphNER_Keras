# Keras Implementation of Multi-Digraph NER Model Incorporating with Gazetteers

This is a Keras (Tensorflow backend) implementation of multi-digraph ner model as described in 
the paper [Ding et.al. A Neural Multi-digraph Model for Chinese NER with Gazetteers. ACL2019](https://www.aclweb.org/anthology/P19-1141.pdf). 
Author's Pytorch implementation is available in this [repo](https://github.com/PhantomGrapes/MultiDigraphNER).  

## Environments 
- python==3.6.6
- Keras==2.3.0
- tensorflow-gpu==1.13.1

## Run

### Prepare data
All the data are copied from author's [repo](https://github.com/PhantomGrapes/MultiDigraphNER/tree/master/data). 

### Preprocess
```python
python3 preprocess.py
```

The processed data will be stored in `data` dir.

### Train
```python
python3 main.py
```

## Performance


## Reference

- [A Neural Multi-digraph Model for Chinese NER with Gazetteers](https://www.aclweb.org/anthology/P19-1141.pdf), ACL2019  
- [PhantomGrapes/MultiDigraphNER](https://github.com/PhantomGrapes/MultiDigraphNER)
