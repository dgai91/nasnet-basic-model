# Network-Architecture-Search-Basic-Model
This repo is an implementation of NASNET, which proposed from [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578).

My code referenced [this repo](https://github.com/wallarm/nascell-automl). 

### Environment:  
CPU i7-7700      Python 3.6.8     Pytorch 1.1.0    

### Repo Contains:
NAS model(NAS cell + reinforce framework)
CNN model
child network manager
### Bash into pyt_nasnet and input:
```
python pyt_train.py
```

### Nasnet is an automl model, the main structure is:

<img src="https://github.com/lawlietAi/Network-Architecture-Search-Basic-Model/blob/master/structure.png?raw=true" height=100% width=100%>

