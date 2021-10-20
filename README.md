

## Mixture weights optimisation for Alpha-Divergence Variational Inference

This project can be used to reproduce the experiments presented in

Kamélia Daudel and Randal Douc. ["Mixture weights optimisation for Alpha-Divergence Variational Inference"](https://arxiv.org/abs/2106.05114). Accepted at Neurips 2021.

Parts of this implementation were modified from the code provided [here](
https://github.com/kdaudel/AlphaGammaDescent) and the code has been developed and tested with python3.6. 

### Reproducing the figures in the paper

To run the experiments, run:

> python mainMixtureModel.py

To create and save the figures, run:

> python plotResults.py

The figures can then be found in
```bash
.
├── results/
```

### Choice of the Exploration step

Setting
```python
main_on = True
```
in both mainMixtureModel.py and plotResults.py, will select the Exploration step described in Section 5 of the paper.

Setting 
```python
main_on = False
```
in both mainMixtureModel.py and plotResults.py, will select the Exploration step described in Appendix D.3.2 of the paper.

### Parallelisation

Note that there is a possibility to parallelise the code via the joblib package. 

To do so, in mainMixtureModel.py replace

```python
#Parallel(nb_cores_used)(delayed(main_function)(i) for i in i_list) 
for i in i_list:
    main_function(i) 
```

by

```python
Parallel(nb_cores_used)(delayed(main_function)(i) for i in i_list) 
#for i in i_list:
#    main_function(i) 
```

## Feedback
Feedback is greatly appreciated. If you have any questions, feel me to send me an [email](mailto:kamelia.daudel@gmail.com).

All rights reserved.
