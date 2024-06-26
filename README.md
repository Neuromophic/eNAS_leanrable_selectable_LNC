# Neural Architecture Search for Highly Robust Printed Neuromorphic Circuits

This github repository is for the paper at ICCAD'24 - Neural Architecture Search for Highly Robust Printed Neuromorphic Circuits. Thhis paper proposed an evolutionary algorithm (EA)-based neural architecture search method for analog printed neuromorphic circuits. Leveraging the capability of EA, we not only search the circuit architecture, but also we let the algorithm to automatically select the best activation circuits of the neuromorphic circuits for target objective.

cite as
```
Neural Architecture Search for Highly Robust Printed Neuromorphic Circuits
Pal, P; Zhao, H.; Gheshlaghi, T; Hefenbrock, M.; Beigl, M.; Tahoori, M.
2024 International Conference on Computer-Aided Design (ICCAD), October, 2024 IEEE/ACM.
```



Usage of the code:

1. Training of printed neural networks with gradient-based method:

Simply run the experiments by running command lines in `run_baseline_pNN_0.0_variation.sh` through `bash` or run them separately, e.g.,

~~~
$ python3 experiment.py --DATASET 00 --SEED 00 --e_train 0.0 --projectname VariationAwareTraining
...
~~~

This is also true for the file `run_baseline_pNN_0.05_variation.sh` and `run_baseline_pNN_0.1_variation.sh`, they just contains command lines with different variation values. The files are the experiment setups reported in the paper, of course one can also modify any reasonable values as they want. Notably, this is a baseline that has already been proposed in
```
Highly Bespoke Robust Printed Neuromorphic Circuits
Zhao, H.; Hefenbrock, M.;Yang, Z; Beigl, M.; Tahoori, M.
2023 Design, Automation & Test in Europe Conference & Exhibition (DATE), April 17-19, 2023, IEEE.
```
and can be found at https://github.com/Neuromophic/LearnableNonlinearCircuits.



2. Training of printed neural networks with proposed EA-based method:

Simply run the experiments by running command lines in `run_EA_0.0_variation.sh` through `bash` or run them separately, e.g.,

~~~
python3 experiment_EA.py --DATASET 00 --SEED 00 --e_train 0.0 --projectname VariationAware_EA
...
~~~

This is also true for the file `run_EA_0.05_variation.sh` and `run_EA_0.1_variation.sh`, they just contains command lines with different variation values.


3. After training printed neural networks, the trained networks are in `./VariationAwareTraining/models/` or `./VariationAware_EA/models/`, the log files for training can be found in `./VariationAwareTraining/log/` or `./VariationAware_EA/log/`. The name of the folder is identical to the `projectname` that you defined in the command line. If there is still files in  e.g.,`./VariationAwareTraining/temp/`, you should run the corresponding command line to train the networks further. Note that, each training is limited to 48 hours, you can change this time limitation in `configuration.py`


4. Evaluation can be done by running the notebooks in `./VariationAware_EA/` folder

* `check_acts.ipynb` collects the number and ratio of activation circuits that has been selected under different variations
* `check_runtime.ipynb` collects the training time of the code
* `Evaluation.ipynb` runs the trained models on both validation set and test set for further processing
* `Summary.ipynb` finds the best model on each dataset based on validation set, and report the performance on test set.