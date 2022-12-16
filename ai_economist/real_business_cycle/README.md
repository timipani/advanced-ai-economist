
# Real Business Cycle (RBC)
This directory implements a **Real-Business-Cycle** (RBC) simulation with many heterogeneous, interacting strategic agents of various types, such as **consumers, firms, and the government**. For details, please refer to this paper "Finding General Equilibria in Many-Agent Economic Simulations using Deep Reinforcement Learning (ArXiv link forthcoming)". We also provide training code that uses deep multi-agent reinforcement learning to determine optimal economic policies and dynamics in these many agent environments. Below are instructions required to launch the training runs.

**Note: The experiments require a GPU to run!**

## Dependencies

- torch>=1.9.0
- pycuda==2021.1
- matplotlib==3.2.1

## Running Local Jobs
To run a hyperparameter sweep of jobs on a local machine, use (see file for command line arguments and hyperparameter sweep dictionaries)

```
python train_multi_exps.py
```

## Configuration Dictionaries

Configuration dictionaries are currently specified in Python code, and then written as `hparams.yaml` in the job directory. For examples, see the file `constants.py`. The dictionaries contain "agents", "world", and "train" dictionaries which contain various hyperparameters.

## Hyperparameter Sweeps

The files `train_multi_exps.py` allow hyperparameter sweeps. These are specified in `*_param_sweeps` dictionaries in the file. For each hyperparameter, specify a list of one or more choices. The Cartesian product of all choices will be used.

## Approximate Best Response Training

To run a single approximate best-response (BR) training job on checkpoint policies, run `python train_bestresponse.py ROLLOUT_DIR NUM_EPISODES_TO_TRAIN --ep-strs ep1 ep2 --agent-type all`. The `--ep-strs` argument specifies which episodes to run on (for example, policies from episode 0, 10000, and 200000). These must be episodes for which policies were saved. It is possible to specify a single agent type.


## What Will Be Saved?

A large amount of data will be saved -- one can set hyperparamter `train.save_dense_every` in the configuration dictionary (`hparams.yaml`/`constants.py`) to reduce this.

At the top level, an experiment directory stores the results of many runs in a hyperparameter sweep. Example structure:

```
experiment/experimentname/
    rollout-999999-99999/
        brconsumer/
            ...
        brfirm/
            episode_XXXX_consumer.npz
            episode_XXXX_government.npz
            episode_XXXX_firm.npz
            saved_models/
                consumer_policy_XXX.pt
                firm_policy_XXX.pt
                government_policy_XXX.pt.
        brgovernment/
            ...
        hparams.yaml