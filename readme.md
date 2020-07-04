# Implementation of [SCARL: Attentive Reinforcement Learning-Based Scheduling in a Multi-Resource Heterogeneous Cluster](https://ieeexplore.ieee.org/document/8876692)


#### Description

- `baselines.py`: baseline heuristics.
- `networks.py`: Model Networks implemented.
- `mc_job_dist.py`: Job sequence generation.
- `mc_env.py`: OpenAI Gym compatible cluster simulation environment.
- `mc_eval.py`: Run model training
- `generate-train-data.ipynb`: train data generation
- `envTest-30-2_6.ipynb`: simple evaluation.



#### Usage example
1. create dataset using `generate-train-data.ipynb`
2. python mc_eval.py --step_size=15 --save_dir=model-2.0 --input=tr_2.0.pkl


-----


I'm sorry that I can't disclose the script for training/running on actual k8s enironment. I can't find things I've done.


Implementation of the environment is mainly brought and extended from env of [deeprm](https://github.com/hongzimao/deeprm)