# MSBD5002-MetricPrompt

This is a repo of MSBD5002 project, reproducing MetricPrompt in KDD 2023

The code structure is as follows：

```
Baseline
	|---baseline.py
	|---scripts/	# config files of openprompts to run baselines
MetricPrompt
	|---classes/	# class labels of datasets
	|---process_data.py	# data processing code
	|---dataloader.py	# data utils for MetricPrompt
	|---main.py	# entrance of train and test
	|---model.py	# implementation of MetricPrompt
	|---utils.py
	|---prepare_data.sh
	|---run.sh
```

## Quick Start

### Data

We conduct experiments on 3 text classification datasets: AG's News, DBPedia and Yahoo Answers Topics. 
Run the following commands to prepare datasets: 

```shell
bash prepare_data.sh
```

### Environment

We list the specific packages used in this project in `requirements.txt`. 
The following commands will help you build a compatible environment: 
```shell
conda create -n metricprompt python=3.9
conda activate metricprompt
pip install -r requirements.txt
```
If your GPU is based on Ampere architecture, you can install [kernel](https://github.com/ELS-RD/kernl/) package with the following command to accelerate the inference process.
```shell
pip install 'git+https://github.com/ELS-RD/kernl'
```
After installing kernel, set `--kernl_accerleration 1` in `run.sh` to activate the acceleration.

### Baseline

All baselines are implemented using OpenPrompt, which is a awesome prompt tuning frame work.

To run baselines, simply use the following command

```shell
cd Baseline
python baseline.py --config-yaml scripts/agnews/manual_verb.yaml --few-shot 2 --train-epochs 120
```

 For more options in the config file, you can refer to [repo](https://github.com/thunlp/OpenPrompt)

### Metric Prompt

You can run Metric Prompt simply with `run.sh`.
The following commands runs MetricPrompt for AG's News 2-shot setting, where the number of training epochs is set as 120.

```shell
bash metricprompt.sh agnews 2 120
```
