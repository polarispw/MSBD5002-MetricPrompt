## Data 
We conduct experiments on 3 text classification datasets: AG's News, DBPedia and Yahoo Answers Topics. 
Run the following commands to prepare datasets: 
```
bash prepare_data.sh
```


## Environment
We list the specific packages used in this project in `requirements.txt`. 
The following commands will help you build a compatible environment: 
```
conda create -n metricprompt python=3.9
conda activate metricprompt
pip install -r requirements.txt
```
If your GPU is based on Ampere architecture, you can install [kernl](https://github.com/ELS-RD/kernl/) package with the following command to accelerate the inference process.
```
pip install 'git+https://github.com/ELS-RD/kernl'
```
After installing kernl, set `--kernl_accerleration 1` in `run.sh` to activate the acceleration.


## Quickstart
You can run MetricPrompt simply with `run.sh`.
The following commands runs MetricPrompt for AG's News 2-shot setting, where the number of training epochs is set as 120.
```
bash metricprompt.sh agnews 2 120
```


## Code
- `main.py` is the entrance of MetricPrompt's training and testing procedure.
- `model.py` contains model definition, as well as the implementation of the optimization and inference process.
- `dataloader.py` contains data processing code. 



