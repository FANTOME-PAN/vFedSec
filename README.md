# vFedSec: Efficient Secure Aggregation for Vertical Federated Learning via Secure Layer
# demo on tabular data
The repo currently has example code for the bank marketing dataset
and can be extended to other tabular datasets via the following steps.

### 1. Preprocess the data

Most data may be in csv form, and it would be easier to work with 
if the data is preprocessed. No specific requirements for this step.

### 2. Implement functions in `data_loaders.py`

Please implement functions
- `get_data_loader`
- `get_sample_selector`

The active party will call `get_data_loader` to get the data loader, 
and each passive party will call `get_sample_selector` to create its selector.

Please turn to the documentation in `IDataLoader` and `ISampleSelector` for more details.

### 3. Implement functions in `models.py`

Please implement functions

- `generate_passive_party_local_module`
- `generate_active_party_local_module`
- `generate_global_module`
- `get_criterion`

The FL solution will use the above functions to create NN models and compute the loss.

### 4. Configure `settings.py`

The file `settings.py` contains global settings and hyperparameters for training.

### 5. run `run.py`

Just run it. The log will be found in `logs` folder.


#### In case of potential conflicts, please avoid modifying files in `core` folder.


## Reference
[Paper link](https://arxiv.org/abs/2305.16794)

```bibtex
@misc{qiu2023vfedsec,
      title={vFedSec: Efficient Secure Aggregation for Vertical Federated Learning via Secure Layer}, 
      author={Xinchi Qiu and Heng Pan and Wanru Zhao and Chenyang Ma and Pedro P. B. Gusmao and Nicholas D. Lane},
      year={2023},
      eprint={2305.16794},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
