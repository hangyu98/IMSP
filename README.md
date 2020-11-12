# Infection Mechanism and Spectrum Prediction (IMSP) Model

This repository is for the paper: Network-based Virus-Host Interaction Prediction with Application to SARS-CoV-2. This model predicts virus-host interactions at both protein and organism levels.

## Installation
- Clone the repository
- Run the following command

```bash
pip install -r requirements.txt
```

## Usage

- ### Arguments:
    --bg: bool, if set to True, then the model will only build the network in NetorkX's GML formatting. Default: False

    --evaluate: bool, if set to True, then the model will only run the evaluation part without making predictions. Default: False

   --eval_iter: int, the number of full iterations to run for performance evaluation. Default: 30

   --pred_iter: int, the number of full iterations to run for making predictions. For example, if set to 5, the model will run for 5 iterations and provide a union of prediction results. This will partially solve the problem caused by lacking proven negative links. Default: 5

- ### Example
     ```python
    '''Perform link prediction task'''
    python main.py
    ```

     ```python
    '''Perform a 50-run model performance measurement'''
    python main.py --evaluate True --eval_iter 50
    ```

- ### Input:
  - Pair-wise similarity matrices. You can obtain pair-wise similarity matrices for protein homologs by using our [Data Parser](https://github.com/SupremeEthan/IMSP-Parser).
  - Protein-Protein Interactions (PPI) and infection relationships should be collected and feed into /model/data/hetero_data.py following this formatting: 
    ```python
    {
        'group_1': 'virus protein',
        'type_1': ['nsp15'],
        'host_list_1': ['Severe acute respiratory syndrome coronavirus 2'],
        'group_2': 'host protein',
        'type_2': ['IRF3', 'RIG-I'],
        'host_list_2': ['Homo sapiens', 'Felis catus', 'Macaca mulatta', 'Canis lupus familiaris',
                        'Rhinolophus ferrumequinum', 'Mesocricetus auratus'],
        'relation': 'interacts'
    }
    ```
    Specifically, the input above will connect nsp15 of SARS-CoV-2 to IRF3 and RIG-I in the following hosts: Homo sapiens, Felis catus, Macaca mulatta, Canis lupus familiaris, Rhinolophus ferrumequinum, and Mesocricetus auratus. The link type will be set as "interacts" (PPI).
  - Protein functions should be collected and feed into /model/data/hetero_data.py following this formatting: 
    ```python
    protein_function_data = {
        'ACE2': 'virus receptor activity',
        'DPP4': 'virus receptor activity',
        'IRF3': 'interferon regulatory factor',
        'IRF7': 'interferon regulatory factor',
        'IRF9': 'interferon regulatory factor',
        'MAVS': 'interferon regulatory factor',
        ...
    }
    ```
  - In addition to these, please also provide two lists of the full name for viruses and hosts
    ```python
    # list of hosts
    list_of_hosts = ['Homo sapiens', 'Felis catus', 'Mus musculus',
                 'Rattus norvegicus', 'Canis lupus familiaris',
                 'Ictidomys tridecemlineatus', 'Camelus dromedarius', 'Bos taurus', 'Pan troglodytes',
                 'Gallus gallus', 'Oryctolagus cuniculus', 'Equus caballus', 'Macaca mulatta', 'Ovis aries',
                 'Sus scrofa domesticus', 'Rhinolophus ferrumequinum', 'Mesocricetus auratus']
    # list of viruses
    list_of_viruses = ['Human coronavirus OC43', 'Human coronavirus HKU1',
                   'Middle East respiratory syndrome-related coronavirus',
                   'Severe acute respiratory syndrome coronavirus 2',
                   'Severe acute respiratory syndrome-related coronavirus', 'Human coronavirus 229E',
                   'Human coronavirus NL63']
    ```
- ### Output:
  - The link prediction results are available at ```/data/prediction```
    - ```/data/prediction/prediction_infects.csv``` contains infection predictions
    - ```/data/prediction/prediction_interacts.csv``` contains PPI predictions
  - The performance evaluation results are available at ```/data/evaluation```
    - ```/data/prediction/comparison_summary.csv``` contains the means and STDs for all evaluation metrics
    - ```/data/prediction/comparison_details.csv``` logs the performance for all the models in each run
  
- ### Customizable filter:
  - We understand that the prediction results might need customized filters in different scenarios. We have provided some sample code available at ```/customize/customized_filter.py```

## Citing
If you find IMSP is useful for your research, please consider citing the following papers:

```bash
Coming soon
```

## License
This project is using [MIT](https://choosealicense.com/licenses/mit/) license.
