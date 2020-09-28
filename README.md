# CrossNELP: Cross-species Network Embedding and Link Prediction

This repository is for the paper: Virus-Host Interaction Network Prediction Model and Its Application to COVID-19. This model is a machine learning-based virus-host interaction network prediction model which capture underlying network features comprehensively and suggest undiscovered links for further in-depth biological investigation.

## Installation
- Clone the repository
- Run the following command

```bash
pip install -r requirements.txt
```

## Usage

- ### Arguments:
    --bg: bool, if set to True, then the model will only build the network in NetorkX's gml formatting. Default: False

    --evaluate: bool, if set to True, then the model will only run the evaluation part without making predictions. Default: False

   --model_iter_eval: int, the number of full iterations to run for performance evaluation. Default: 30

   --model_iter_pred: int, the number of full iterations to run for making predictions. For example, if set to 5, the model will run for 5 iterations and provide a union of prediction results. This will partially solve the problem caused by lacking proven negative links. Default: 5
   
   --verbose: bool, verbose printing. Default: False

- ### Example
     ```python
    '''Perform a 50-run model performance measurement'''
    python main.py --evaluate True --model_iter_eval 50
    ```
     ```python
    '''Perform link prediction task'''
    python main.py
    ```
   

- ### Input please refer to the file below for formatting details:
    ```python
    /model/data/hetero_data.py
    ```
  - Pair-wise similarity matrices. You can obtain pair-wise similarity matrices for homogeneous proteins by [Data Parser](https://github.com/SupremeEthan/COVID-19-Research-Data-Parser).
  - Protein-Protein Interactions (PPI) and infection relationships should be collected and feed into /model/data/hetero_data.py following this formatting: 
    ```python
    {
        'layer_1': 'virus protein',
        'type_1': ['nsp15'],
        'host_list_1': ['Severe acute respiratory syndrome coronavirus 2'],
        'layer_2': 'host protein',
        'type_2': ['IRF3', 'RIG-I'],
        'host_list_2': ['Homo sapiens', 'Felis catus', 'Macaca mulatta', 'Canis lupus familiaris',
                        'Rhinolophus ferrumequinum', 'Mesocricetus auratus'],
        'relation': 'interacts'
    }
    ```
    The above representation will connect nsp15 of SARS-CoV-2 to IRF3 and RIG-I in the following 
    hosts: Homo sapiens, Felis catus, Macaca mulatta, Canis lupus familiaris, Rhinolophus 
    ferrumequinum and Mesocricetus auratus. The link type will be set as "interacts"
  - Protein functions should be collected and feed into /model/data/hetero_data.py following this 
    formatting: 
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

## Citing
If you find CrossNELP is useful for your research, please consider citing the following papers:

```bash
TBD
```

## License
This project is using [MIT](https://choosealicense.com/licenses/mit/) license.
