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

hetero_edge_data = [
    {
        'group_1': 'virus protein',
        'type_1': ['nsp15'],
        'host_list_1': ['Severe acute respiratory syndrome coronavirus 2'],
        'group_2': 'host protein',
        'type_2': ['IRF3', 'RIG-I'],
        'host_list_2': ['Homo sapiens', 'Felis catus', 'Macaca mulatta', 'Canis lupus familiaris',
                        'Rhinolophus ferrumequinum', 'Mesocricetus auratus'],
        'relation': 'interacts'
    },

    {
        'group_1': 'virus protein',
        'type_1': ['nsp15'],
        'host_list_1': ['Severe acute respiratory syndrome-related coronavirus'],
        'group_2': 'host protein',
        'type_2': ['IRF3', 'MAVS'],
        'host_list_2': ['Homo sapiens', 'Felis catus', 'Rattus norvegicus', 'Rhinolophus ferrumequinum',
                        'Mus musculus'],
        'relation': 'interacts'
    },

    {
        'group_1': 'virus protein',
        'type_1': ['ORF3b'],
        'host_list_1': ['Severe acute respiratory syndrome-related coronavirus'],
        'group_2': 'host protein',
        'type_2': ['IRF3', 'MAVS', 'MDA5', 'RIG-I', 'IRF9', 'STAT1', 'STAT2'],
        'host_list_2': ['Homo sapiens', 'Felis catus', 'Rattus norvegicus', 'Rhinolophus ferrumequinum',
                        'Mus musculus'],
        'relation': 'interacts'
    },

    {
        'group_1': 'virus protein',
        'type_1': ['ORF6'],
        'host_list_1': ['Severe acute respiratory syndrome-related coronavirus'],
        'group_2': 'host protein',
        'type_2': ['IRF3', 'IRF9', 'STAT1', 'STAT2'],
        'host_list_2': ['Homo sapiens', 'Felis catus', 'Rattus norvegicus', 'Rhinolophus ferrumequinum',
                        'Mus musculus'],
        'relation': 'interacts'
    },

    {
        'group_1': 'virus protein',
        'type_1': ['ORF6'],
        'host_list_1': ['Severe acute respiratory syndrome coronavirus 2'],
        'group_2': 'host protein',
        'type_2': ['IRF3', 'RIG-I'],
        'host_list_2': ['Homo sapiens', 'Felis catus', 'Macaca mulatta', 'Canis lupus familiaris',
                        'Rhinolophus ferrumequinum', 'Mesocricetus auratus'],
        'relation': 'interacts'
    },

    {
        'group_1': 'virus protein',
        'type_1': ['PLpro'],
        'host_list_1': ['Severe acute respiratory syndrome-related coronavirus'],
        'group_2': 'host protein',
        'type_2': ['NF-kB', 'IRF3', 'TBK1'],
        'host_list_2': ['Homo sapiens', 'Felis catus', 'Rattus norvegicus', 'Rhinolophus ferrumequinum',
                        'Mus musculus'],
        'relation': 'interacts'
    },

    {
        'group_1': 'virus protein',
        'type_1': ['PLpro'],
        'host_list_1': ['Middle East respiratory syndrome-related coronavirus'],
        'group_2': 'host protein',
        'type_2': ['NF-kB', 'IRF3', 'TBK1'],
        'host_list_2': ['Homo sapiens', 'Camelus dromedarius', 'Mus musculus', 'Rattus norvegicus'],
        'relation': 'interacts'
    },

    {
        'group_1': 'virus protein',
        'type_1': ['N protein'],
        'host_list_1': ['Severe acute respiratory syndrome-related coronavirus'],
        'group_2': 'host protein',
        'type_2': ['MAVS', 'MDA5', 'RIG-I', 'IRF3'],
        'host_list_2': ['Homo sapiens', 'Felis catus', 'Rattus norvegicus', 'Rhinolophus ferrumequinum',
                        'Mus musculus'],
        'relation': 'interacts'
    },

    {
        'group_1': 'virus protein',
        'type_1': ['M protein'],
        'host_list_1': ['Severe acute respiratory syndrome-related coronavirus'],
        'group_2': 'host protein',
        'type_2': ['NF-kB', 'IRF3'],
        'host_list_2': ['Homo sapiens', 'Felis catus', 'Rattus norvegicus', 'Rhinolophus ferrumequinum',
                        'Mus musculus'],
        'relation': 'interacts'
    },

    {
        'group_1': 'virus protein',
        'type_1': ['M protein'],
        'host_list_1': ['Middle East respiratory syndrome-related coronavirus'],
        'group_2': 'host protein',
        'type_2': ['STAT1', 'STAT2', 'IRF9', 'IRF3', 'TBK1'],
        'host_list_2': ['Homo sapiens', 'Camelus dromedarius', 'Mus musculus', 'Rattus norvegicus'],
        'relation': 'interacts'
    },

    {
        'group_1': 'virus protein',
        'type_1': ['nsp1'],
        'host_list_1': ['Severe acute respiratory syndrome-related coronavirus'],
        'group_2': 'host protein',
        'type_2': ['STAT1', 'STAT2'],
        'host_list_2': ['Homo sapiens', 'Felis catus', 'Rattus norvegicus', 'Rhinolophus ferrumequinum',
                        'Mus musculus'],
        'relation': 'interacts'
    },

    {
        'group_1': 'virus protein',
        'type_1': ['ORF4a'],
        'host_list_1': ['Middle East respiratory syndrome-related coronavirus'],
        'group_2': 'host protein',
        'type_2': ['STAT1', 'STAT2', 'IRF9', 'NF-kB', 'PRKRA', 'MDA5', 'IRF3'],
        'host_list_2': ['Homo sapiens', 'Camelus dromedarius', 'Mus musculus', 'Rattus norvegicus'],
        'relation': 'interacts'
    },

    {
        'group_1': 'virus protein',
        'type_1': ['ORF4b'],
        'host_list_1': ['Middle East respiratory syndrome-related coronavirus'],
        'group_2': 'host protein',
        'type_2': ['STAT1', 'STAT2', 'IRF9', 'NF-kB', 'IRF3', 'IRF7', 'MAVS', 'TBK1'],
        'host_list_2': ['Homo sapiens', 'Camelus dromedarius', 'Mus musculus', 'Rattus norvegicus'],
        'relation': 'interacts'
    },

    {
        'group_1': 'virus protein',
        'type_1': ['Spike'],
        'host_list_1': ['Middle East respiratory syndrome-related coronavirus'],
        'group_2': 'host protein',
        'type_2': ['DPP4'],
        'host_list_2': ['Homo sapiens', 'Camelus dromedarius', 'Mus musculus', 'Rattus norvegicus'],
        'relation': 'interacts'
    },

    {
        'group_1': 'virus protein',
        'type_1': ['Spike'],
        'host_list_1': ['Severe acute respiratory syndrome coronavirus 2'],
        'group_2': 'host protein',
        'type_2': ['ACE2'],
        'host_list_2': ['Homo sapiens', 'Felis catus', 'Macaca mulatta', 'Canis lupus familiaris',
                        'Rhinolophus ferrumequinum', 'Mesocricetus auratus'],
        'relation': 'interacts'
    },

    {
        'group_1': 'virus protein',
        'type_1': ['Spike'],
        'host_list_1': ['Severe acute respiratory syndrome-related coronavirus'],
        'group_2': 'host protein',
        'type_2': ['ACE2'],
        'host_list_2': ['Homo sapiens', 'Felis catus', 'Rhinolophus ferrumequinum'],
        'relation': 'interacts'
    },

    {
        'group_1': 'virus protein',
        'type_1': ['Spike'],
        'host_list_1': ['Human coronavirus NL63'],
        'group_2': 'host protein',
        'type_2': ['ACE2'],
        'host_list_2': ['Homo sapiens', 'Rhinolophus ferrumequinum'],
        'relation': 'interacts'
    },

    {
        'group_1': 'virus',
        'type_1': ['virus'],
        'host_list_1': ['Severe acute respiratory syndrome-related coronavirus'],
        'group_2': 'host',
        'type_2': ['host'],
        'host_list_2': ['Homo sapiens', 'Felis catus', 'Rattus norvegicus', 'Rhinolophus ferrumequinum',
                        'Mus musculus'],
        'relation': 'infects'
    },

    {
        'group_1': 'virus',
        'type_1': ['virus'],
        'host_list_1': ['Middle East respiratory syndrome-related coronavirus'],
        'group_2': 'host',
        'type_2': ['host'],
        'host_list_2': ['Homo sapiens', 'Camelus dromedarius', 'Mus musculus', 'Rattus norvegicus'],
        'relation': 'infects'
    },

    {
        'group_1': 'virus',
        'type_1': ['virus'],
        'host_list_1': ['Severe acute respiratory syndrome coronavirus 2'],
        'group_2': 'host',
        'type_2': ['host'],
        'host_list_2': ['Homo sapiens', 'Felis catus', 'Macaca mulatta', 'Canis lupus familiaris',
                        'Rhinolophus ferrumequinum', 'Mesocricetus auratus'],
        'relation': 'infects'
    },

    {
        'group_1': 'virus',
        'type_1': ['virus'],
        'host_list_1': ['Human coronavirus HKU1'],
        'group_2': 'host',
        'type_2': ['host'],
        'host_list_2': ['Homo sapiens', 'Mus musculus', 'Rattus norvegicus', 'Rhinolophus ferrumequinum'],
        'relation': 'infects'
    },

    {
        'group_1': 'virus',
        'type_1': ['virus'],
        'host_list_1': ['Human coronavirus OC43'],
        'group_2': 'host',
        'type_2': ['host'],
        'host_list_2': ['Homo sapiens', 'Bos taurus', 'Mus musculus', 'Rattus norvegicus'],
        'relation': 'infects'
    },

    {
        'group_1': 'virus',
        'type_1': ['virus'],
        'host_list_1': ['Human coronavirus NL63'],
        'group_2': 'host',
        'type_2': ['host'],
        'host_list_2': ['Homo sapiens', 'Rhinolophus ferrumequinum'],
        'relation': 'infects'
    },

    {
        'group_1': 'virus',
        'type_1': ['virus'],
        'host_list_1': ['Human coronavirus 229E'],
        'group_2': 'host',
        'type_2': ['host'],
        'host_list_2': ['Homo sapiens', 'Rhinolophus ferrumequinum'],
        'relation': 'infects'
    },
]

protein_function_data = {
    'ACE2': 'virus receptor activity',
    'DPP4': 'virus receptor activity',
    'IRF3': 'interferon regulatory factor',
    'IRF7': 'interferon regulatory factor',
    'IRF9': 'interferon regulatory factor',
    'MAVS': 'interferon regulatory factor',
    'MDA5': 'interferon regulatory factor',
    'NF-kB': 'interferon regulatory factor',
    'PRKRA': 'interferon regulatory factor',
    'RIG-I': 'interferon regulatory factor',
    'STAT1': 'interferon regulatory factor',
    'STAT2': 'interferon regulatory factor',
    'TBK1': 'interferon regulatory factor',
    'Spike': 'virus receptor activity',
    'M protein': 'interferon regulatory factor',
    'N protein': 'interferon regulatory factor',
    'nsp1': 'interferon regulatory factor',
    'nsp15': 'interferon regulatory factor',
    'ORF3b': 'interferon regulatory factor',
    'ORF4a': 'interferon regulatory factor',
    'ORF4b': 'interferon regulatory factor',
    'ORF6': 'interferon regulatory factor',
    'PLpro': 'interferon regulatory factor',
    'host': '',
    'virus': ''
}
