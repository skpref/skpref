import pandas as pd

# SUBSET CHOICE PRIMARY TABLE
SUBSET_CHOICE_TABLE = pd.DataFrame(
    {'choice': [[512709, 529703, 696056], [723354]],
     'alternatives': [[512709, 529703, 696056, 490972,  685450, 5549502],
                      [550707, 551375, 591842, 601195, 732624, 778197, 813892,
                       817040, 576214, 673995, 723354]]}
)

# SUBSET CHOICE SECONDARY TABLE
SUBSET_CHOICE_FEATS_TABLE = pd.DataFrame(
    {'ID': [490972,  512709,  529703,  550707,  551375,  576214,  591842,
            601195,  673995,  685450,  696056,  723354,  732624,  778197,
            813892,  817040, 5549502],
     'feat1': [6, 6, 6, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
     }
)

# SUBSET CHOICE PRIMARY TABLE WITH MERGE COLUMN (SEASON)
SUBSET_CHOICE_TABLE_season = pd.DataFrame(
    {'choice': [[512709, 529703, 696056], [723354]],
     'alternatives': [[512709, 529703, 696056, 490972,  685450, 5549502],
                      [550707, 551375, 591842, 601195, 732624, 778197, 813892,
                       817040, 576214, 673995, 723354]],
     'season': [7, 8]}
)

# SUBSET CHOICE SECONDARY TABLE WITH MERGE COLUMN (SEASON)
SUBSET_CHOICE_FEATS_TABLE_season = pd.DataFrame(
    {'ID': [490972,  512709,  529703,  550707,  551375,  576214,  591842,
            601195,  673995,  685450,  696056,  723354,  732624,  778197,
            813892,  817040, 5549502,
            490972, 512709, 529703, 550707, 551375, 576214, 591842,
            601195, 673995, 685450, 696056, 723354, 732624, 778197,
            813892, 817040, 5549502
            ],
     'feat1': [6, 6, 6, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     'season': [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
     }
)

# SUBSET CHOICE DATA WITH STRINGS AS ALTERNATIVES
DATA_het = pd.DataFrame({'alternatives': [['B', 'C'], ['A', 'B'], ['D', 'C'],
                                          ['C', 'D', 'A']],
                         'result': [['C', 'B'], ['A'], ['D'], ['D']]})

# PAIRWISE COMPARISON PRIMARY TABLE WITH CORRESPONDING COLUMN
DATA = pd.DataFrame({'ent1': ['C', 'B', 'C', 'D'],
                     'ent2': ['B', 'A', 'D', 'C'],
                     'result': [1, 0, 0, 1]})


# PAIRWISE COMPARISON TABLE WITH CORRESPONDING COLUMN AND MERGE COLUMN (SEASON)
DATA_c_several_merge_cols = pd.DataFrame(
    {'ent1': ['C', 'B', 'C', 'D', 'A', 'B'],
     'ent2': ['B', 'A', 'D', 'C', 'B', 'A'],
     'result': [1, 0, 0, 1, 1, 0],
     'season': [1, 1, 1, 1, 2, 2]})

# PAIRWISE COMPARISON IN SUBSET CHOICE FORMAT
DATA2 = pd.DataFrame({'alternatives': [['B', 'C'], ['A', 'B'], ['D', 'C'],
                                       ['C', 'D']],
                     'result': ['C', 'A', 'D', 'D']})


# PAIRWISE COMPARISON ATTRIBUTES
ENT1_ATTRIBUTES = pd.DataFrame(
    {'ent1': ['A', 'B', 'C', 'D'],
     'feat1': [1, 11, 12, 15]}
)

# PAIRWISE COMPARISON ATTRIBUTES WITH MERGE COLUMN
ENT1_ATTRIBUTES_several_merge_cols = pd.DataFrame(
    {'ent1': ['A', 'B', 'C', 'D', 'A', 'B'],
     'feat1': [1, 11, 12, 15, 2, 5],
     'season': [1, 1, 1, 1, 2, 2]}
)

# DISCRETE CHOICE DATA
discrete_choice_data = pd.DataFrame({
    'alt': [['a', 'b', 'c'], ['a', 'b'], ['c', 'd'], ['d']],
    'choice': ['b', 'a', 'c', 'd']
})

discrete_choice_data2 = pd.DataFrame({
    'alt': [['a', 'b', 'c'], ['a', 'b'], ['c', 'd'], ['c','d']],
    'choice': ['b', 'a', 'c', 'd']
})

discrete_choice_data_secondary_covariates = pd.DataFrame({
    'alternative': ['a', 'b', 'c', 'd'],
    'covariate': [1, 2, 3, 4]
})
