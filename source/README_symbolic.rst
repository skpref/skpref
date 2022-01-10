|Build Status| |Coverage|


Welcome to skpref
=====================

Skpref is a python package that aims to create an infrastructure for supervsied preference models. The long-term vision
of this repository is that it becomes a comprehensive supervised machine learning interface for practitioners, that has
the familiar interface and many developer principles of the popular package scikit-learn, but designed in a way to better
suit preference models.

A preference contains the following components:

* A set of alternatives. This can be anything ranging from, products in a store, options of transportation, homes to rent and many more.
* A question over the set of alternatives, which can be: "Which products would you rather buy from this list of products?", "Which house would you rather rent", "Would you take a cab or a bus?", "Which team has won the match?", "Which box is the heaviest?" or "How would you rank these three things?".
* A relation expressed over the set of alternatives for a given question. This can have many forms for example to the question "Which products would you rather buy?" the relation can be a choice in the form of a selection of a subset of the alternatives. Or to the question "Which Sushi was the best tasting?" a relation expressed could be fully ranked order of all the Sushis that were tried from best to worst. Currently skpref has been developed with the following applications in mind: pairwise comparisons (choose one of two, or tie, e.g. football match results), discrete choices (choose one of many, e.g. choice of method of transportation), subset choices (choose many from many, e.g. shopping in at a grocery store). We believe it could be expandable to partial orders (ranking with ties e.g. start rating system) and full orders (full ranks).
* A deciding process or choice maker. Someone, something or a process that is presented the set of alternatives and generates a relation over this set of alternatives based on the question asked over the set of alternatives.

skpref is designed to hold supervised  learning models where the ground truth is the relation expressed over the set of
alternatives. In its current form, it is a demonstration of the architectural ideas that in our opinion would facilitate
these kinds of models:

* The user has the ability to express relations of: pairwise comparison, discrete choice and subset choice. This is done via the SubsetPosetVector which is explained in the next section. The interface was designed to be expanded to hold partial and full orders (ranks) also, and we invite the community to do so.
* It contains architectural demos of how the package could be interfaced with the scikit-learn infrastructure, such as GridSearchCV
* It is designed to be used with a relational data set up thereby inviting researchers to figure out efficiency gains in numerical optimisation that the relational set-up may hold. To do this there's an additional type of object which would be new to previous scikit-learn users, which is the Task object.
* It has been designed in a way to facilitate reduction and aggregation a popular technique in the field.


Acknowledgements
-------------------
Although the current state of the package is a humble demo of the original conceptualisation. Even this could not have
been possible without the tireless guidance and brilliant ideas of Dr. Franz Kiraly and Dr. Ioanna Manolopoulou and the
energetic brainstorming sessions with Karlson Pfannschmidt and Pritha Gupta.


Future contributions
---------------------
We welcome anyone who would like to continue working on this concept. If you would like to contribute to this package
please reach out to Istvan Papp at istvan.papp.16@ucl.ac.uk


Models in the package
======================
Currently skpref contains only implementations of the Bradley-Terry model for pairwise comparisons and the more generalised
Luce formulation for discrete choice, which can be augmented with covariates. It achieves this by interfacing with the
pylogit_ and choix_ packages.

Novelties in skpref to the scikit-learn user
===============================================

The task based interface
-------------------------
The main object with which end users will interact and the one that is the furthest step away in intuition from the scikit-learn
set-up is the Task object. The purpose of the Task object's main purpose is to containerise key aspects of the data that is
being modelled, such as, what are the dataframes that will be used, what is the ground truth, what is the set of alternatives
for each observation and how these all link together.

Tasks are specific to relations expressed, therefore, at the moment skpref has a demo of a PairwiseComparisonTask,
a ChoiceTask and a ClassificationReducer. Between these the interface covers Pairwise Comparisons, Discrete Choice and reductions
from both of those to classification problems, and also a reduction from Discrete Choice to Pairwise Comparisons.

We justify asking end users to learn how to use the task based interface with the following:

* It saves time and energy compared to having to merge normalised tables together into denormalised tables. In the future
  we believe it should be possible to create computational efficiencies from using normalised tables, so we designed an
  an interface that allows for this already.
* It allows reduction and aggregation to happen in the back-end of the code, rather than putting the onus on the user to
  create data transformations that may not be trivial.

Suppose you have two tables that you would like to use for analysis, one contains information about choices that people
made:

+---------+-----------------+
| Choices |   Alternatives  |
+=========+=================+
|   'A'   |    ['A', 'B']   |
+---------+-----------------+
|   'C'   | ['A', 'B', 'C'] |
+---------+-----------------+
|   'B'   | ['A', 'B', 'C'] |
+---------+-----------------+
|   'A'   | ['A', 'B', 'C'] |
+---------+-----------------+
|   'B'   |    ['A', 'B']   |
+---------+-----------------+

and the other features that describe each alternative:

+-------------+-----------+-----------+-----------+
| Alternative | Feature 1 | Feature 2 | Feature 3 |
+=============+===========+===========+===========+
|     'A'     | 9         |     5     | 9         |
+-------------+-----------+-----------+-----------+
|     'B'     | 3         |     4     | 6         |
+-------------+-----------+-----------+-----------+
|     'C'     | 7         |     4     | 10        |
+-------------+-----------+-----------+-----------+

Without the task based set up a user would have to combine the choice table with the alternative table to be able to use
the features describing the alternatives in a model. However, apart from this being a repetitive task, it might even be
non-trivial in difficulty and different models and packages might deal with different ways of combining this data. The
task-based set up allows flexibility on handling the data differently for different models.

To allow for this in skpref all model object have a train_task and a predict_task function (and a predict_proba_task
function, where appropriate) to take the burden off from the user of having to create these joint tables.

A task for this data would be set up in the following way:

.. code:: python

  from skpref.task import SomeTypeOfTask
  from skpref.model_type_folder import some_type_of_model

  example_train_task = SomeTypeOfTask(
    primary_table=test_example_choice_table,
    primary_table_alternatives_names='Alternatives',
    primary_table_target_name ='Choices',
    features_to_use=['Feature 1', 'Feature 2', 'Feature 3'],
    secondary_table=example_alternative_level_feature_table
    secondary_to_primary_link={'Alternative': 'Alternatives'}
  )

  example_test_task = SomeTypeOfTask(
    primary_table=train_example_choice_table,
    primary_table_alternatives_names='Alternatives',
    features_to_use=['Feature 1', 'Feature 2', 'Feature 3'],
    secondary_table=example_alternative_level_feature_table
    secondary_to_primary_link={'Alternative': 'Alternatives'}
  )

  my_initalised_model = some_type_of_model()
  my_initialised_model.fit_task(example_train_task)
  my_outcome_predictions = my_initialised_model.predict_task(example_test_task)
  my_probabilistic_predictions = my_initialised_model.predict_proba_task(example_test_task)

Below we will show examples of pairwise comparisons and discrete choices and show how the task based interface can be used
for setting up the models.

SubsetPosetVectors
-------------------
Understanding SubsetPosetVectors will be useful for those who are considering becoming future developers in for the skpref
package. A SubsetPosetVector is an object that has been designed to contain relations where the nature of the relation is to select
a subset of the list of alternatives. As such it contains two numpy arrays :code:`top_input_data` and :code:`boot_input_data`
to indicate the alternatives chosen and not chosen respectively. Each SubsetPosetVector represents the choices and discards
of an entire data set, so these numpy arrays, may include ragged-nested arrays also, we expand on these in below. The end user
is not expected to be interacting much with SubsetPosetVectors, however, it is the internal representation of the data
that models use and it is the data type that is returned in outcome predictions.

Types of relations that can be currently modelled in skpref
==================================================================================================
In this section we discuss three types of relations that are currently supported by the infrastructure of skpref,
pairwise comparisons, discrete choice and subset choice. The way the package currently deals with such relations is via
the SubsetPosetVector. Below are more detailed descriptions of these types of relations, and examples of how the
SubsetPosetVector represents these. We do not expect end users to use SubsetPosetVectors to read in their data, as the main
use right now is a back-end representation of the data. This section, however, provides a useful discussion on what types
of relations the SubsetPosetVector supports.

Pairwise Comparisons
---------------------
For Pairwise comparisons only two alternatives are presented to decision makers or decision processes, examples could be
football matches, where there are only two teams playing at a time and either one team wins or there's a draw. As another example,
consider the table below which contains made up examples of US college basketball matches. In the first column we identify
the winning team, and in the second column we identify the two teams that played. The defining characteristic of pairwise
comparisons is that the number of alternatives presented is always two.

+--------------+------------------------+
| Winning team |         Matchup        |
+==============+========================+
|  'Virginia'  | ['Purdue', 'Virginia'] |
+--------------+------------------------+
|   'Auburn'   | ['Auburn', 'Kentucky'] |
+--------------+------------------------+
|  'MI State'  |  ['MI State', 'Duke']  |
+--------------+------------------------+

Setting up a PairwiseComparisonTask
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For the table above we would have the following set up for a PairwiseComparisonTask

.. code:: python

  from skpref.task import PairwiseComparisonTask
  example_PCTask = PairwiseComparisonTask(
    primary_table=basketball_data,
    primary_table_alternatives_names='Matchup',
    primary_table_target_name ='Winning Team',
    features_to_use=None
  )

The PairwiseComparisonTask automatically sets up the SubsetPosetVector which in this case would be stored in the object
:code:`example_PCTask.subset_vec`.

Note that pairwise comparison tasks also often follow the below data structure

+------------+-------------+------------+
|   Team 1   | Team 2      | Team 1 won |
+============+=============+============+
|  'Purdue'  | 'Virginina' | 0          |
+------------+-------------+------------+
|  'Auburn'  | 'Kentucky'  | 1          |
+------------+-------------+------------+
| 'MI State' | 'Duke'      | 1          |
+------------+-------------+------------+

In which case the PairwiseComparisonTask can be set up this way:

.. code:: python

  from skpref.task import PairwiseComparisonTask
  example_PCTask = PairwiseComparisonTask(
    primary_table=basketball_data,
    primary_table_alternatives_names=['Team 1', 'Team 2'],
    primary_table_target_name ='Team 1 won',
    target_column_correspondence = 'Team 1'
    features_to_use=None
  )

Now that the PairwiseComparisonTask is set up, it is possible to fit a model, say a Bradley-Terry model:

.. code:: python

  from skpref.random_utility import BradleyTerry
  my_bt_model = BradleyTerry()
  my_bt_model.fit_task(example_PCTask)

We can also use the :code:`my_bt_model.predict_task()` or the :code:`my_bt_model.predict_proba_task()` to predict the
outcomes. Since the Bradley-Terry model has a ranking output too, we can query this by running :code:`mybt.rank_entities()`.

Setting up a SubsetPosetVectors for Pairwise Comparisons
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The way the SubsetPosetVector would represent this information is the following way.

.. code:: python

  from skpref.data_processing import SubsetPosetVec
  example_pairwise_comparison_vec = SubsetPosetVec(
      top_input_data=np.array(['Virginia', 'Auburn', 'MI State']),
      boot_input_data=np.array(['Purdue', 'Kentucky', 'Duke'])
      )


Discrete Choice
----------------
In a discrete choice the deciding process or decision maker only chooses one and only one alternative from the set of alternatives.
It is used widely for modelling choices made in transportation, where an individual can naturally be only taking one mode
of transportation at a time (it is rare to see someone riding a bike whilst driving a car). Below we show an example of this data
where each row is a commuting decision, the first column is the mode of transport taken, and the second column is the available options.

+-------------------------------+-----------------------------+
| Chosen Mode of Transportation | Options                     |
+===============================+=============================+
|           'bicycle'           | ['train', 'bicycle']        |
+-------------------------------+-----------------------------+
|             'car'             | ['train', 'bicycle', 'car'] |
+-------------------------------+-----------------------------+
|            'train'            | ['train', 'bicycle', 'car'] |
+-------------------------------+-----------------------------+
|             'car'             | ['train', 'bicycle', 'car'] |
+-------------------------------+-----------------------------+
|            'train'            | ['train', 'bicycle']        |
+-------------------------------+-----------------------------+

Setting up a ChoiceTask
^^^^^^^^^^^^^^^^^^^^^^^^^^
For the table above we would have the following set up for a ChoiceTask

.. code:: python

  from skpref.task import ChoiceTask
  example_choice_task = ChoiceTask(
    primary_table=public_transport_data,
    primary_table_alternatives_names='Options',
    primary_table_target_name ='Chosen Mode of Transportation',
    features_to_use=None
  )

Since currently skpref does not contain any discrete choice models, we can only show an example in which the problem is
reduced to a pairwise comparison. We invite the community to build discrete choice models for skpref.

Reduction and aggregation of Discrete choices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the user would want to run now a reduction to pairwise comparisons, and run a Bradley-Terry model on this discrete
choice data, the code would look exactly the same as above for pairwise comparison models

.. code:: python

  from skpref.random_utility import BradleyTerry
  my_bt_model = BradleyTerry()
  my_bt_model.fit_task(example_choice_task)

The models in skpref need to detect what type of task is being passed to the model and then perform the reduction via
the functions available in the subset-poset vector which is generated in every task, see below for an example on the
pairwise reduction.

When skpref is expanded to contain discrete choice models also, users will be able to run fitting and prediction on both
reduced and same-level models.

Currently there are two ways to aggregate the Bradley-Terry model in skpref. One of them is to insert the learned parameters
in the Luce formulation, the other is via the Independent Transitive method. For Bradley-Terry the default setting is
via the Luce formulation and code for running both aggregations would look like the following:

.. code:: python

  # predicting the probability of taking a car
  agg_luce = my_bt_model.predict_proba(example_choice_task, ['Car'])
  agg_indep_trans = my_bt_model.predict_proba(example_choice_task, ['Car'], aggregation_method='independent transitive')

Setting up a SubsetPosetVectors for Discrete Choice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The way the SubsetPosetVector would represent this information is the following way.

.. code:: python

  from skpref.data_processing import SubsetPosetVec
  example_pairwise_comparison_vec = SubsetPosetVec(
      top_input_data=np.array([np.array(['bicycle']), np.array(['car']),
                               np.array(['train']), np.array(['car']),
                               np.array(['train'])], dtype=object),
      boot_input_data=np.array([np.array(['train', 'bicycle']),
                               np.array(['train', 'bicycle', 'car']),
                               np.array(['train', 'bicycle', 'car']),
                               np.array(['train', 'bicycle', 'car']),
                               np.array(['train', 'bicycle'])
                               ], dtype=object)
  )

A useful function contained in the SubsetPosetVec object is that it can be used to create data reductions, for example,
if we wanted to reduce this discrete choice data to pairwise comparison we could use:

.. code:: python

  >>> example_pairwise_comparison_vec.pairwise_reducer()
  (      alt1     alt2  alt1_top
   0  bicycle    train         1
   1    train      car         0
   2      car  bicycle         1
   3  bicycle    train         0
   4      car    train         0
   5      car    train         1
   6  bicycle      car         0
   7    train  bicycle         1,
   array([0, 1, 1, 2, 2, 3, 3, 4]))

Where the first element that is returned is a pandas DataFrame that represents that data as pairwise comparisons and the
second element is a numpy array that is the index of the oringinal observation. For example rows one and two of the
pairwise comparison table all correspond to row one in the original table ('car' being chosen from ['car', 'train',
'bicycle'] note we initialise rows by 0). Thus allowing us to use pairwise comparison models also to work with this
discrete choice data. All of this is accomplished in the back end of the package through the Task object design.


.. |Build Status| image:: https://travis-ci.org/skpref/skpref.svg?branch=master
.. |Coverage| image:: https://coveralls.io/repos/github/skpref/skpref/badge.svg?branch=master&service=github
   :target: https://coveralls.io/github/skpref/skpref?branch=master

.. _pylogit: https://github.com/timothyb0912/pylogit
.. _choix: https://github.com/lucasmaystre/choix
