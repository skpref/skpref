Module documentation
=====================

Tasks
-------

.. autosummary::
   :nosignatures:

   skpref.task.ChoiceTask
   skpref.task.PairwiseComparisonTask


Models
-------

.. autosummary::
   :nosignatures:

   skpref.base.Model
   skpref.random_utility.BradleyTerry
   skpref.base.ClassificationReducer


Model Selection
----------------
.. autosummary::
   :nosignatures:

   skpref.model_selection.GridSearchCV

Metrics
--------

.. autosummary::
   :nosignatures:

   skpref.metrics.true_positives
   skpref.metrics.true_negatives
   skpref.metrics.false_positives
   skpref.metrics.false_negatives
   skpref.metrics.accuracy
   skpref.metrics.recall
   skpref.metrics.f1_score
   skpref.metrics.log_loss
   skpref.metrics.log_loss_compare_with_t_test


Tasks
-------

ChoiceTask
^^^^^^^^^^^^
.. automodule:: skpref.task
    :members: ChoiceTask

PairwiseComparisonTask
^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: skpref.task
    :members: PairwiseComparisonTask


Models
-------
All models inherit from base.models the fit_task and predict_task methods. This helps maintain consistency within the
package for how reduction and aggregation is treated.

.. autoclass:: skpref.base.Model
    :members: fit_task, predict_task

BradleyTerry
^^^^^^^^^^^^^

.. autoclass:: skpref.random_utility.BradleyTerry
    :members: rank_entities, predict_choice_task, predict_proba_task

ClassificationReducer
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: skpref.base.ClassificationReducer
    :members: __init__

Model Selection
----------------

GridSearchCV
^^^^^^^^^^^^^

.. autoclass:: skpref.model_selection.GridSearchCV
    :members: fit_task, inspect_results, rank_entities, predict_task, predict_proba_task, predict_choice_task

Metrics
--------

.. autofunction:: skpref.metrics.true_positives
.. autofunction:: skpref.metrics.true_negatives
.. autofunction:: skpref.metrics.false_positives
.. autofunction:: skpref.metrics.false_negatives
.. autofunction:: skpref.metrics.accuracy
.. autofunction:: skpref.metrics.recall
.. autofunction:: skpref.metrics.f1_score
.. autofunction:: skpref.metrics.log_loss
.. autofunction:: skpref.metrics.log_loss_compare_with_t_test
