# Emir Module

This module's purpose is to provide a tool enabling embedder comparison.

To perform this comparison, one might have access to a set of embeddings originating from the different models to compare.
A wrapper enabling this evaluation is provided in [embedder_evaluator.py](embedder_evaluator.py), based on the estimator available in [knife_estimator.py](estimators/knife_estimator.py).


*Different implementations can be found in the domain specifics scripts, notably in [estimation_utils.py](../molecule/utils/estimator_utils/estimation_utils.py), reusing the different marginal kernels across estimations to reduce the time complexity.*




