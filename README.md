# BE-Project

This project can automatically infer citation style from a reference string. The classifier is a Logistic Regression model trained on 90,000 reference strings. The following citation styles are supported by default:

  * acm-sig-proceedings
  * american-chemical-society
  * american-chemical-society-with-titles
  * american-institute-of-physics
  * american-sociological-association
  * apa
  * bmc-bioinformatics
  * chicago-author-date
  * elsevier-without-titles
  * elsevier-with-titles
  * harvard3
  * ieee
  * iso690-author-date-en
  * modern-language-association
  * springer-basic-author-date
  * springer-lecture-notes-in-computer-science
  * vancouver
  * unknown

## Data

The Project contains [two datasets]: training set and test set. Each of them contains a sample of 5,000 DOIs formatted in 17 citation styles (listed above), which gives 85,000 reference strings. Both datasets were generated automatically using Crossref REST API.

## Models

The model was trained on the training dataset. Before the training, the dataset was cleaned and enriched with random noise. 5,000 strings with "unknown" style were also generated and added to the dataset.

## Evaluation

`styleclass_evaluate` script can be used to evaluate exisitng model on a test set, in terms of accuracy.

The accuracy of the default model estimated on our test set is 95%.
