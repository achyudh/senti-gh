# Senti-GH
> Empirical modeling of developer sentiments through user discussions on GitHub.

## Getting Started:

### Prerequisites:
Apart from Tensorflow, the implementation made use of the classifiers and evaluation metrics implemented in the Scikit-learn library for Python, and Scikit-fuzzy package was used for the FCM algorithm. Word Tokenizer from the NLTK library was used to preprocess the corpora. To perform some linear algebra computations and store the intermediate results, the popular Numpy package was used. The Pandas package provided the necessary utilities for reading and storing the pre-processed text data.

### Data
The data being used for this project is stored in a separate repository here: https://github.com/achyudhk/Senti-GH-Data. The complete dataset has more than 3000 repositories and is not a part of the repository due to its size. Further, trained Word2Vec models are also not a part of the repository due to their size and are available upon request. 

## Contributing:
When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change. Ensure any install or build dependencies are removed before the end of the layer when doing a build. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters.

## License:
This project is licensed under the GNU General Public License v3.0 - see the LICENSE.md file for details

## References:
* GloVe: Global Vectors for Word Representation: https://nlp.stanford.edu/projects/glove/. From Pennington, Jeffrey, Richard Socher, and Christopher Manning. "Glove: Global vectors for word representation." Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014.
* Repo-Reaper Datatset: https://reporeapers.github.io/results/1.html. From Munaiah, Nuthan, et al. "Curating GitHub for engineered software projects." Empirical Software Engineering 22.6 (2017): 3219-3253.
