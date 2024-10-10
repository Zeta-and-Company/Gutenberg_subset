# Gutenberg_subset
This is a selected subset of the [Gutenberg corpus](https://github.com/pgcorpus/gutenberg]). It has 1752 texts. The criteria for selection are 
- English text
- authoryearofdeath between 1900 and 1969
- no less than 10,000 words in length
- falls into one and only one of the four subgenres (texts belonging to more than one subgenres are not included):
  - science fiction
  - historical fiction
  - love story
  - detective fiction

In addtion, topic models are trained on the subset using MALLET and the distinctive topics for each subgenres are identified using the Welch's T-Test. 
