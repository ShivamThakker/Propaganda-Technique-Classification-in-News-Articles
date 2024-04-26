# Propaganda-Technique-Classification-in-News-Articles
This repository contains the code and resources for a propaganda technique classification project aimed at identifying and categorizing propaganda techniques in news articles. The project leverages machine learning techniques and natural language processing (NLP) models to analyze and classify text data.

## Project Structure
### datasets/: Contains the dataset used for training and evaluation, along with any preprocessed or augmented data.

### models/: Includes our custom trained hugging face models for classification.

### notebooks/: Jupyter notebooks providing step-by-step walkthroughs of data preprocessing, model training, evaluation, and analysis.

After preprocessing is done and data is extracted from the text files, we select only 4 categories i.e., Loaded_Language, Exaggeration & Minimization which is converted to Hyperbole, Flag-Waving which is converted to Jingoism and Doubt. After removing every other category of data from the original dataset, we get the dataset named final.csv which is available in Datasets folder. We then use this dataset for training distilbert text-classification model. 

We use Project Dataset.csv which is also uploaded in Datasets folder. This dataset consists of 500 rows which was curated specifically for training zero-shot model as it does not require a lot of labelled data to train the model. We manually selected 100 rows for each label which we think was accurately labelled as per the senetence. So for 5 labels we have roughly 500 rows which we used to train our zero shot model.
