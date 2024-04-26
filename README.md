# Propaganda-Technique-Classification-in-News-Articles
This repository contains the code and resources for a propaganda technique classification project aimed at identifying and categorizing propaganda techniques in news articles. The project leverages machine learning techniques and natural language processing (NLP) models to analyze and classify text data.

## Project Structure
### datasets/: Contains the dataset used for training and evaluation, along with any preprocessed or augmented data.

### models/: Includes our custom trained hugging face models for classification.

### notebooks/: Jupyter notebooks providing step-by-step walkthroughs of data preprocessing, model training, evaluation, and analysis.

## Detailed Steps

- We have used the SemEval2020 dataset for this project.
- Firstly Preprocessed and extracted data from text files.
- Four categories are selected: Loaded_Language, Exaggeration & Minimization (converted to Hyperbole), Flag-Waving (converted to Jingoism), and Doubt, while removing other categories from the original dataset.
- The resulting dataset, named final.csv which have roughly 3300 rows, is available in the Datasets folder and is utilized for training the distilbert text-classification model.
- The Project Dataset.csv which have roughly 500 rows, also uploaded in the Datasets folder, is utilized for training the zero-shot model. This dataset consists of 500 rows specifically curated for zero-shot model training, with approximately 100 rows manually selected for each label based on accurate labeling of the sentences.

## Model Results 

We followed 3 approaches

### Approach 1: Creating DTM + Random Forest

We created a DTM out of the preprocessed text that we had. Later in order to reduce the dimensions, we plotted cumulative explained variance curve to find the no. of components that cover 90% of variance in PCA. Finally we applied random forest model to check whether the data is really classifiable or not. We got an accuracy of 54% and F1 score of 50%

### Approach 2: Created Word Embeddings using FastText + Random Forest & (RNN & CNN)

Secondly, we followed another approach where we created word embeddings using FastText out of the text we had. Then again we build a Random Forest model which gave us an accuracy of 57% and F1 score of 52%. CNN and RNN gave us 48% which is pretty low.

### Approach 3: Fine-tuning Hugging Face Models

Lastly, in order to improve the model performance and to classify the propaganda techniques effectively, we finetuned hugging face models based on our dataset. The exact parameters used to finetune the model are mentioned in the python notebook file. After finetuning we received an accuracy junp of 82% in distilbert's text classification model and 89% in facebook's bart-large-mnli zero-shot classification model. Also, we did a straight face test on 40 samples from 4 propaganda techniques generated using ChatGpt out of which text-classification model correctly identified 26 and zero-shot model identified 29 which were pretty good given the number of data the models were trained on.

## Hugging face spaces link

We have also created a hugging face spaces for both the models which you can try. Below is the link to the hugging face spaces:

Zero-Shot Classification model: https://huggingface.co/spaces/karthikvarunn/karthikvarunn-skar_propaganda_v1.1

Text Classification model: https://huggingface.co/spaces/karthikvarunn/karthikvarunn-skar_propaganda_v3
