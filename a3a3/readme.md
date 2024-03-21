Task Definition - Given two sentences, calculate the similarity between these two sentences.
The similarity is given as a score ranging from 0 to 5.
Train datapoint examples -
score sentence1 sentence2
4.750 A young child is riding a horse. A child is riding a horse.
2.400 A woman is playing the guitar. A man is playing guitar.
The dataset is already divided into training and validation sets in the files - ‘train.csv’ and
‘dev.csv’, respectively. Both files are given to you in the zip file attached to the assignment.
Please note that it is tab-separated. A testing file excluding the score field will be provided to
you during the demo to run inference on. You are required to create dataset classes and data
loaders appropriately for your training and evaluation setups.
For this task, you are required to implement three setups:

● Setup 1A - You are required to train a BERT model (google-bert/bert-base-uncased ·
Hugging Face) using HuggingFace for the task of Text Similarity. You are required to
obtain BERT embeddings while making use of a special token used by BERT for
separating multiple sentences in an input text and an appropriate linear layer or setting
of BertForSequenceClassification (BERT) framework for a float output. Choose a
suitable loss function. Report the required evaluation metric on the validation set.
(10 marks)
● Setup 1B - You are required to make use of the Sentence-BERT model
(https://arxiv.org/pdf/1908.10084.pdf) and the SentenceTransformers framework
(Sentence-Transformers). For this setup, make use of the Sentence-BERT model to
encode the sentences and determine the cosine similarity between these embeddings
for the validation set. Report the required evaluation metric on the validation set.
(5 marks)
● Setup 1C - In this setup, you must fine-tune the Sentence-BERT model for the task of
STS. Make use of the CosineSimilarityLoss function (Losses — Sentence-Transformers
documentation). Report the required evaluation metric on the validation set—reference:
Semantic Textual Similarity — Sentence-Transformers documentation. You must train for
at least two epochs and surpass the performance of Setup 2B. (15 marks)
You must save and submit your model checkpoints for 1A and 1C in an appropriate format.
Note - For setups 1B and 1C, the data has a score out of 5. However, cosine similarity returns a
value between 0 and 1. Hence, you must appropriately scale the cosine similarity to the score
column's scale before evaluation. Hint: You may also be required to scale down the score to a
scale of 1 for training the sentence transformers.
Evaluation Metrics - Pearson Correlation
Report - (5 marks)
● Generate the following plots for Setup 1A and 1C:
a) Loss Plot: Training Loss and Validation Loss V/s Epochs
b) Analyse and Explain the plots obtained as well
● Provide a brief comparison and explanation for the performance differences between the
three setups in the report.
● Provide all evaluation metrics for all the setups in your report pdf.
Demo - During the demo, you will receive a testing file, excluding the score field, which will be
provided for you to run an inference during the demo. You must create an inference pipeline that
can load your model checkpoint for setup 1C, read the data in the given test file, generate
predictions of the text-similarity for the given sentences, and generate a CSV file in the format of
‘sample_demo.csv’. You must submit the CSV file with your predictions to the TA during the
demo, who will then calculate and report your test set evaluation metrics.

(10 marks)
