# Intent-classification-assignment

## Objective:

In this home test, the task was to create an intent classification model to understand and predict the intentions of renters' messages using the provided training data.

 
## Data:

The training set consists of messages from potential renters. These messages can be responses to our messages or initial communications from the renter. Our messages typically ask about the number of desired bedrooms and the desired move-in date, or offer tour dates for property viewing. In each conversation, there might be “noise” in the message, noise that does not relate to any one of the intentions. 


## Task: 

Train a classifier to predict the following four intentions:


- SC - Date: The renter indicates the date or time they can or want to come for a tour.

- SC - Not Date: The renter indicates the date or time they can’t or don’t want to come for a tour.

- Prospects - Unit pricing: The renter asks or shares information about the rent/price of the unit.

- Prospects - GC - Bed count: The renter indicates the number of bedrooms they want.

 
## Dataset Features:

The training data contains eight columns:

1. Stamp: Timestamp of the conversation message from the renter.

2. Original_text: The renters’ message.

3. Context: Parameter indicating the question we asked the renter in the previous message.

     - "Select" = We offered a tour date in the previous message.

     - "bedcountqualification" = We asked the renter how many bedrooms they need in the previous message.

4. InitialSMSInteraction: A boolean parameter indicating if the renter initiated the conversation.


Labels:

5. SC - Date
 
6. SC - Not Date

7. Prospects - Unit pricing

8. Prospects - GC - Bed count


## Requirements

Required packages:

1. pandas

2. openpyxl

3. transformers

4. torch

5. datasets

6. scikit-learn

7. matplotlib


## Instructions (Run locally)
<pre>
pip install -r requirements.txt
python main.py
</pre>


## Results
The results are available in the following document- https://docs.google.com/document/d/1slaj0zE2nOCP-xkIQDiirvxqyUkRdBi2-K4-Ui7WJsM/edit?usp=sharing.
