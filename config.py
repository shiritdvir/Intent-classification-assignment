import os

data_path = 'C:\Documents\different_DS_home'
train_path = os.path.join(data_path, 'Different_DS_home_test.xlsx')
test_path = os.path.join(data_path, 'Different_test_set.xlsx')
train_validation_ratio = 0.3

results_path = 'results_3_epochs'
os.makedirs(results_path, exist_ok=True)
logs_path = os.path.join(results_path, 'logs')

labels = ['SC - Date', 'SC - Not Date', 'Prospects - Unit pricing', 'Prospects - GC - Bed count']
num_labels = len(labels)

model_name = "distilbert-base-uncased"
epochs = 3
batch_size = 32
weight_decay = 0.01
steps = 10
