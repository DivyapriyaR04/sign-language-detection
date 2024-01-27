import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

# Find the maximum length of sequences
max_length = max(len(seq) for seq in data)

# Pad sequences to the same length
data_padded = [seq + [0] * (max_length - len(seq)) for seq in data]

# Convert to NumPy array
data = np.asarray(data_padded)
labels = np.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save only the model parameters
model_params = {
    'n_estimators': model.n_estimators,
    'max_depth': model.max_depth,
    # Add other relevant parameters here
}

# Save the model parameters
# Save the model parameters
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

