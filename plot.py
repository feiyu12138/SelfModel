import matplotlib.pyplot as plt
import pandas as pd

# Load the train_loss.csv and val_loss.csv files
train_loss = pd.read_csv('train_loss.csv')
val_loss = pd.read_csv('val_loss.csv')
train_loss_base = pd.read_csv('train_loss_ori.csv')
val_loss_base = pd.read_csv('val_loss_ori.csv')

# Extract the loss values from the dataframes
train_loss_values = train_loss['Value'].values
val_loss_values = val_loss['Value'].values
train_loss_values_base = train_loss_base['Value'].values
val_loss_values_base = val_loss_base['Value'].values

# Create a plot
plt.plot(train_loss_values, label='Train Loss for modulated model')
plt.plot(train_loss_values_base, label='Train Loss for base model')
plt.plot(val_loss_values, label='Validation Loss for modulated model')
plt.plot(val_loss_values_base, label='Validation Loss for base model')


# Add labels and title to the plot
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')

# Add a legend
plt.legend()

# Show the plot
plt.savefig('loss_curve.png')
