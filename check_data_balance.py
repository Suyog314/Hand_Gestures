
import pandas as pd

df = pd.read_csv('hand_number_data.csv')
print("📊 Sample counts per label (0-9):\n")
print(df['label'].value_counts().sort_index())
