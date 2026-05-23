# plot_results.py
import matplotlib.pyplot as plt
import pandas as pd

# After running eval with --save results.csv:
# python evaluate_coreference_v2.py --csv ../dataset/context_dataset_with_gt.csv --save results.csv

df = pd.read_csv('results.csv')

# Progressive accuracy
df['cumulative_accuracy'] = df['passed'].expanding().mean() * 100

plt.figure(figsize=(10, 5))
plt.plot(df.index, df['cumulative_accuracy'])
plt.title('Progressive Accuracy of Coreference Resolution System')
plt.xlabel('Number of Evaluated Samples')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('progressive_accuracy.png', dpi=150)
plt.show()