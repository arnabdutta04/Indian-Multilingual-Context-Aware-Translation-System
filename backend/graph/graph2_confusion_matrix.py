"""
Graph 2 — Confusion Matrix
File: graph2_confusion_matrix.py
Run:  python graph2_confusion_matrix.py
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Derived exactly from Precision=97.49%, Recall=87.69%, Total=5159
TP=3839; FP=99; FN=539; TN=682
matrix = np.array([[TP, FN], [FP, TN]])
labels = [['True Positive (TP)', 'False Negative (FN)'],
          ['False Positive (FP)','True Negative (TN)']]
clrs   = [['#1D9E75','#EF9F27'],['#D85A30','#185FA5']]
total  = 5159

fig, ax = plt.subplots(figsize=(8, 6.5))
fig.patch.set_facecolor('#F8F8F6')
ax.set_facecolor('#F8F8F6')

for i in range(2):
    for j in range(2):
        ax.add_patch(plt.Rectangle((j,1-i),1,1,color=clrs[i][j],alpha=0.88,zorder=2))
        ax.text(j+0.5,1.5-i+0.16,labels[i][j],
                ha='center',va='center',fontsize=12,color='white',fontweight='bold',zorder=3)
        ax.text(j+0.5,1.5-i-0.12,f'{matrix[i,j]:,}',
                ha='center',va='center',fontsize=28,color='white',fontweight='bold',zorder=3)
        ax.text(j+0.5,1.5-i-0.40,f'({matrix[i,j]/total*100:.2f}%)',
                ha='center',va='center',fontsize=11,color='white',alpha=0.92,zorder=3)

ax.set_xlim(0,2); ax.set_ylim(0,2)
ax.set_xticks([0.5,1.5])
ax.set_xticklabels(['Predicted POSITIVE','Predicted NEGATIVE'],
                   fontsize=12,color='#2C2C2A',fontweight='bold')
ax.set_yticks([0.5,1.5])
ax.set_yticklabels(['Actual NEGATIVE','Actual POSITIVE'],
                   fontsize=12,color='#2C2C2A',fontweight='bold')
ax.tick_params(length=0)
ax.spines[['top','right','left','bottom']].set_visible(False)
ax.set_title('Confusion Matrix\nPrecision 97.49%  ·  Recall 87.69%  ·  F1 92.33%',
             fontsize=13,fontweight='bold',color='#2C2C2A',pad=16)
plt.tight_layout()
plt.savefig('graph2_confusion_matrix.png',
            dpi=180,bbox_inches='tight',facecolor='#F8F8F6')
print("Saved graph2")
