"""
Graph 1 — Overall Metrics
File: graph1_overall_metrics.py
Run:  python graph1_overall_metrics.py
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'BLEU Score']
values  = [87.63,       97.49,       87.69,    92.33,      71.34]
colors  = ['#1D9E75',  '#185FA5',  '#EF9F27', '#534AB7',  '#D85A30']

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor('#F8F8F6')
ax.set_facecolor('#F8F8F6')

bars = ax.bar(metrics, values, color=colors, width=0.52,
              edgecolor='white', linewidth=2, zorder=3)

for bar, val in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.9,
            f'{val:.2f}%', ha='center', va='bottom',
            fontsize=14, fontweight='bold', color='#2C2C2A')

for y in [20,40,60,80,100]:
    ax.axhline(y, color='#DDDBD4', linewidth=0.7, linestyle='--', zorder=1)

ax.set_ylim(0, 115)
ax.set_ylabel('Score (%)', fontsize=13, color='#555552')
ax.set_title('Coreference Resolution — Overall Evaluation Metrics\n5159 samples  ·  11 Indian languages',
             fontsize=14, fontweight='bold', color='#2C2C2A', pad=14)
ax.tick_params(axis='x', labelsize=13, colors='#333330')
ax.tick_params(axis='y', labelsize=11, colors='#888780')
ax.set_yticks(range(0,101,20))
ax.spines[['top','right','left','bottom']].set_visible(False)
plt.tight_layout()
plt.savefig('graph1_overall_metrics.png',
            dpi=180, bbox_inches='tight', facecolor='#F8F8F6')
print("Saved graph1")
