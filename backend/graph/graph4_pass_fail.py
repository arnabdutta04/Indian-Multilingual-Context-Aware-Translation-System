"""
Graph 4 — Pass / Fail + TP FP FN TN
File: graph4_pass_fail.py
Run:  python graph4_pass_fail.py
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# EXACT numbers from eval run
passed=4521; failed=638; total=5159
TP=3839; FP=99; FN=539; TN=682

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.patch.set_facecolor('#F8F8F6')
fig.suptitle('Resolution Results — 5,159 Total Samples',
             fontsize=14, fontweight='bold', color='#2C2C2A', y=1.01)

# Left pie
ax1 = axes[0]; ax1.set_facecolor('#F8F8F6')
wedges, texts, autotexts = ax1.pie(
    [passed, failed],
    labels=[f'Resolved correctly\n{passed:,} samples',
            f'Failed\n{failed:,} samples'],
    colors=['#1D9E75','#D85A30'], explode=(0.04,0.04),
    autopct='%1.2f%%', startangle=130,
    textprops={'fontsize':12,'color':'#2C2C2A'},
    wedgeprops={'edgecolor':'white','linewidth':2.5})
for at in autotexts:
    at.set_fontsize(14); at.set_fontweight('bold'); at.set_color('white')
ax1.set_title('Overall Pass vs Fail', fontsize=13,
              fontweight='bold', color='#2C2C2A', pad=12)

# Right donut
ax2 = axes[1]; ax2.set_facecolor('#F8F8F6')
wedges2, texts2, autotexts2 = ax2.pie(
    [TP, TN, FN, FP],
    labels=[f'TP  {TP:,}',f'TN  {TN:,}',f'FN  {FN:,}',f'FP  {FP:,}'],
    colors=['#1D9E75','#185FA5','#EF9F27','#D85A30'],
    explode=[0.03]*4, autopct='%1.1f%%', startangle=80,
    pctdistance=0.76,
    textprops={'fontsize':11,'color':'#2C2C2A'},
    wedgeprops={'edgecolor':'white','linewidth':2,'width':0.56})
for at in autotexts2:
    at.set_fontsize(10); at.set_fontweight('bold'); at.set_color('white')
ax2.text(0,0,'P  97.49%\nR  87.69%\nF1 92.33%',
         ha='center',va='center',fontsize=11,fontweight='bold',color='#2C2C2A')
ax2.set_title('TP / TN / FP / FN Breakdown', fontsize=13,
              fontweight='bold', color='#2C2C2A', pad=12)

plt.tight_layout()
plt.savefig('graph4_pass_fail.png',
            dpi=180, bbox_inches='tight', facecolor='#F8F8F6')
print("Saved graph4")
