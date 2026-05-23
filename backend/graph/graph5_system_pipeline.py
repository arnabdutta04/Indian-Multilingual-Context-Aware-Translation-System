"""
Graph 5 — System Pipeline
File: graph5_system_pipeline.py
Run:  python graph5_system_pipeline.py
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(18, 7))
fig.patch.set_facecolor('#F8F8F6')
ax.set_facecolor('#F8F8F6')
ax.set_xlim(0,18); ax.set_ylim(0,7); ax.axis('off')
ax.set_title('System Architecture — Context-Aware Coreference Resolution & Translation Pipeline',
             fontsize=14, fontweight='bold', color='#2C2C2A', pad=14)

def box(ax,x,y,w,h,color,l1,l2='',tc='white'):
    ax.add_patch(mpatches.FancyBboxPatch((x,y),w,h,
        boxstyle='round,pad=0.12',facecolor=color,edgecolor='white',linewidth=2,zorder=3))
    ax.text(x+w/2,y+h*(0.63 if l2 else 0.5),l1,
            ha='center',va='center',fontsize=10.5,fontweight='bold',color=tc,zorder=4)
    if l2:
        ax.text(x+w/2,y+h*0.28,l2,
                ha='center',va='center',fontsize=8.5,color=tc,alpha=0.92,zorder=4)

def arr(ax,x1,y1,x2,y2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->',color='#555552',lw=2.2))

box(ax,0.2,2.8,2.2,1.4,'#534AB7','Input Text','Raw English sentence')
arr(ax,2.4,3.5,2.8,3.5)
box(ax,2.8,2.8,2.4,1.4,'#185FA5','Contraction\nExpander',"it's→it is / can't→cannot")
arr(ax,5.2,3.5,5.6,3.5)
box(ax,5.6,2.8,2.4,1.4,'#1D9E75','Idiom\nExpander','80+ idioms · Groq + rules')
arr(ax,8.0,3.5,8.4,3.5)
box(ax,8.4,2.8,2.6,1.4,'#EF9F27','Coreference\nResolver',
    'South-Asian names · Gender inference',tc='#2C2C2A')
arr(ax,11.0,3.5,11.4,3.5)
box(ax,11.4,2.8,2.6,1.4,'#D85A30','Translation\nBackend',
    'Google→MyMemory→Libre→NLLB')
arr(ax,14.0,3.5,14.4,3.5)
box(ax,14.4,2.8,2.4,1.4,'#639922','Output +\nEmotion','11 languages · 100+ emotions')

ax.annotate('',xy=(8.4,2.3),xytext=(14.4,2.3),
            arrowprops=dict(arrowstyle='<-',color='#888780',lw=1.5,
                            connectionstyle='arc3,rad=0'))
ax.text(11.4,1.75,'Session history fed back as context for next call',
        ha='center',fontsize=9,color='#888780',style='italic')

box(ax,8.4,5.1,2.6,1.2,'#B4B2A9','Dataset\n5159 rows',
    '11 langs · 469 unique',tc='#2C2C2A')
ax.annotate('',xy=(9.7,4.2),xytext=(9.7,5.1),
            arrowprops=dict(arrowstyle='->',color='#B4B2A9',lw=1.5,linestyle='dashed'))

# Real metric values
metric_data = [
    ('Acc\n87.63%','#1D9E75'),('Prec\n97.49%','#185FA5'),
    ('Rec\n87.69%','#EF9F27'),('F1\n92.33%','#534AB7'),('BLEU\n71.34%','#D85A30'),
]
for i,(label,col) in enumerate(metric_data):
    xp = 5.8 + i*1.45
    box(ax,xp,0.25,1.25,1.05,col,label)
    arr(ax,xp+0.62,2.8,xp+0.62,1.3)

ax.text(9.15,0.03,'Evaluation Metrics — real values from eval run',
        ha='center',fontsize=10,color='#444441',fontweight='bold')

plt.tight_layout()
plt.savefig('graph5_system_pipeline.png',
            dpi=180,bbox_inches='tight',facecolor='#F8F8F6')
print("Saved graph5")
