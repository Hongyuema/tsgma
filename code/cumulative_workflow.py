"""Render the cumulative grading diagram over the original figure artwork."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def draw(source, destination):
    original=Image.open(source)
    w,h=original.size
    fig=plt.figure(figsize=(8.2,8.2*h/w))
    ax=fig.add_axes([0,0,1,1])
    ax.imshow(original,extent=(0,w,h,0),interpolation='none')
    ax.set(xlim=(0,w),ylim=(h,0));ax.axis('off')
    # Text changes are vector overlays. The supplied maps, plots, branches,
    # colours and star symbols remain at their original coordinates.
    def replace(box,words,points=12.5,background='white'):
        x0,y0,x1,y1=box
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor=background,edgecolor='none'))
        ax.text((x0+x1)/2,(y0+y1)/2,words,ha='center',va='center',
                fontsize=points,fontfamily='Arial',weight='bold',linespacing=1.2)
    replace((1348,137,1590,181),'Temporal Association',8.2,'#ead3d1')
    replace((1305,533,1650,600),'Lagged Within-Country\nAssociation Analysis',9.0)
    for box,color,words in [
      ((485,775,625,814),'#c6ccdc','Not met'),
      ((643,896,742,938),'#f2e0fa','Met'),
      ((921,775,1063,813),'#ecf5d2','Not met'),
      ((1055,896,1158,937),'#ead3d1','Met'),
      ((1300,775,1440,815),'#f2e0fa','Not met'),
      ((1512,775,1614,814),'#ead3d1','Met')]:
        # Sample the unprinted interior to reproduce each original fill exactly.
        color=tuple(v/255 for v in original.convert('RGB').getpixel((box[0],box[1])))
        replace(box,words,9.2,color)
    replace((503,1010,590,1049),'0 stars',8.5,tuple(v/255 for v in original.convert('RGB').getpixel((505,1005))))
    replace((942,1010,1025,1049),'1 star',8.5,tuple(v/255 for v in original.convert('RGB').getpixel((942,1005))))
    replace((1321,1010,1410,1049),'2 stars',8.5,tuple(v/255 for v in original.convert('RGB').getpixel((1321,1005))))
    replace((1518,1010,1607,1049),'3 stars',8.5,tuple(v/255 for v in original.convert('RGB').getpixel((1518,1005))))
    ax.text(w/2,1175,'Sequential screening and cumulative grading do not establish causality.',
            ha='center',va='center',fontfamily='Arial',fontsize=9.1)
    ax.text(w/2,1210,'In this application, all 48 pairs were evaluated at every stage.',
            ha='center',va='center',fontfamily='Arial',fontsize=9.1)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination.with_suffix('.pdf'),facecolor='white')
    fig.savefig(destination.with_suffix('.png'),dpi=300,facecolor='white')
    plt.close(fig)

if __name__=='__main__':
    draw('outputs/NEXRES_revision_package/artwork_sources/flow.png','figures/Figure_1')
