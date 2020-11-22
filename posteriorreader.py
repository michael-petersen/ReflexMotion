
import numpy as np
import kde as kde_3d



def norm_histogram(arr,bins=-1):
    """ make a very simple normalised histogram"""
    
    if bins == -1:
        bins = int(len(arr)/100.)
        
    a = np.percentile(arr,0.0)
    A = np.percentile(arr,99.9)
    binvals = np.linspace(a,A,bins)
    outvals = np.zeros(binvals.size)
    #print(a,A,bins)
    da = (A-a)/(bins)
    
    for val in arr:
        #print(val-a,(val-a)/(da))
        indx = (val-a)/da
        if indx>bins-1: indx=bins-1
        #print(indx)
        outvals[int(indx)] += 1
        
    return binvals,outvals/(len(arr)*(A-a)/(bins-1))



def read_posterior(pfile):
    """read posteriors coming from Multinest"""
    
    A = np.genfromtxt(pfile)
    
    dname = {}
    dname['vtravel'] = A[:,0]
    dname['phi'] = (180./np.pi)*A[:,1]
    dname['phi'][dname['phi']<0] += 360.
    dname['theta'] = 90.-(180./np.pi)*np.arccos(A[:,2])
    dname['sigmar'] = 1./np.sqrt(A[:,3])
    dname['sigmap'] = 1./np.sqrt(A[:,4])
    dname['sigmat'] = 1./np.sqrt(A[:,5])
    dname['vra'] = A[:,6]
    dname['vth'] = A[:,7]
    dname['vphi'] = A[:,8]
    
    return dname

def total_bounds(dictnames,cats):
    """find the boundaries from posteriors"""
    
    bounddict = {}
    
    for cat in cats:
        bounddict[cat] = [np.percentile(np.array(np.concatenate([dictname[cat] for dictname in dictnames])),0.2),\
                          np.percentile(np.array(np.concatenate([dictname[cat] for dictname in dictnames])),99.8)]

    return bounddict




def make_banana(array1,array2,bounds=[-1,-1,-1,-1],gridsize=128):
    """bin data to make the banana plots"""
    
    a,A = np.nanmin(array1),np.nanmax(array1)
    b,B = np.nanmin(array2),np.nanmax(array2)
    da,db = (A-a)/128.,(B-b)/128.
    
    if bounds[0]==bounds[1]:
        bounds = [a,A,b,B]
    
    
    xx,yy,dens = kde_3d.total_kde_two(array1,array2,\
                                       gridsize=gridsize,\
                                       extents=bounds,\
                                       ktype='gaussian',npower=8.)


    #print(da,db)
    return xx,yy,np.flipud(dens),da*db



def plot_aitoff_banana(ax,catx,caty,color,border=False,bounds=[-1,-1,-1,-1],gridsize=100,binset=[92.,98.,99.5],alphaspace=0.2,zorder=0):


    xx,yy,dens,dadb = make_banana(catx,caty,bounds=bounds,gridsize=gridsize)
    densflat = dens.reshape(-1,)
    bins = np.percentile(densflat,binset)
    for ib,b in enumerate(bins):
        lobin = bins[ib]
        if ib==(len(bins)-1):
            hibin = np.inf
        else:
            hibin = bins[ib+1]
        if border:
            #print(1+ib)
            ax.contourf(xx,yy,dens,[lobin,hibin],colors=color,alpha=alphaspace*(1+ib)+0.4,zorder=zorder)
        else:
            ax.contourf(xx,yy,dens,[lobin,hibin],colors=color,alpha=alphaspace*(1+ib)+0.4,zorder=zorder)
            



