import numpy as np


def galactic(x0,y0,z0,u0,v0,w0,twopi=True):
    """
    transform Cartesian coordinates to a spherical coordinate system that agrees with 
      galactic coordinates


    twopi : (bool, True) if True, returns azimuth in [0,2pi). If False, returns azimuth in [-pi,pi).

    returns
    ------------
    dL
    dB
    """
    rad = np.sqrt(x0**2+y0**2+z0**2)
    xphi= np.arctan2(y0,x0)
    xth = np.arccos(z0/rad)
    
    try:
        nobj = x0.size
    except:
        nobj = 1

    xur = np.zeros([3,nobj])
    xur[0]= np.sin(xth)*np.cos(xphi)
    xur[1]= np.sin(xth)*np.sin(xphi)
    xur[2]= np.cos(xth)
         
    xuth = np.zeros([3,nobj])
    xuth[0]= np.cos(xth)*np.cos(xphi)
    xuth[1]= np.cos(xth)*np.sin(xphi)
    xuth[2]=-np.sin(xth)

    xuphi = np.zeros([3,nobj])
    xuphi[0]=-np.sin(xphi)
    xuphi[1]=+np.cos(xphi)
    xuphi[2]= 0.
    
    vr =    u0*  xur[0] + v0*  xur[1] + w0*  xur[2]
    vth=    u0* xuth[0] + v0* xuth[1] + w0* xuth[2]
    vphi=   u0*xuphi[0] + v0*xuphi[1] + w0*xuphi[2]
          
    vb= -vth
    
    # match the astropy output
    vl= vphi
         
    dk  =4.74057           #conversion from km/s
    par =1./rad             #arc sec --> rad in [kpc]
    dmul=vl / dk * par
    dmub=vb / dk * par

    f=np.pi/180.
    dB=np.arcsin(z0/rad)/f
    
    if twopi:
        dL=np.arctan(y0/(x0+1.e-10))/f
        if nobj>1:
            dL[(y0<0.)&(x0>0.)] += 360.
            dL[(y0>0.)&(x0<0.)] += 180.
            dL[(y0<0.)&(x0<0.)] += 180.
        else:
            if (y0<0.)&(x0>0.): dL += 360.
            if (y0>0.)&(x0<0.): dL += 180.
            if (y0<0.)&(x0<0.): dL += 180.
    else:
        dL = np.arctan2(y0,x0)/f

    
    return dL,dB,rad,vr,dmul,dmub


