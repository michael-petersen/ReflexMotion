
# basic imports
from __future__ import print_function
import numpy as np
from numpy.linalg import eig, inv
import time

# plotting elements
import matplotlib.pyplot as plt
import matplotlib.cm as cm
cmap = cm.RdBu_r
import matplotlib as mpl
mpl.rcParams['font.weight'] = 'medium'
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10


# exptool imports
from exptool.io import psp_io
from exptool.utils import kde_3d
from exptool.observables import transform
from exptool.analysis import pattern
from exptool.analysis import trapping
from exptool.utils import *
from exptool.observables import visualize

# astropy imports
import astropy.coordinates as coord
import astropy.units as u

import scipy.interpolate as interpolate
from scipy.interpolate import UnivariateSpline
import scipy




def wolfram_euler(PSPDump,phid,thetad,psid,velocity=True):
    '''
    rotate_points
        take a PSP dump and return the positions/velocities rotated by a specified set of angles

    inputs
    ------------------
    PSPDump     : input set of points
    xrotation   : rotation into/out of page, in degrees
    yrotation   :
    zrotation   : 
    velocity    : boolean
        if True, return velocity transformation as well
    euler       : boolean
        if True, transform as ZXZ' convention


    returns
    ------------------
    PSPOut      : the rotated phase-space output


    '''
    
    radfac = np.pi/180.
    
    # set rotation in radians
    phi   = phid*radfac
    theta = thetad*radfac
    psi   = psid*radfac
    
    # construct the rotation matrix TAIT-BRYAN method (x-y-z,
    # extrinsic rotations)
    Rmatrix = np.array([[np.cos(psi)*np.cos(phi) - np.cos(theta)*np.sin(phi)*np.sin(psi),\
                         np.cos(psi)*np.sin(phi) + np.cos(theta)*np.cos(phi)*np.sin(psi),\
                         np.sin(psi)*np.sin(theta)],\
                        [-np.sin(psi)*np.cos(phi) - np.cos(theta)*np.sin(phi)*np.cos(psi),\
                         -np.sin(psi)*np.sin(phi) + np.cos(theta)*np.cos(phi)*np.cos(psi),\
                          np.cos(psi)*np.sin(theta)],\
                        [ np.sin(theta)*np.sin(phi),\
                         -np.sin(theta)*np.cos(phi),\
                          np.cos(theta)]])


    # note: no guard against bad PSP here.
    pts = np.array([PSPDump.xpos,PSPDump.ypos,PSPDump.zpos])
    
    #
    # instantiate new blank PSP item
    PSPOut = psp_io.particle_holder()
    
    #
    # do the transformation in position
    tmp = np.dot(pts.T,Rmatrix)
    PSPOut.xpos = tmp[:,0]
    PSPOut.ypos = tmp[:,1]
    PSPOut.zpos = tmp[:,2]
    #

    # and velocity
    if velocity:
        vpts = np.array([PSPDump.xvel,PSPDump.yvel,PSPDump.zvel])
        tmp = np.dot(vpts.T,Rmatrix)
        PSPOut.xvel = tmp[:,0]
        PSPOut.yvel = tmp[:,1]
        PSPOut.zvel = tmp[:,2]
    #
    
    return PSPOut


def euler_xyz(phi,theta,psi=0.):
    Rmatrix = np.array([[ np.cos(theta)*np.cos(phi),\
                          np.cos(theta)*np.sin(phi),\
                         -np.sin(theta)],\
                        [np.sin(psi)*np.sin(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi),\
                         np.sin(psi)*np.sin(theta)*np.sin(phi) + np.cos(psi)*np.cos(phi),\
                         np.cos(theta)*np.sin(psi)],\
                        [np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi),\
                         np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi),\
                         np.cos(theta)*np.cos(psi)]])
    return Rmatrix


def wolfram_xyz(PSPDump,phid,thetad,psid,velocity=True,reverse=False,matrix=False,dot2=True):
    '''
    rotate_points
        take a PSP dump and return the positions/velocities rotated by a specified set of angles

    inputs
    ------------------
    PSPDump     : input set of points
    xrotation   : rotation into/out of page, in degrees
    yrotation   :
    zrotation   : 
    velocity    : boolean
        if True, return velocity transformation as well
    euler       : boolean
        if True, transform as ZXZ' convention


    returns
    ------------------
    PSPOut      : the rotated phase-space output


    '''
    #
    radfac = np.pi/180.
    #
    # set rotation in radians
    phi   = phid*radfac
    theta = thetad*radfac
    psi   = psid*radfac
    #
    Rmatrix = euler_xyz(phi,theta,psi)
    #
    if not reverse:
        Rmatrix = Rmatrix.T
    # note: no guard against bad PSP here.
    pts = np.array([PSPDump.xpos,PSPDump.ypos,PSPDump.zpos])
    #
    # instantiate new blank PSP item
    PSPOut = psp_io.particle_holder()    
    #
    # do the transformation in position
    tmp = np.dot(pts.T,Rmatrix)
    PSPOut.xpos = tmp[:,0]
    PSPOut.ypos = tmp[:,1]
    PSPOut.zpos = tmp[:,2]
    #
    #
    # and velocity
    if velocity:
        vpts = np.array([PSPDump.xvel,PSPDump.yvel,PSPDump.zvel])
        tmp = np.dot(vpts.T,Rmatrix)
        PSPOut.xvel = tmp[:,0]
        PSPOut.yvel = tmp[:,1]
        PSPOut.zvel = tmp[:,2]
        if dot2:
            tmp = np.dot(Rmatrix,vpts)
            PSPOut.xvel = tmp[0]
            PSPOut.yvel = tmp[1]
            PSPOut.zvel = tmp[2]
    #
    if matrix:
        return PSPOut,Rmatrix
    else:
        return PSPOut










def rotate_points(PSPDump,xrotation,yrotation,zrotation,velocity=True,euler=False):
    '''
    rotate_points
        take a PSP dump and return the positions/velocities rotated by a specified set of angles

    inputs
    ------------------
    PSPDump     : input set of points
    xrotation   : rotation into/out of page, in degrees
    yrotation   :
    zrotation   : 
    velocity    : boolean
        if True, return velocity transformation as well
    euler       : boolean
        if True, transform as ZXZ' convention


    returns
    ------------------
    PSPOut      : the rotated phase-space output


    '''
    
    radfac = np.pi/180.
    
    # set rotation in radians
    a = xrotation*radfac#np.pi/2.2  # xrotation (the tip into/out of page)
    b = yrotation*radfac#np.pi/3.   # yrotation
    c = zrotation*radfac#np.pi      # zrotation
    
    # construct the rotation matrix TAIT-BRYAN method (x-y-z,
    # extrinsic rotations)
    Rx = np.array([[1.,0.,0.],[0.,np.cos(a),np.sin(a)],[0.,-np.sin(a),np.cos(a)]])
    Ry = np.array([[np.cos(b),0.,-np.sin(b)],[0.,1.,0.],[np.sin(b),0.,np.cos(b)]])
    Rz = np.array([[np.cos(c),np.sin(c),0.,],[-np.sin(c),np.cos(c),0.],[0.,0.,1.]])
    Rmatrix = np.dot(Rx,np.dot(Ry,Rz))

    # construct the rotation matrix EULER ANGLES (z-x-z) (phi, theta,
    # psi)
    # follow the Wolfram Euler angle conventions
    if euler:
        phi = a
        theta = b
        psi = c
        D = np.array([[np.cos(phi),np.sin(phi),0.,],[-np.sin(phi),np.cos(phi),0.],[0.,0.,1.]])
        C = np.array([[1.,0.,0.],[0.,np.cos(theta),np.sin(theta)],[0.,-np.sin(theta),np.cos(theta)]])
        B = np.array([[np.cos(psi),np.sin(psi),0.,],[-np.sin(psi),np.cos(psi),0.],[0.,0.,1.]])
        Rmatrix = np.dot(B,np.dot(C,D))

    
    # structure the points for rotation

    # note: no guard against bad PSP here.
    pts = np.array([PSPDump.xpos,PSPDump.ypos,PSPDump.zpos])
    
    #
    # instantiate new blank PSP item
    PSPOut = psp_io.particle_holder()
    
    #
    # do the transformation in position
    tmp = np.dot(pts.T,Rmatrix)
    PSPOut.xpos = tmp[:,0]
    PSPOut.ypos = tmp[:,1]
    PSPOut.zpos = tmp[:,2]
    #

    # and velocity
    if velocity:
        vpts = np.array([PSPDump.xvel,PSPDump.yvel,PSPDump.zvel])
        tmp = np.dot(vpts.T,Rmatrix)
        PSPOut.xvel = tmp[:,0]
        PSPOut.yvel = tmp[:,1]
        PSPOut.zvel = tmp[:,2]
    #
    
    return PSPOut






def compute_spherical(x,y,z,vx,vy,vz,twopi=True):
    """compute spherical coordinates from cartesian
    
    inputs
    ----------------
    x
    y
    z
    vx
    vy
    vz
    twopi : (bool, default=True) set range from 
    
    returns
    ----------------
    
    
    """

    r3 = np.sqrt(x*x + y*y + z*z)

    # azimuthal angle
    phi   = np.arctan2(y,x)
    

    
    # polar angle
    theta = np.arccos(z/(r3+1.e-18))

    # polar angle
    cost = (z/(r3+1.e-18)) 
    sint = np.sqrt(1. - cost*cost)
    cosp = np.cos(phi)
    sinp = np.sin(phi)

    vr      = sint*(cosp*vx + sinp*vy) + cost*vz
    vphi    = (-sinp*vx + cosp*vy)
    # jorge's version:
    #vphi    = sinp*vx - cosp*vy
    vtheta  = (cost*(cosp*vx + sinp*vy) - sint*vz)
    
    # move to down here so the transform happens AFTER computing
    if twopi:
        # reset azimuthal coordinate to be in [0,2pi] -- not [-pi,pi]
        if phi.size>1:
            phi[phi<0.] += 2.*np.pi 
        else:
            if phi<0:
                phi += 2.*np.pi  
     
    return r3,theta,phi,vr,vtheta,vphi

    
    
def compute_cartesian(r,theta,phi,vr,vtheta,vphi):
    """
    
    
    r      : radius
    phi    : azimuthal coordinate. must be in [-pi,pi] or [0,2pi]
    theta  : polar coordinate. must be in [0,pi]
    
    vr     :
    vphi   :
    vtheta :
    
    
    """
    
    sint = np.sin(theta)
    cost = np.cos(theta)
    sinp = np.sin(phi)
    cosp = np.cos(phi)

    # transformation matrix, following Wolfram convention
    ur1 = sint*cosp
    ur2 = sint*sinp
    ur3 = cost
    
    uth1 = cost*cosp
    uth2 = cost*sinp
    uth3 =-sint
    
    uphi1 =-sinp
    uphi2 = cosp
    uphi3 = 0.    
    
    xpos =  r * ur1
    ypos =  r * ur2
    zpos =  r * ur3
    
    xvel = vr * ur1 + vtheta * uth1 + vphi * uphi1
    yvel = vr * ur2 + vtheta * uth2 + vphi * uphi2
    zvel = vr * ur3 + vtheta * uth3 + vphi * uphi3

    return xpos,ypos,zpos,xvel,yvel,zvel

    
    
def compute_spherical_u(x,y,z,vx,vy,vz):
    """

    using the u convention from Jorge
    
    """
    rr    = np.sqrt(x*x + y*y + z*z)
    phi   = np.arctan2(y,x)
    theta = np.arccos(z/rr)

    # transformation matrix, following Wolfram convention
    sint = np.sin(theta)
    cost = np.cos(theta)
    sinp = np.sin(phi)
    cosp = np.cos(phi)

    ur1 = sint*cosp
    ur2 = sint*sinp
    ur3 = cost
    
    uth1 = cost*cosp
    uth2 = cost*sinp
    uth3 =-sint
    
    uphi1 =-sinp
    uphi2 = cosp
    uphi3 = 0.    
    
    vr     = 0.
    vtheta = 0.
    vphi   = 0.#vx * uphi
    
    xvel = vr * ur1 + vtheta * uth1 + vphi * uphi1
    yvel = vr * ur2 + vtheta * uth2 + vphi * uphi2
    zvel = vr * ur3 + vtheta * uth3 + vphi * uphi3

    return rr,theta,phi,vr,vtheta,vphi

    

def compute_spherical_old(x,y,z,vx,vy,vz,source=False,linear=True):
    """return spherical coordinates
    
    the options are source or destination coordinates...not sure what I've been doing this whole time
    
    """
    
    r3 = np.sqrt(x*x + y*y + z*z)
    r2 = np.sqrt(x*x + y*y)
    
    phi   = np.arctan2(y,x)
    
    # define to go from +pi/2 overhead to -pi/2 below
    theta = np.arccos(-z/r3) - np.pi/2.
    
    cost = (z/(r3+1.e-18))
    sint = np.sqrt(1. - cost*cost)
    cosp = np.cos(phi)
    sinp = np.sin(phi)


    if source:
        vr      = cost*(cosp*vx + sinp*vy) + cost*vz
        vphi    = (-sinp*vx + cosp*vy)
        vtheta  = (sint*(cosp*vx + sinp*vy) - cost*vz)
        
    else:
        vr      = (x*vx + y*vy + z*vz)/r3
        dphi    = (x*vy - y*vx)/r2
        dtheta  = ((x*vx + y*vy)*z - r2*r2*vz)/(r2*r3)

        # modulate out of radians
        if linear:
            vphi    = dphi*r3#*np.sin(theta) # cos because of how we have defined theta
            vtheta  = dtheta*r3
            #vphi    = dphi/r3#*np.sin(theta) # cos because of how we have defined theta
            #vtheta  = dtheta/r3
            
        else:
            vphi   = dphi
            vtheta = dtheta


    return r3,theta,phi,vr,vtheta,vphi



def read_outlog(indir,runtag):
    """read the exp outlog to make the rough simulation overview
    
    inputs
    -----------
    indir   : (string) file directory
    runtag  : (string) tag of the run
    
    
    outputs
    -----------
    OLog    : (dictionary)
    
    
    """
    f = open(indir+'OUTLOG.'+runtag)
    a = f.readline()
    a = f.readline()
    splitter = f.readline()

    headervals = [d.strip() for d in splitter.split('|')]
    typevals = ['f8' for d in splitter.split('|')]

    #print(headervals)

    f.close()

    OLog = np.genfromtxt(indir+'OUTLOG.'+runtag,\
                            dtype={'names': headervals ,\
                                 'formats': typevals},\
                       skip_header=6,delimiter='|')

    # this creates many keys that are not formally allowed, so they are translated
    #print(OLog.dtype)
    
    return OLog





def make_velocity_map(phi,theta,velweight,mass,\
                      velscale=(225./1.1),\
                      gridsize=64,\
                      ktype='gaussian',\
                      npower=7.,return_density=False):
    '''simple velocity mapping from kde3d
    
    
    '''

    xx,yy,vv = kde_3d.total_kde_two((180./np.pi)*phi,(180./np.pi)*theta,\
                                       gridsize=gridsize,\
                                       extents=(-180,180.,-90,90),\
                                    weights=(velweight*mass),\
                                       ktype=ktype,npower=npower)

    xx,yy,mm = kde_3d.total_kde_two((180./np.pi)*phi,(180./np.pi)*theta,\
                                       gridsize=gridsize,\
                                       extents=(-180,180.,-90,90),\
                                       weights=mass,\
                                       ktype=ktype,npower=npower)


    pltval = velscale*(vv/(mm+1.e-18))
    
    if return_density:
        return xx,yy,mm
    else:
        return xx,yy,pltval

    

def make_dispersion_map(phi,theta,velweight,mass,\
                      velscale=(225./1.1),\
                      gridsize=64,\
                      ktype='gaussian',\
                      npower=7.,return_density=False):
    '''simple velocity mapping from kde3d
    
    
    '''
    xx,yy,dd = kde_3d.total_kde_two((180./np.pi)*phi,(180./np.pi)*theta,\
                                       gridsize=gridsize,\
                                       extents=(-180,180.,-90,90),\
                                    weights=(velweight*velweight*mass),\
                                       ktype=ktype,npower=npower)

    xx,yy,vv = kde_3d.total_kde_two((180./np.pi)*phi,(180./np.pi)*theta,\
                                       gridsize=gridsize,\
                                       extents=(-180,180.,-90,90),\
                                    weights=(velweight*mass),\
                                       ktype=ktype,npower=npower)

    xx,yy,mm = kde_3d.total_kde_two((180./np.pi)*phi,(180./np.pi)*theta,\
                                       gridsize=gridsize,\
                                       extents=(-180,180.,-90,90),\
                                       weights=mass,\
                                       ktype=ktype,npower=npower)


    pltval = velscale*(vv/(mm+1.e-18))
    
    if return_density:
        return xx,yy,mm
    else:
        return xx,yy,pltval






def sine_func(x,a,b,c,d):
    """a simple sine curve for fitting
    
    inputs
    ------------
    x     : angle, in radians
    a     : amplitude
    b     : offset
    c     : (hard-coded to 2. right now) sine power
    d     : angle scaling
    
    """
    return a*np.sin(d*x)**2. + b


def fit_sine(xdata,ydata):
    """fit a simple sine curve to the data"""
    
    w = np.where( (np.isfinite(xdata)) & (np.isfinite(ydata)))
    
    popt,pcov = scipy.optimize.curve_fit(sine_func,xdata[w],ydata[w])
    
    return popt



def median_line(angular_distance,velmap,angles=np.linspace(0.,np.pi,30),median=True):
    """compute a median line for the angular distance-amplitude plots"""
    
    amplitude = np.zeros(angles.size)
    
    for angleindx,angle in enumerate(angles):
        anglemin = angle
        if angleindx==angles.size-1:
            anglemax = np.inf
        else:
            anglemax = angles[angleindx+1]
            
        w = np.where( (angular_distance > anglemin) & (angular_distance < anglemax))[0]
        
        if median:
            amplitude[angleindx] = np.nanmedian(velmap[w])
        else:
            amplitude[angleindx] = np.nanmean(velmap[w])
            
    return angles+0.5*(angles[1]-angles[0]),amplitude


def median_line_equal(angular_distancetmp,velmaptmp,nangles=15,median=True):
    """compute a median line for the angular distance-amplitude plots
    
    with equal numbers of bins
    
    """
    
    angles = np.zeros(nangles)
    amplitude = np.zeros(nangles)
    
    distsort = angular_distancetmp.argsort()
    binsize = int(np.floor(angular_distancetmp.size/nangles))
    
    print(binsize)
    
    angular_distance = angular_distancetmp[distsort]
    velmap = velmaptmp[distsort]
    
    for angleindx in range(0,nangles):
        
        #print(int(angleindx*binsize),int(angleindx*(binsize+1)))
        
        if angleindx==nangles-1:
            angles[angleindx] = np.nanmean(angular_distance[int(angleindx*binsize):])
            
            if median:
                amplitude[angleindx] = np.nanmedian(velmap[int(angleindx*binsize):])
            else:
                amplitude[angleindx] = np.nanmean(velmap[int(angleindx*binsize):])
            
        else:
            angles[angleindx] = np.nanmean(angular_distance[int(angleindx*binsize):int(angleindx*(binsize+1))])
        
            if median:
                amplitude[angleindx] = np.nanmedian(velmap[int(angleindx*binsize):int(angleindx*(binsize+1))])
            else:
                amplitude[angleindx] = np.nanmean(velmap[int(angleindx*binsize):int(angleindx*(binsize+1))])
            
    return angles,amplitude
        


        

    
def do_transform_and_plot(SatPoints,xrot,yrot,zrot,separation,tol=50.,angletol=10.,euler=False):
    """
    """
    TransSat = rotate_points(SatPoints,xrot,yrot,zrot,euler=euler)

    satpos_l_t = np.arctan2(TransSat.ypos[0:np.argmin(separation)],TransSat.xpos[0:np.argmin(separation)])
    satpos_b_t = np.arccos(TransSat.zpos[0:np.argmin(separation)]/np.sqrt(\
        TransSat.xpos*TransSat.xpos + TransSat.ypos*TransSat.ypos + TransSat.zpos*TransSat.zpos)[0:np.argmin(separation)]) - np.pi/2.

    #print(xrot,yrot,zrot,np.sum(np.abs(satpos_b_t)))

    finalpos = np.sqrt(((180./np.pi)*satpos_l_t[-1]*np.cos(satpos_b_t[-1]))**2. + (180./np.pi)*satpos_b_t[-1]**2.)
    
    if (np.sum(np.abs(satpos_b_t))<tol) & (finalpos < angletol):
        print(xrot,yrot,zrot,np.sum(np.abs(satpos_b_t)),finalpos)
        plt.plot((180./np.pi)*satpos_l_t,(180./np.pi)*satpos_b_t,color='red',lw=0.5)
        plt.scatter((180./np.pi)*satpos_l_t[-1],(180./np.pi)*satpos_b_t[-1],color='red')
    
    # any annotation?
        plt.text((180./np.pi)*satpos_l_t[0],(180./np.pi)*satpos_b_t[0],str(xrot)+','+str(yrot)+','+str(zrot),color='red',size=6)




def fit_by_radius(r3,vphi,rbins=np.arange(0.,95.,10.),distscale=300.):
    signal = np.zeros(rbins.size)
    rbins /= distscale

    for rindx,rval in enumerate(rbins):
        rmin = rval
        if rindx==rbins.size-1:
            rmax = np.inf
        else:
            rmax = rbins[rindx+1]
    
        w = np.where( (r3 > rmin) & (r3 < rmax))[0] # bin 'em up
    
        signal[rindx] = np.nanmean(vphi[w])
        
    return rbins*distscale,signal





def de_rotate_points(PSPDump,xrotation,yrotation,zrotation,euler=False):
    '''
    de_rotate_points
        take a PSP dump and return the positions/velocities derotated by a specified set of angles

    simply the inverse of rotate_points.

    inputs
    ------------------
    PSPDump
    xrotation   : rotation into/out of page, in degrees
    yrotation
    zrotation



    returns
    ------------------
    PSPOut      : the rotated phase-space output


    '''
    
    radfac = np.pi/180.
    
    # set rotation in radians
    a = xrotation*radfac#np.pi/2.2  # xrotation (the tip into/out of page)
    b = yrotation*radfac#np.pi/3.   # yrotation
    c = zrotation*radfac#np.pi      # zrotation
    
    # construct the rotation matrix
    Rx = np.array([[1.,0.,0.],[0.,np.cos(a),np.sin(a)],[0.,-np.sin(a),np.cos(a)]])
    Ry = np.array([[np.cos(b),0.,-np.sin(b)],[0.,1.,0.],[np.sin(b),0.,np.cos(b)]])
    Rz = np.array([[np.cos(c),np.sin(c),0.,],[-np.sin(c),np.cos(c),0.],[0.,0.,1.]])
    Rmatrix = np.linalg.inv(np.dot(Rx,np.dot(Ry,Rz)))
  
    # construct the rotation matrix EULER ANGLES (z-x-z) (phi, theta,
    # psi)
    # follow the Wolfram Euler angle conventions
    if euler:
        phi = a
        theta = b
        psi = c
        D = np.array([[np.cos(phi),np.sin(phi),0.,],[-np.sin(phi),np.cos(phi),0.],[0.,0.,1.]])
        C = np.array([[1.,0.,0.],[0.,np.cos(theta),np.sin(theta)],[0.,-np.sin(theta),np.cos(theta)]])
        B = np.array([[np.cos(psi),np.sin(psi),0.,],[-np.sin(psi),np.cos(psi),0.],[0.,0.,1.]])
        Rmatrix = np.dot(B,np.dot(C,D))

    
    # structure the points for rotation

    # note: no guard against bad PSP here.
    pts = np.array([PSPDump.xpos,PSPDump.ypos,PSPDump.zpos])
    vpts = np.array([PSPDump.xvel,PSPDump.yvel,PSPDump.zvel])
    
    #
    # instantiate new blank PSP item
    PSPOut = psp_io.particle_holder()
    
    #
    # do the transformation in position
    tmp = np.dot(pts.T,Rmatrix)
    PSPOut.xpos = tmp[:,0]
    PSPOut.ypos = tmp[:,1]
    PSPOut.zpos = tmp[:,2]
    #

    # and velocity
    tmp = np.dot(vpts.T,Rmatrix)
    PSPOut.xvel = tmp[:,0]
    PSPOut.yvel = tmp[:,1]
    PSPOut.zvel = tmp[:,2]
    #
    PSPOut.mass = PSPDump.mass
    
    return PSPOut




def find_anticentre(SatPoints,traj=True,angleres=5,angletol=1.,verbose=1):
    """use the transformation to compute the anticentre"""
    
    # compute the best transformation to some tolerance
    xb,yb,zb = find_transform(SatPoints,traj=True,angleres=5,angletol=1.)
    
    # make a placeholder of the direction
    ZeroPoints = psp_io.particle_holder()
    ZeroPoints.xpos = np.array([1.,-1.])
    ZeroPoints.ypos = np.array([0.,0.])
    ZeroPoints.zpos = np.array([0.,0.])
    ZeroPoints.xvel = np.array([0.,0.])
    ZeroPoints.yvel = np.array([0.,0.])
    ZeroPoints.zvel = np.array([0.,0.])
    
    Tst = de_rotate_points(ZeroPoints,xb,yb,zb)
    
    tstlt = np.arctan2(Tst.ypos,Tst.xpos)
    tstbt = np.arccos(-Tst.zpos/np.sqrt(Tst.xpos*Tst.xpos + Tst.ypos*Tst.ypos + Tst.zpos*Tst.zpos)) - np.pi/2.
    
    if verbose:
        print('Best fit x,y,z: ({},{},{})'.format(xb,yb,zb))

    return tstlt,tstbt



def find_transform(SatPoints,traj=False,angleres=5,angletol=1.,euler=False):
    """find a transform for a single point, or a trajectory, which places them on the phi1,phi2 plane
    
    
    
    """
    testangles = np.zeros(int((180/angleres)**3))
    anglelist = np.zeros([int((180/angleres)**3),3])
    
    if traj:
        testtraj = np.zeros(int((180/angleres)**3))


    num = 0
    for xrot in range(0,180,angleres):
        for yrot in range(0,180,angleres):
            for zrot in range(0,180,angleres):
    
                TransSat = rotate_points(SatPoints,xrot,yrot,zrot,euler=euler)

                satpos_l_t = np.arctan2(TransSat.ypos,TransSat.xpos)
                satpos_b_t = np.arccos(TransSat.zpos/np.sqrt(\
                    TransSat.xpos*TransSat.xpos + TransSat.ypos*TransSat.ypos + TransSat.zpos*TransSat.zpos)) - np.pi/2.

                # guard against??
                try:
                    testangles[num] = np.sqrt(((180./np.pi)*satpos_l_t*np.cos(satpos_b_t))**2. + (180./np.pi)*satpos_b_t**2.)
                except:
                    testangles[num] = np.sqrt(((180./np.pi)*satpos_l_t*np.cos(satpos_b_t))**2. + (180./np.pi)*satpos_b_t**2.)[-1]
                    
                anglelist[num] = np.array([xrot,yrot,zrot])
                
                if traj:
                    try:
                        testtraj[num] = np.sum(np.abs(satpos_b_t))
                    except:
                        testtraj[num] = np.sum(np.abs(satpos_b_t))[-1]
                
                num+=1
                
    #print(testangles)
    w = np.where(testangles<angletol)[0]
    
    if traj:
        ww = np.argmin(testtraj[w])
        return anglelist[w[ww]]
    
    else:
        return anglelist[w]

    

def plt_dispmap(xx,yy,pltval1,fit=False,return_axes=False):

    colorvals = np.linspace(np.percentile(np.abs(pltval1),0.5),np.percentile(np.abs(pltval1),99.5),64)



    pltval1[pltval1 < colorvals[0]] = colorvals[0]
    pltval1[pltval1 > colorvals[-1]] = colorvals[-1]


    fig = plt.figure(figsize=(6,5))
    ax1 = fig.add_axes([0.16,0.53,0.65,0.45])
    ax2 = fig.add_axes([0.83,0.53,0.02,0.45])
    ax3 = fig.add_axes([0.16,0.15,0.65,0.27])

    ax1.contourf(xx,yy,pltval1,colorvals,cmap=cmap)

    ax1.axis([-180.,180.,-90.,90])

    norm = mpl.colors.Normalize(vmin=colorvals[0], vmax=colorvals[-1])
    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,norm=norm)
    cb1.set_label('V$_{tan}$ [km/s]',size=18)
    #cb1.set_label('V$_{\phi}$ [km/s]',size=18)


    ax1.set_xlabel('$\ell$',size=18)
    ax1.set_ylabel('$b$',size=18)

    larr = (xx*np.pi/180.).reshape(-1,)
    barr = (yy*np.pi/180.).reshape(-1,)
    
    angular_distance = compute_angular_distance(barr,larr)


    velmap = pltval1.reshape(-1,)
    maxb = 0.48
    ax3.scatter(angular_distance[np.abs(barr)<maxb*np.pi]*180./np.pi,velmap[np.abs(barr)<maxb*np.pi],color='gray',s=0.5)



    angles,amplitude = median_line(angular_distance[np.abs(barr)<0.52*np.pi],velmap[np.abs(barr)<0.52*np.pi],angles=np.linspace(0.,np.pi,30))
    ax3.plot((180./np.pi)*angles,amplitude,color='black',drawstyle='steps-mid',lw=3.)

    if fit:
        popt = fit_sine(angles,amplitude)
        #plt.plot(angle_sample*180./np.pi,sine_func(angle_sample,*popt),color='red',linestyle='dashed')

        angle_sample = np.linspace(0.,np.pi,200)
        popt = fit_sine(angular_distance[np.abs(barr)<0.42*np.pi],velmap[np.abs(barr)<0.42*np.pi])
        #ax3.plot(angle_sample*180./np.pi,rm.sine_func(angle_sample,*popt),color='red')


    ax3.set_xlabel('Degree separation [deg]',size=12)
    ax3.set_ylabel('Velocity Boost [km/s]',size=12)

    if return_axes:
        return ax1,ax2,ax3
    
    

def plt_velmap(xx,yy,pltval1,fit=False,return_axes=False):

    colorvals = np.linspace(-np.percentile(np.abs(pltval1),99.5),np.percentile(np.abs(pltval1),99.5),64)



    pltval1[pltval1 < colorvals[0]] = colorvals[0]
    pltval1[pltval1 > colorvals[-1]] = colorvals[-1]


    fig = plt.figure(figsize=(6,5))
    ax1 = fig.add_axes([0.16,0.53,0.65,0.45])
    ax2 = fig.add_axes([0.83,0.53,0.02,0.45])
    ax3 = fig.add_axes([0.16,0.15,0.65,0.27])

    ax1.contourf(xx,yy,pltval1,colorvals,cmap=cmap)

    ax1.axis([-180.,180.,-90.,90])

    norm = mpl.colors.Normalize(vmin=colorvals[0], vmax=colorvals[-1])
    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,norm=norm)
    cb1.set_label('V$_{tan}$ [km/s]',size=18)
    #cb1.set_label('V$_{\phi}$ [km/s]',size=18)


    ax1.set_xlabel('$\ell$',size=18)
    ax1.set_ylabel('$b$',size=18)

    larr = (xx*np.pi/180.).reshape(-1,)
    barr = (yy*np.pi/180.).reshape(-1,)
    
    angular_distance = compute_angular_distance(barr,larr)


    velmap = pltval1.reshape(-1,)
    maxb = 0.48
    ax3.scatter(angular_distance[np.abs(barr)<maxb*np.pi]*180./np.pi,velmap[np.abs(barr)<maxb*np.pi],color='gray',s=0.5)



    angles,amplitude = median_line(angular_distance[np.abs(barr)<0.52*np.pi],velmap[np.abs(barr)<0.52*np.pi],angles=np.linspace(0.,np.pi,30))
    ax3.plot((180./np.pi)*angles,amplitude,color='black',drawstyle='steps-mid',lw=3.)

    if fit:
        popt = fit_sine(angles,amplitude)
        #plt.plot(angle_sample*180./np.pi,sine_func(angle_sample,*popt),color='red',linestyle='dashed')

        angle_sample = np.linspace(0.,np.pi,200)
        popt = fit_sine(angular_distance[np.abs(barr)<0.42*np.pi],velmap[np.abs(barr)<0.42*np.pi])
        #ax3.plot(angle_sample*180./np.pi,rm.sine_func(angle_sample,*popt),color='red')


    ax3.set_xlabel('Degree separation [deg]',size=12)
    ax3.set_ylabel('Velocity Boost [km/s]',size=12)

    if return_axes:
        return ax1,ax2,ax3
    
    
    
   
def compute_angular_distance(barr,larr):
    """compute the angular distance from a point, haversine formula
    
    thanks to https://gist.github.com/mazzma12/6dbcc71ab3b579c08d66a968ff509901
    
    """
        
    lmin = np.pi/2.
    angular_distance = (np.sin((barr))*np.cos((larr-lmin)) + \
                               np.cos((barr)))
    
    dlon = larr
    dlat = barr
    
    a = np.sin(
        dlat / 2.0)**2 + np.cos(barr) * np.cos(0.) * np.sin(dlon / 2.0)**2

    angular_distance = 2 * np.arcsin(np.sqrt(a))
    

    return angular_distance


    
def compute_angular_distance_2(barr,larr):
    """compute the angular distance from a point, brute-force-ish"""
        
    lmin = 0.
    angular_distance_0 = np.sqrt(np.cos(barr)*(larr-lmin)*(larr-lmin) + \
                               (barr)*(barr))
    
    lmin = 0#-np.pi
    angular_distance_1 = np.sqrt(np.cos(barr)*(larr-lmin)*(larr-lmin) + \
                               (barr)*(barr))
    
    lmin = 0#np.pi
    angular_distance_2 = np.sqrt(np.cos(barr)*(larr-lmin)*(larr-lmin) + \
                               (barr)*(barr))
    
    angular_distance = np.nanmin(np.array([angular_distance_0,angular_distance_1,angular_distance_2]),axis=0)

    angular_distance[np.abs(larr)>np.pi/2.] = np.pi - angular_distance[np.abs(larr)>np.pi/2.] 
    
    return angular_distance



def compute_proper_motion(velocity,distance):
    """simple transformation from velocity (in km/s) to proper motion in arcsec/yr"""
    
    # convert distance from kpc to pc
    return velocity/(4.74057*distance*1000.)


def undo_proper_motion(propermotion,distance):
    """simple transformation from velocity (in km/s) to proper motion in arcsec/yr"""
    
    # convert distance from kpc to pc
    return propermotion*(4.74057*distance*1000.)



def generate_observables(PSPDump,distscale=300.,velscale=(240./1.4),verbose=0,twopi=True):
    """generate specific observables for comparison with reflext motion"""
    
    
    xpos,ypos,zpos = PSPDump.xpos,PSPDump.ypos,PSPDump.zpos
    xvel,yvel,zvel = PSPDump.xvel,PSPDump.yvel,PSPDump.zvel
    
    dist,btmp,l,vlos,vtheta,vphi = compute_spherical(xpos,ypos,zpos,xvel,yvel,zvel,twopi=twopi)
    
    if verbose:
        print('polar range:',np.round(np.nanmin(btmp),2),np.round(np.nanmax(btmp),2))
        print('z range:',np.round(zpos[btmp.argmin()],2),np.round(zpos[btmp.argmax()],2))

    
    mul = compute_proper_motion(velscale*vphi,distscale*dist)
    
    # should there be a negative on vtheta to account for the reversed observable direction?
    # trying with...
    mub = compute_proper_motion(velscale*vtheta,distscale*dist)
    
    sdss = np.zeros(xpos.size)
    
    b = np.pi/2. - btmp
    
    if verbose:
        print('b range:',np.round(np.nanmin(b),2),np.round(np.nanmax(b),2))
        print('z range:',np.round(zpos[b.argmin()],2),np.round(zpos[b.argmax()],2))
 
    return l,b,\
           distscale*dist,\
           mul,mub,\
           velscale*vlos,\
           sdss
    

    
def generate_observables_prescale(PSPDump,distscale=300.,velscale=(240./1.4)):
    """generate specific observables for comparison with reflext motion"""
    
    
    xpos,ypos,zpos = PSPDump.xpos,PSPDump.ypos,PSPDump.zpos
    xvel,yvel,zvel = PSPDump.xvel,PSPDump.yvel,PSPDump.zvel
    
    dist,b,l,vlos,vtheta,vphi = rm.compute_spherical(distscale*xpos,distscale*ypos,distscale*zpos,\
                                                     velscale*xvel,velscale*yvel,velscale*zvel)
    
    mul = compute_proper_motion(vphi,dist)
    mub = compute_proper_motion(vtheta,dist)
    
    sdss = np.zeros(xpos.size)
    
    return l,b,\
           dist,\
           mul,mub,\
           vlos,\
           sdss
    





def make_map(l,b,vel,weight,twopi=True):
    if twopi:
        xx,yy,velmapv = kde_3d.total_kde_two(l,b,\
                                           gridsize=256,\
                                           extents=(0.,360.,-90,90),\
                                           weights=vel*weight,\
                                           ktype='gaussian',npower=6.)

        xx,yy,velmapw = kde_3d.total_kde_two(l,b,\
                                           gridsize=256,\
                                           extents=(0.,360.,-90,90),\
                                           weights=weight,\
                                           ktype='gaussian',npower=6.)
    else:
        xx,yy,velmapv = kde_3d.total_kde_two(l,b,\
                                           gridsize=256,\
                                           extents=(-180,180.,-90,90),\
                                           weights=vel*weight,\
                                           ktype='gaussian',npower=6.)

        xx,yy,velmapw = kde_3d.total_kde_two(l,b,\
                                           gridsize=256,\
                                           extents=(-180,180.,-90,90),\
                                           weights=weight,\
                                           ktype='gaussian',npower=6.)


    velmap = velmapv/velmapw
    
    return xx,yy,np.flipud(velmap)


# populate the model...

def make_model(phi,theta,psi=0.,pointres=180,reverse=False,twopi=True,travel='u',flip=False,verbose=False,vtravel=1.,fullreturn=False,solreflex=[0.,0.,0.,0.,0.,0.]):

    pointres = 180

    phirange = np.linspace(0,2.*np.pi,pointres)
    thrange = np.linspace(-np.pi/2.,np.pi/2.,pointres)


    pp,tt = np.meshgrid(phirange,thrange)

    ppflat = pp.reshape(-1,)
    ttflat = tt.reshape(-1,)

    vx = np.zeros(pp.size)+solreflex[3]
    vy = np.zeros(pp.size)+solreflex[4]
    if travel=='u':
        vz = -vtravel*np.ones(pp.size)+solreflex[5]
    else:
        vz = vtravel*np.ones(pp.size)+solreflex[5]


    Model = psp_io.particle_holder()


    Model.xpos = np.cos(ttflat)*np.cos(ppflat) + solreflex[0]
    Model.ypos = np.cos(ttflat)*np.sin(ppflat) + solreflex[1]
    Model.zpos = np.sin(ttflat)                + solreflex[2]
    

    Model.xvel = vx
    Model.yvel = vy
    Model.zvel = vz


    # the 180 shift brings the definition to spherical
    #Undo = rm.wolfram_xyz(Model,phi+180.,theta,psi,reverse=False)
    
    # but is not necessary here...
    Undo = wolfram_xyz(Model,phi,theta,psi,reverse=reverse)

    l,b,dist,mul,mub,vlos,sdssflag = generate_observables(Undo,verbose=verbose,velscale=1.,distscale=1.,twopi=twopi)
    
    # this has still proven not useful
    #if flip:
    #    mub*=-1
    #    
    #else:
    #    mul*=-1
    

    
    if flip:
        # this is what matches coming out of Jorge's check_coord.f matching.
        mub*=-1
        mul*=-1
    else:
        mub*=-1
        mul*=-1
    
    if fullreturn:
        return l,b,dist,mul,mub,vlos,Undo


    else:
        return l,b,dist,mul,mub,vlos




def make_model2(phi,theta,amp2=1.,psi=0.,pointres=180,reverse=False,twopi=True):

    pointres = 180

    phirange = np.linspace(0,2.*np.pi,pointres)
    thrange = np.linspace(-np.pi/2.,np.pi/2.,pointres)


    pp,tt = np.meshgrid(phirange,thrange)

    ppflat = pp.reshape(-1,)
    ttflat = tt.reshape(-1,)

    vx = np.zeros(pp.size)
    vy = np.zeros(pp.size)
    vz = -np.ones(pp.size)


    Model = psp_io.particle_holder()


    Model.xpos = np.cos(ttflat)*np.cos(ppflat)
    Model.ypos = np.cos(ttflat)*np.sin(ppflat)
    Model.zpos = np.sin(ttflat)
    
    
    """
    Model.xpos = np.sin(ttflat)*np.cos(ppflat)
    Model.ypos = np.sin(ttflat)*np.sin(ppflat)
    Model.zpos = np.cos(ttflat)
    """
        

    Model.xvel = vx#+0.5#+np.cos(2.*(np.pi/2. - ttflat))
    Model.yvel = vy
    Model.zvel = vz+0.5*np.cos(ppflat)#+np.sin((np.pi/2. - ttflat))


    # the 180 shift brings the definition to spherical
    #Undo = rm.wolfram_xyz(Model,phi+180.,theta,psi,reverse=False)
    
    # but is not necessary here...
    Undo = rm.wolfram_xyz(Model,phi,theta,psi,reverse=reverse)

    l,b,dist,mul,mub,vlos,sdssflag = generate_observables(Undo,verbose=True,velscale=1.,distscale=1.,twopi=twopi)
    
    # this has still proven not useful
    #mub*=-1
    
    # this is correct for galactic coordinates...
    mul*=-1
    
    return l,b,dist,mul,mub,vlos



def compare_file_and_model(compfile,modelangles=[0,0,0],outputfile='',dlim=[0,300]):
    """compare a mock file and a model
    
    
    inputs
    ----------------
    compfile     : input mock file; needs standard format.
    modelangles  : phi,theta,psi for model to test against
    outputfile   : string for image name to save
    dlim         : lower and upper bound on distance to probe, in kpc
    
    returns
    ----------------
    none
    
    """
    pointres=45

    fig = plt.figure(figsize=(8,3))

    xmin = 0.08
    ymin = 0.14
    dx = 0.26
    dy = 0.39
    buffer = 0.01

    ax1 = fig.add_axes([xmin+0.*(dx+buffer),ymin+1.*(dy+buffer),dx,dy])
    ax2 = fig.add_axes([xmin+1.*(dx+buffer),ymin+1.*(dy+buffer),dx,dy])
    ax3 = fig.add_axes([xmin+2.*(dx+buffer),ymin+1.*(dy+buffer),dx,dy])

    ax4 = fig.add_axes([xmin+0.*(dx+buffer),ymin+0.*(dy+buffer),dx,dy])
    ax5 = fig.add_axes([xmin+1.*(dx+buffer),ymin+0.*(dy+buffer),dx,dy])
    ax6 = fig.add_axes([xmin+2.*(dx+buffer),ymin+0.*(dy+buffer),dx,dy])


    ax7 = fig.add_axes([xmin+3.*(dx+buffer),ymin+0.*(dy+buffer),0.01,2.*dy+buffer])


    cmap = cm.coolwarm; norm = mpl.colors.Normalize(vmin=-1., vmax=1.)
    cb1 = mpl.colorbar.ColorbarBase(ax7, cmap=cmap,norm=norm)
    cb1.set_label('velocity [normalised]',size=12)


    velstretch = np.linspace(-32,32,64)


    twopi=False
    In = np.genfromtxt(compfile,skip_header=1)

    l      = In[:,0]
    b      = In[:,1]
    dist   = In[:,2]
    edist  = In[:,3]
    mul    = In[:,4]
    emul   = In[:,5]
    mub    = In[:,6]
    emub   = In[:,7]
    vlos   = In[:,8]
    evlos  = In[:,9]
    rapo   = In[:,10]
    weight = In[:,11]
    sdss   = In[:,12]

    # need to pass this in
    dcut = np.where( (dist<dlim[1]) & (dist>dlim[0]))[0]

    velstretch = np.linspace(-60,60,64)

    xx,yy,velmap = make_map(l[dcut],b[dcut],vlos[dcut],weight[dcut])
    velmap[velmap>np.nanmax(velstretch)] = np.nanmax(velstretch)
    velmap[velmap<np.nanmin(velstretch)] = np.nanmin(velstretch)
    ax1.contourf(xx,yy,velmap,velstretch,cmap=cm.coolwarm)

    vel = undo_proper_motion(0.001*mul,dist)
    xx,yy,velmap = make_map(l[dcut],b[dcut],vel[dcut],weight[dcut])
    velmap[velmap>np.nanmax(velstretch)] = np.nanmax(velstretch)
    velmap[velmap<np.nanmin(velstretch)] = np.nanmin(velstretch)
    ax2.contourf(xx,yy,velmap,velstretch,cmap=cm.coolwarm)

    vel = undo_proper_motion(0.001*mub,dist)
    xx,yy,velmap = make_map(l[dcut],b[dcut],vel[dcut],weight[dcut])
    velmap[velmap>np.nanmax(velstretch)] = np.nanmax(velstretch)
    velmap[velmap<np.nanmin(velstretch)] = np.nanmin(velstretch)
    ax3.contourf(xx,yy,velmap,velstretch,cmap=cm.coolwarm)


    # set up model 2

    twopi=False
    phi = 325.;theta=145.;psi=0.
    #phi = 0.;theta=90.;psi=0.
    pointres=45
    phi = modelangles[0];theta=modelangles[1];psi=modelangles[2]

    l,b,dist,mul,mub,vlos = make_model(phi,theta,psi=psi,pointres=pointres,flip=True)
    vel = vlos
    velstretch = np.linspace(-1.,1.,64)
    ax4.scatter((180./np.pi)*l,(180./np.pi)*b,color=cm.coolwarm((vel-np.nanmin(velstretch))/(np.nanmax(velstretch)-np.nanmin(velstretch)),1.),s=20.)

    vel = undo_proper_motion(mul,dist)
    ax5.scatter((180./np.pi)*l,(180./np.pi)*b,color=cm.coolwarm((vel-np.nanmin(velstretch))/(np.nanmax(velstretch)-np.nanmin(velstretch)),1.),s=20.)

    vel = undo_proper_motion(mub,dist)
    ax6.scatter((180./np.pi)*l,(180./np.pi)*b,color=cm.coolwarm((vel-np.nanmin(velstretch))/(np.nanmax(velstretch)-np.nanmin(velstretch)),1.),s=20.)

    ax4.text(10,80,'$\phi={}^\circ, \\theta={}^\circ, \psi={}^\circ$'.format(str(int(phi)),\
                        str(int(theta)),\
                        str(int(psi))),size=10,ha='left',va='top')



    for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
        _ = ax.yaxis.set_ticks_position('both')
        _ = ax.xaxis.set_ticks_position('both')
        _ = ax.tick_params(axis="both",which='both',direction="in")

        if not twopi:
            ax.axis([-(180./np.pi)*np.pi,(180./np.pi)*np.pi,-(180./np.pi)*np.pi/2.,(180./np.pi)*np.pi/2.])
            ax.set_xticks([-180.,-90.,0.,90.,180.])
            ax.axis([0.,(180./np.pi)*2*np.pi,-(180./np.pi)*np.pi/2.,(180./np.pi)*np.pi/2.])
            ax.set_xticks([0.,90.,180.,270.,360.])
            ax.set_yticks([-90.,-45.,0.,45.,90.])
            #ax.set_xlabel('$\ell$ [deg]',size=12)
        else:
            ax.axis([-(180./np.pi)*np.pi,(180./np.pi)*np.pi,(180./np.pi)*np.pi,0.])
            ax.set_xlabel('$\phi_1$',size=12)


    for ax in [ax2,ax3,ax5,ax6]:
        ax.set_xticklabels(['','-90','0','90','180'])
        ax.set_xticklabels(['','90','180','270','360'])

    ax1.set_yticklabels(['','-45','0','45','90'])


    ax2.set_yticklabels(())
    ax3.set_yticklabels(())
    ax5.set_yticklabels(())
    ax6.set_yticklabels(())
    ax1.set_xticklabels(())
    ax2.set_xticklabels(())
    ax3.set_xticklabels(())




    if not twopi:
        ax1.set_title('v$_{\\rm LOS}$',size=12)
        ax2.set_title('v$_\ell$',size=12)
        ax3.set_title('v$_b$',size=12)
    else:
        ax1.set_title('v$_{\\rm LOS}$',size=12)
        ax2.set_title('v$_{\phi_1}$',size=12)
        ax3.set_title('v$_{\phi_2}$',size=12)


    if twopi:
        ax4.set_ylabel('$\phi_2$',size=12,y=1.)
        ax5.set_xlabel('$\ell$ [deg]',size=12)
    else:
        ax4.set_ylabel('$b$ [deg]',size=12,y=1.)
        ax5.set_xlabel('$\ell$ [deg]',size=12)

    fig.tight_layout()


    plt.savefig(outputfile,dpi=300)






def make_debias_model(l,b,phi,theta,vtravel,psi=0.,verbose=False):

    nposteriors = phi.size

    # translate l and b to rotators
    

    inphi   = l*(np.pi/180.) # make radians!
    intheta = (90.-b)*(np.pi/180.) # make radians!
    
    vx = 0.
    vy = 0.
    vz = -1.
    
    Model = psp_io.particle_holder()

    Model.xpos = np.cos(intheta)*np.cos(inphi)
    Model.ypos = np.cos(intheta)*np.sin(inphi)
    Model.zpos = np.sin(intheta)
    
    Model.xvel = vx
    Model.yvel = vy
    Model.zvel = vz

    l = np.zeros(nposteriors)
    b = np.zeros(nposteriors)
    dist = np.zeros(nposteriors)
    mul = np.zeros(nposteriors)
    mub = np.zeros(nposteriors)
    vlos = np.zeros(nposteriors)

    # but is not necessary here...
    for i in range(0,nposteriors):
        Model.zvel = -vtravel[i]
        Undo = wolfram_xyz_single(Model,phi[i],theta[i],0.,reverse=False)

        l[i],b[i],dist[i],vlos[i],mul[i],mub[i] = jorge_galactic(Undo.xpos,Undo.ypos,Undo.zpos,Undo.xvel,Undo.yvel,Undo.zvel)
 
    return l,b,dist,vlos,mul,mub



def wolfram_xyz_single(PSPDump,phid,thetad,psid,velocity=True,reverse=False,matrix=False,dot2=True):
    '''
    rotate_points
        take a PSP dump and return the positions/velocities rotated by a specified set of angles

    inputs
    ------------------
    PSPDump     : input set of points
    xrotation   : rotation into/out of page, in degrees
    yrotation   :
    zrotation   : 
    velocity    : boolean
        if True, return velocity transformation as well
    euler       : boolean
        if True, transform as ZXZ' convention


    returns
    ------------------
    PSPOut      : the rotated phase-space output


    '''
    #
    radfac = np.pi/180.
    #
    # set rotation in radians
    phi   = phid*radfac
    theta = thetad*radfac
    psi   = psid*radfac
    #
    Rmatrix = euler_xyz(phi,theta,psi)
    #
    if not reverse:
        Rmatrix = Rmatrix.T
    # note: no guard against bad PSP here.
    pts = np.array([PSPDump.xpos,PSPDump.ypos,PSPDump.zpos])
    #
    # instantiate new blank PSP item
    PSPOut = psp_io.particle_holder()    
    #
    # do the transformation in position
    tmp = np.dot(pts.T,Rmatrix)
    PSPOut.xpos = tmp[0]
    PSPOut.ypos = tmp[1]
    PSPOut.zpos = tmp[2]
    #
    #
    # and velocity
    if velocity:
        vpts = np.array([PSPDump.xvel,PSPDump.yvel,PSPDump.zvel])
        tmp = np.dot(vpts.T,Rmatrix)
        PSPOut.xvel = tmp[0]
        PSPOut.yvel = tmp[1]
        PSPOut.zvel = tmp[2]
        if dot2:
            tmp = np.dot(Rmatrix,vpts)
            PSPOut.xvel = tmp[0]
            PSPOut.yvel = tmp[1]
            PSPOut.zvel = tmp[2]
    #
    if matrix:
        return PSPOut,Rmatrix
    else:
        return PSPOut







def jorge_galactic(x0,y0,z0,u0,v0,w0):
    
    rad = np.sqrt(x0**2+y0**2+z0**2)
    xphi= np.arctan2(y0,x0)
    xth = np.arccos(z0/rad)
    
    xur = np.zeros([3,x0.size])
    xur[0]= np.sin(xth)*np.cos(xphi)
    xur[1]= np.sin(xth)*np.sin(xphi)
    xur[2]= np.cos(xth)
         
    xuth = np.zeros([3,x0.size])
    xuth[0]= np.cos(xth)*np.cos(xphi)
    xuth[1]= np.cos(xth)*np.sin(xphi)
    xuth[2]=-np.sin(xth)

    xuphi = np.zeros([3,x0.size])
    xuphi[0]=-np.sin(xphi)
    xuphi[1]=+np.cos(xphi)
    xuphi[2]= 0.
    
    vr =    u0*  xur[0] + v0*  xur[1] + w0*  xur[2]
    vth=    u0* xuth[0] + v0* xuth[1] + w0* xuth[2]
    vphi=   u0*xuphi[0] + v0*xuphi[1] + w0*xuphi[2]
          
    vb= -vth
    
    # following the astropy convention
    vl= vphi
         
    dk  =4.74057           #conversion from km/s
    par =1./rad             #arc sec --> rad in [kpc]
    dmul=vl / dk * par
    dmub=vb / dk * par

    f=np.pi/180.
    dB=np.arcsin(z0/rad)/f
    #dL=np.arctan(y0/x0)/f
    
    #dL[(y0<0)&(x0>0.)] += 360.
    #dL[(y0>0)&(x0<0.)] += 180.
    #dL[(y0<0)&(x0<0.)] += 180.
    
    dL = np.arctan2(y0,x0)/f
    
    #print(dL)
    
    if dL.size>1:
        dL[np.array(dL)<0.] += 360.
    else:
        if dL<0.: dL+=360.
    #if ((y0<0)&(x0>0.)): dL=dL+360.
    #if ((y0>0)&(x0<0.)): dL=dL+180.
    #if ((y0<0)&(x0<0.)): dL=dL+180.
    
    return dL,dB,rad,vr,dmul,dmub





def make_rotation_model(vphi,solpos=[0.01,0.,0.0],pointres=180,verbose=False):

    phirange = np.linspace(0,2.*np.pi,pointres)
    thrange = np.linspace(-np.pi/2.,np.pi/2.,pointres)

    pp,tt = np.meshgrid(phirange,thrange)

    ppflat = pp.reshape(-1,)
    ttflat = tt.reshape(-1,)

    vx0 = -np.sin(ppflat)*vphi
    vy0 =  np.cos(ppflat)*vphi
    
    vz0 =  np.zeros(ppflat.size)

    x0 = np.cos(ttflat)*np.cos(ppflat) + solpos[0]
    y0 = np.cos(ttflat)*np.sin(ppflat) + solpos[1]
    z0 = np.sin(ttflat) + solpos[2]

    vr = (x0*vx0 + y0*vy0)/np.sqrt(x0*x0 + y0*y0)
    # now revise vx0,vy0
    vx0 = np.cos(ppflat)*vr - np.sin(ppflat)*vphi
    vy0 = np.sin(ppflat)*vr + np.cos(ppflat)*vphi
    
    dL,dB,rad,vr,dmul,dmub = jorge_galactic(x0,y0,z0,vx0,vy0,vz0)
     
    return dL,dB,rad,dmul,dmub,vr






