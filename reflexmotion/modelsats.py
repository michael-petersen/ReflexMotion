"""
modelsats.py

a collection of routines to
1) read in EXP trajectories
2) compute analytic orbits
3) find satisfactory rotations

"""

import numpy as np

from scipy.interpolate import UnivariateSpline

from exptool.utils import halo_methods

from scipy import interpolate


#####################################################################
#                      read trajectories
#####################################################################


def read_outlog(indir,runtag,full_return=True):
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
    
    if full_return:
        return headervals,OLog
    else:
        return OLog




def make_positions(OLog,headers):
    
    # get components out
    comps = [x.strip(" R(x)") for x in headers if " R(x)" in x]
    
    masterdict = {}
    
    masterdict['time'] = OLog['Time']
    
    # find unique times
    tt,ttind = np.unique(OLog['Time'],return_index=True)
    
    for comp in comps:
        masterdict[comp] = {}
        masterdict[comp]['xpos'] = UnivariateSpline(OLog['Time'][ttind],OLog[comp+'_Rx'][ttind],s=0)(OLog['Time'])
        masterdict[comp]['ypos'] = UnivariateSpline(OLog['Time'][ttind],OLog[comp+'_Ry'][ttind],s=0)(OLog['Time'])
        masterdict[comp]['zpos'] = UnivariateSpline(OLog['Time'][ttind],OLog[comp+'_Rz'][ttind],s=0)(OLog['Time'])
        masterdict[comp]['xvel'] = UnivariateSpline(OLog['Time'][ttind],OLog[comp+'_Vx'][ttind],s=0)(OLog['Time'])
        masterdict[comp]['yvel'] = UnivariateSpline(OLog['Time'][ttind],OLog[comp+'_Vy'][ttind],s=0)(OLog['Time'])
        masterdict[comp]['zvel'] = UnivariateSpline(OLog['Time'][ttind],OLog[comp+'_Vz'][ttind],s=0)(OLog['Time'])
        
        masterdict[comp]['rpos'] = np.sqrt(masterdict[comp]['xpos']*masterdict[comp]['xpos']+\
                                           masterdict[comp]['ypos']*masterdict[comp]['ypos']+\
                                           masterdict[comp]['zpos']*masterdict[comp]['zpos'])
        
        
    return masterdict


def comp_sep(M,comp1,comp2,vectors=False):
    """compute the separation of the components"""
    
    rsep = np.sqrt( (M[comp1]['xpos']-M[comp2]['xpos'])**2. +\
                    (M[comp1]['ypos']-M[comp2]['ypos'])**2. +\
                    (M[comp1]['zpos']-M[comp2]['zpos'])**2.)
    
    if vectors:
        xdiff  = (M[comp2]['xpos']-M[comp1]['xpos'])
        ydiff  = (M[comp2]['ypos']-M[comp1]['ypos'])
        zdiff  = (M[comp2]['zpos']-M[comp1]['zpos'])
        vxdiff = (M[comp2]['xvel']-M[comp1]['xvel'])
        vydiff = (M[comp2]['yvel']-M[comp1]['yvel'])
        vzdiff = (M[comp2]['zvel']-M[comp1]['zvel'])
    
        return xdiff,ydiff,zdiff,vxdiff,vydiff,vzdiff
    
    else:
        
        return rsep
    
    



#####################################################################
#                      for analytic orbits
######################################################################

# read in the model spherical potential: or it could be nonspherical?

#modfile = '/Users/mpetersen/Downloads/mdweinberg-exp-9d7cef75fc11/examples/LowFi/SLGridSph.model'

class sph_model():
    """class structure to read in and return spherical potential tables as functions
    """
    
    def __init__(self,modfile):
        """read in and create spline representations for model input file modfile"""
        
        R,D,M,P = halo_methods.read_sph_model_table(modfile)
        
        self.pfunc = interpolate.splrep(R, P, s=0)
        self.dfunc = interpolate.splrep(R, 4.*np.pi*D, s=0)
        self.mfunc = interpolate.splrep(R, M, s=0)
        
    def get_pot(self,rad):
        """return potential at radius rad"""
        
        return interpolate.splev(rad, self.pfunc, der=0)
     
    def get_dpot(self,rad):
        """return potential _derivative_ at radius rad"""
        
        return interpolate.splev(rad, self.pfunc, der=1)
    
    def get_dens(self,rad):
        """return density at radius rad"""
        
        return interpolate.splev(rad, self.dfunc, der=0)
    
    def get_mass(self,rad):
        """return mass enclosed at radius rad"""
        
        return interpolate.splev(rad, self.mfunc, der=0)


def return_euler_slater(PHI, THETA, PSI, BODY):
    """Return Euler angle matrix (ref. Slater)
    
    originally in MDW's euler_slater.cc
 
    inputs
    -----------
    Euler angles
    PHI    : 
    THETA  :
    PSI    :
    BODY   : if True, rotate body, if False, rotate axes, keep vector fixed in space
    
    returns
    -----------
    euler  : transformation matrix

    
    """

    euler = np.zeros([3,3])

    sph = np.sin(PHI);
    cph = np.cos(PHI);
    sth = np.sin(THETA);
    cth = np.cos(THETA);
    sps = np.sin(PSI);
    cps = np.cos(PSI);
  
    euler[0][0] = -sps*sph + cth*cph*cps;
    euler[0][1] =  sps*cph + cth*sph*cps;
    euler[0][2] =  cps*sth;
      
    euler[1][0] = -cps*sph - cth*cph*sps;
    euler[1][1] =  cps*cph - cth*sph*sps;
    euler[1][2] = -sps*sth;
      
    euler[2][0] = -sth*cph;
    euler[2][1] = -sth*sph;
    euler[2][2] =  cth;

    if (BODY):
        return euler.T
    else:
        return euler

    



class UnboundOrbit():
    """compute an unbound orbit in a given spherical potential
    """

    def __init__(self,modfile,rperi,E=0.,\
                thetainit=0.,psiinit=0.,phipinit=0.):
        """
        
        inputs
        ------------
        modfile
        Rperi
        E
        
        
        """
    
        self.mod = sph_model(modfile)
        self.rperi = rperi
        self.E = E
        
        self.deltaR = 0.0005; # set as the current default value
        self.Redge = 2.0;

        self.thetainit = thetainit
        self.psiinit   = psiinit  
        self.phipinit  = phipinit

        
        self.setup_orbit()
        
        self.first_step()
        
        while (self.R[self.step-1] < self.Redge):

            self.take_step()
            
        # rotate and return Cartesian
        self.set_cartesian()
        
        # compute velocities
        self.compute_velocities()
        
        
    def setup_orbit(self):
        """specify conditions at pericenter
        """
            
        self.VTperi = np.sqrt(2.0*(self.E - self.mod.get_pot(self.rperi)));
        self.J = self.rperi*self.VTperi;
            
        # this is simply an orbital integration buffer
        norb=10000
        self.R = np.zeros(norb)
        self.T = np.zeros(norb)
        self.PHI = np.zeros(norb)


        self.step = 0
        self.R[self.step] = self.rperi
        self.T[self.step] = 0.0    # time at outset
        self.PHI[self.step] = 0.0  # need to pick initial


        #// Trapezoidal increments

        self.rnext, self.rlast = self.rperi,self.rperi;
        self.tnext, self.tlast = 0.,0.;
        self.phinext, self.philast = 0.,0.;


    def first_step(self):
        """perform the first step in solving equations of motion
        """
        denom = np.sqrt(2.0*(self.VTperi*self.VTperi/self.rperi - self.mod.get_dpot(self.rperi)));

        # move one timestep
        self.rnext = self.rlast + self.deltaR;
        self.tnext = self.tlast + 2.0*np.sqrt(self.rnext - self.rperi)/denom;
        self.phinext = self.philast + 2.0*np.sqrt(self.rnext - self.rperi)/denom * self.J/(self.rperi*self.rperi);

        self.step = 1
        self.R[self.step] = self.rnext
        self.T[self.step] = self.tnext
        self.PHI[self.step] = self.phinext

        self.rlast = self.rnext;
        self.tlast = self.tnext;
        self.philast = self.phinext;

        self.step = 2

        
    def take_step(self):
        """advance one step by solving equations of motion
        """

        self.rnext = self.rlast + self.deltaR;
        self.tnext = self.tlast + 0.5*(self.rnext - self.rlast)*\
          (1.0/np.sqrt(2.0*(self.E - self.mod.get_pot(self.rlast)) - self.J*self.J/(self.rlast*self.rlast)) +\
           1.0/np.sqrt(2.0*(self.E - self.mod.get_pot(self.rnext)) - self.J*self.J/(self.rnext*self.rnext)));

        self.phinext = self.philast + 0.5*(self.rnext - self.rlast)*\
          (self.J/(self.rlast*self.rlast) /np.sqrt(2.0*(self.E - self.mod.get_pot(self.rlast)) - self.J*self.J/(self.rlast*self.rlast)) +\
           self.J/(self.rnext*self.rnext) /np.sqrt(2.0*(self.E - self.mod.get_pot(self.rnext)) - self.J*self.J/(self.rnext*self.rnext)));

        self.rlast = self.rnext;
        self.tlast = self.tnext;
        self.philast = self.phinext;

        self.R[self.step] = self.rnext
        self.T[self.step] = self.tnext
        self.PHI[self.step] = self.phinext

        self.step+=1

    def set_cartesian(self):
        """rotate coordinates as indicated by inpute coordinates
        and return as Cartesian
        """
        THETA   = self.thetainit   * np.pi/180.0;
        PSI     = self.psiinit     * np.pi/180.0;
        PHIP    = self.phipinit    * np.pi/180.0;

        Trans = return_euler_slater(PHIP, THETA, PSI, 1);

        self.TIME = np.zeros(2*self.step -1)
        self.XPOS = np.zeros(2*self.step - 1)
        self.YPOS = np.zeros(2*self.step - 1)
        self.ZPOS = np.zeros(2*self.step - 1)

        # do the infall of the orbit
        In = np.zeros(3)
        #for (unsigned i=step-1; i>=1; i--) {

        ostep = 0
        for i in range(self.step-1,0,-1):
            In[0] = self.R[i]*np.cos(-self.PHI[i]);
            In[1] = self.R[i]*np.sin(-self.PHI[i]);
            In[2] = 0.0;
            twisted  = np.dot(Trans,In)

            self.TIME[ostep] = -self.T[i]
            self.XPOS[ostep] = twisted[0]
            self.YPOS[ostep] = twisted[1]
            self.ZPOS[ostep] = twisted[2]
            ostep += 1


        for i in range(0,self.step):
            In[0] = self.R[i]*np.cos(self.PHI[i]);
            In[1] = self.R[i]*np.sin(self.PHI[i]);
            In[2] = 0.0;
            twisted  = np.dot(Trans,In)

            self.TIME[ostep] = self.T[i]
            self.XPOS[ostep] = twisted[0]
            self.YPOS[ostep] = twisted[1]
            self.ZPOS[ostep] = twisted[2]
            ostep += 1 
            
            
    def compute_velocities(self):
        """compute the velocities in Cartesian coordinates
        
        a good upgrade might be returning velocities in native observation space?
        """
        
        xfunc = interpolate.splrep(self.TIME, self.XPOS, s=0)
        yfunc = interpolate.splrep(self.TIME, self.YPOS, s=0)
        zfunc = interpolate.splrep(self.TIME, self.ZPOS, s=0)
        
        self.XVEL = interpolate.splev(self.TIME, xfunc, der=1)
        self.YVEL = interpolate.splev(self.TIME, yfunc, der=1)
        self.ZVEL = interpolate.splev(self.TIME, zfunc, der=1)









#####################################################################
#                      for best rotations
######################################################################


def find_best_rotation(Trajectory,\
                       targetx,targety,targetz,\
                      targetdx,targetdy,targetdz,\
                      targetvx,targetvy,targetvz,\
                      targetdvx,targetdvy,targetdvz,\
                      ntests=10,all=True,\
                      lenscale=300.,\
                      velscale=(240./1.4),\
                      verbose=0):
    """given targets, 
    BRUTE FORCE
    find the best rotation for a satellite trajectory
    
    
    inputs
    ------------------
    
    
    
    returns
    -----------------
    
    
    """
    
    
    TRAJMAT = np.array([Trajectory['xpos'],Trajectory['ypos'],Trajectory['zpos']])
    VELMAT  = np.array([Trajectory['xvel'],Trajectory['yvel'],Trajectory['zvel']])

    besttwist = np.zeros([int(ntests**3.),11])

    ntest = 0
    for phi in range(0,ntests):
        for theta in range(0,ntests):
            for psi in range(0,ntests):
    
                PHIP    = phi   * (2*np.pi/ntests)
                THETA   = theta * (  np.pi/ntests)
                PSI     = psi   * (2*np.pi/ntests)

                Trans = return_euler_slater(PHIP, THETA, PSI, 1);
                
                twistedpos  = np.dot(Trans,TRAJMAT)
                twistedvel  = np.dot(Trans,VELMAT)
                
                # search for the minimum and save
                
                if all:
                    sigdiff = ( np.abs(lenscale*twistedpos[0]-targetx)/targetdx   + \
                                np.abs(lenscale*twistedpos[1]-targety)/targetdy   + \
                                np.abs(lenscale*twistedpos[2]-targetz)/targetdz   + \
                                np.abs(velscale*twistedvel[0]-targetvx)/targetdvx + \
                                np.abs(velscale*twistedvel[1]-targetvy)/targetdvy + \
                                np.abs(velscale*twistedvel[2]-targetvz)/targetdvz)
                else:
                    sigdiff = ( np.abs(lenscale*twistedpos[0]-targetx)/targetdx + \
                                np.abs(lenscale*twistedpos[1]-targety)/targetdy + \
                                np.abs(lenscale*twistedpos[2]-targetz)/targetdz)
                    
                #print(sigdiff)
                if verbose>0:
                    plt.plot(sigdiff,lw=0.5,color='black')

                mintime = np.nanmin(sigdiff)
                minindx = np.nanargmin(sigdiff)
                
                besttwist[ntest] = np.array([PHIP,THETA,PSI,mintime,minindx,\
                                             lenscale*twistedpos[0][minindx],\
                                             lenscale*twistedpos[1][minindx],\
                                             lenscale*twistedpos[2][minindx],\
                                             velscale*twistedvel[0][minindx],\
                                             velscale*twistedvel[1][minindx],\
                                             velscale*twistedvel[2][minindx]])
                    

                ntest += 1

    return besttwist



