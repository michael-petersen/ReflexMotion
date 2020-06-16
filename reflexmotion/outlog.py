import numpy as np



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



