# Plotting Script for Plotting 1-D Histograms
import numpy as np
import glob
import uproot
import matplotlib.pyplot as plt
import concurrent.futures
import copy
import matplotlib
executor = concurrent.futures.ThreadPoolExecutor(12)

base = '/home/dgj1118/LDMX-scripts/GraphNet/background_230_trunk/evaluation/'
files = glob.glob(base+'4gev_v12_pn_enlarged_191_ldmx-det-v12_run91_seeds_182_183_None.root')

load_branches = ['TargetScoringPlaneHits_v12.x_', 'TargetScoringPlaneHits_v12.y_',
 'TargetScoringPlaneHits_v12.z_', 'TargetScoringPlaneHits_v12.px_', 
 'TargetScoringPlaneHits_v12.py_', 'TargetScoringPlaneHits_v12.pz_', 
 'TargetScoringPlaneHits_v12.trackID_', 'TargetScoringPlaneHits_v12.pdgID_',
 'EcalScoringPlaneHits_v12.x_', 'EcalScoringPlaneHits_v12.y_',
 'EcalScoringPlaneHits_v12.z_', 'EcalScoringPlaneHits_v12.px_', 
 'EcalScoringPlaneHits_v12.py_', 'EcalScoringPlaneHits_v12.pz_', 
 'EcalScoringPlaneHits_v12.trackID_', 'EcalScoringPlaneHits_v12.pdgID_']

# Constants
EcalSP = 240.5015
EcalFace = 248.35
cell_radius = 5.0

# Projection Functions 
def projectionX(x,y,z,px,py,pz,zFinal):
    if (px == 0):
        return x + (zFinal - z)/99999
    else:
        return x + px/pz*(zFinal - z)

def projectionY(x,y,z,px,py,pz,zFinal):
    if (py == 0):
        return y + (zFinal - z)/99999
    else:
        return y + py/pz*(zFinal - z)

# Distance Function
def dist(p1, p2):
    return np.sqrt(np.sum( ( np.array(p1) - np.array(p2) )**2 ))

# Load the Cell Map
def loadCellMap():
    cellMap = {}
    for i, x, y in np.loadtxt('/home/dgj1118/plotting/cellmodule.txt'):
        cellMap[i] = (x, y)
    global cells
    cells = np.array(list(cellMap.values()))
    print("Loaded detector info")

# Function for getting the magnitudes of the recoil angles
def getANGLES(filelist):
      
    print("Reading files")
    
    Angles = []
    total_events = 0
       
    for f in filelist:
        print("    File: {}".format(f))
        t = uproot.open(f)['LDMX_Events']
        if len(t.keys()) == 0:
            print("    File empty, skipping")
        table_temp = t.arrays(expressions=load_branches, interpretation_executor=executor)
        table = {}
        for k in load_branches:
            table[k] = table_temp[k]

        print('    1. Primary Cut')
        
        # filter out non-fiducial/fiducial events
        cut = np.zeros(len(table['EcalScoringPlaneHits_v12.pdgID_']), dtype=bool)
        for event in range(len(table['EcalScoringPlaneHits_v12.pdgID_'])):

            fiducial = False
                        
            for hit in range(len(table['EcalScoringPlaneHits_v12.pdgID_'][event])):
                if ((table['EcalScoringPlaneHits_v12.pdgID_'][event][hit] == 11) and
                   (table['EcalScoringPlaneHits_v12.trackID_'][event][hit] == 1) and 
                   (table['EcalScoringPlaneHits_v12.z_'][event][hit] > 240) and
                   (table['EcalScoringPlaneHits_v12.z_'][event][hit] < 241) and
                   (table['EcalScoringPlaneHits_v12.pz_'][event][hit] > 0)): 

                    recoilX = table['EcalScoringPlaneHits_v12.x_'][event][hit]
                    recoilY = table['EcalScoringPlaneHits_v12.y_'][event][hit]
                    recoilPx = table['EcalScoringPlaneHits_v12.px_'][event][hit]
                    recoilPy = table['EcalScoringPlaneHits_v12.py_'][event][hit]
                    recoilPz = table['EcalScoringPlaneHits_v12.pz_'][event][hit]

                    # check if it's non-fiducial/fiducial
                    finalXY = (projectionX(recoilX,recoilY,EcalSP,recoilPx,recoilPy,recoilPz,EcalFace),projectionY(recoilX,recoilY,EcalSP,recoilPx,recoilPy,recoilPz,EcalFace))
                    if not recoilX == -9999 and not recoilY == -9999 and not recoilPx == -9999 and not recoilPy == -9999:
                        for cell in range(len(cells)):
                            celldis = dist(cells[cell], finalXY)
                            if celldis <= cell_radius:
                                fiducial = True
                                break
                    
            if fiducial == True: # filter for non-fiducial/fiducial
                cut[event] = 1
            
            if (event % 10000 == 0):
                print('    Finished Event ' + str(event))

        # perform the cut on the table values
        for k in load_branches:
            table[k] = table[k][cut]
        
        print('     -> Finished.')
              
        print('    2. Retrieving angles')

        total_events += len(table['TargetScoringPlaneHits_v12.z_'])

        for event in range(len(table['TargetScoringPlaneHits_v12.z_'])):
            
            for hit in range(len(table['TargetScoringPlaneHits_v12.z_'][event])):
                
                if (table['TargetScoringPlaneHits_v12.z_'][event][hit] > -1.7535 and \
                table['TargetScoringPlaneHits_v12.z_'][event][hit] < 1.7535 and \
                table['TargetScoringPlaneHits_v12.trackID_'][event][hit] == 1 and \
                table['TargetScoringPlaneHits_v12.pdgID_'][event][hit] == 11):
                    
                    # Position and Momentum values
                    X = table['TargetScoringPlaneHits_v12.x_'][event][hit]
                    Y = table['TargetScoringPlaneHits_v12.y_'][event][hit]
                    Z = table['TargetScoringPlaneHits_v12.z_'][event][hit]
                    pX = table['TargetScoringPlaneHits_v12.px_'][event][hit]
                    pY = table['TargetScoringPlaneHits_v12.py_'][event][hit]
                    pZ = table['TargetScoringPlaneHits_v12.pz_'][event][hit]
                    
                    # Calculate recoil angle (in degrees)
                    theta = abs(np.arccos(pZ / np.sqrt(pX**2 + pY**2 + pZ**2)) * 180 / np.pi)
                    Angles.append(theta)
                    break
    
            if (event % 10000 == 0):
                print('        Finished loading event number ' + str(event))
        
        print('    -> Finished.')
            
    return Angles, total_events

if __name__ == '__main__':  
    print('--- 1D Histogram Plotting Program ---')
    loadCellMap() # Load Cell Map
    vals, num = getANGLES(files) # Get Recoil Angles
    print()
    print('=== General Info ===')
    print('Total number of events: ' + str(num))
    print('Maximum recoil angle: ' + str(max(vals)))
    print('Minimum recoil angle: ' + str(min(vals)))   

    bin_list = np.linspace(0,1.5,150)
    print("Done. Plotting...")
    plt.figure()
    plt.hist(vals, bins=bin_list, range=(0,1.5))
    plt.xlabel('Recoil Angles (degrees)')
    plt.title('Magnitude of Recoil Angles at Target SP (fiducial)')
    plt.savefig('/home/dgj1118/plotting/plots/TargetSP_Angles(F2).png') # Save Image

