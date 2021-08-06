# Plotting Script for ROC Curves
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

target_branches = ['TargetScoringPlaneHits_v12.x_', 'TargetScoringPlaneHits_v12.y_',
 'TargetScoringPlaneHits_v12.z_', 'TargetScoringPlaneHits_v12.px_', 
 'TargetScoringPlaneHits_v12.py_', 'TargetScoringPlaneHits_v12.pz_', 
 'TargetScoringPlaneHits_v12.trackID_', 'TargetScoringPlaneHits_v12.pdgID_']

ecal_branches = ['EcalScoringPlaneHits_v12.x_', 'EcalScoringPlaneHits_v12.y_',
 'EcalScoringPlaneHits_v12.z_', 'EcalScoringPlaneHits_v12.px_', 
 'EcalScoringPlaneHits_v12.py_', 'EcalScoringPlaneHits_v12.pz_', 
 'EcalScoringPlaneHits_v12.trackID_', 'EcalScoringPlaneHits_v12.pdgID_']

# Constants (mm)
EcalSP = 240.5005
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
    return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))

# Load the Cell Map
def loadCellMap():
    cellMap = {}
    for i, x, y in np.loadtxt('/home/dgj1118/plotting/cellmodule.txt'):
        cellMap[i] = (x, y)
    global cells
    cells = np.array(list(cellMap.values()))
    print("Loaded detector info")

def getANGLES(filelist):
      
    print("Reading files")
    
    nf_Angles = []
    f_Angles = []
    nf_events = 0
    f_events = 0

    for f in filelist:
        print("    File: {}".format(f))
        t = uproot.open(f)['LDMX_Events']
        if len(t.keys()) == 0:
            print("    File empty, skipping")
        table_temp = t.arrays(expressions=ecal_branches, interpretation_executor=executor)
        table = {}
        for k in ecal_branches:
            table[k] = table_temp[k]
        
        print('    1. Primary Cut')
        
        # filter out non-fiducial/fiducial events
        nf_cut = np.zeros(len(table['EcalScoringPlaneHits_v12.pdgID_']), dtype=bool)
        f_cut = np.zeros(len(table['EcalScoringPlaneHits_v12.pdgID_']), dtype=bool)

        for event in range(len(table['EcalScoringPlaneHits_v12.pdgID_'])):

            fiducial = False
                        
            for hit in range(len(table['EcalScoringPlaneHits_v12.pdgID_'][event])):
                if ((table['EcalScoringPlaneHits_v12.pdgID_'][event][hit] == 11) and
                   (table['EcalScoringPlaneHits_v12.trackID_'][event][hit] == 1) and 
                   (table['EcalScoringPlaneHits_v12.z_'][event][hit] > 240.500) and
                   (table['EcalScoringPlaneHits_v12.z_'][event][hit] < 240.501) and
                   (table['EcalScoringPlaneHits_v12.pz_'][event][hit] > 0)): 

                    recoilX = table['EcalScoringPlaneHits_v12.x_'][event][hit]
                    recoilY = table['EcalScoringPlaneHits_v12.y_'][event][hit]
                    recoilZ = table['EcalScoringPlaneHits_v12.z_'][event][hit]
                    recoilPx = table['EcalScoringPlaneHits_v12.px_'][event][hit]
                    recoilPy = table['EcalScoringPlaneHits_v12.py_'][event][hit]
                    recoilPz = table['EcalScoringPlaneHits_v12.pz_'][event][hit]

                    # check if it's non-fiducial/fiducial
                    finalXY = (projectionX(recoilX,recoilY,recoilZ,recoilPx,recoilPy,recoilPz,EcalFace),projectionY(recoilX,recoilY,recoilZ,recoilPx,recoilPy,recoilPz,EcalFace))
                    if not recoilX == -9999 and not recoilY == -9999 and not recoilPx == -9999 and not recoilPy == -9999:
                        for cell in range(len(cells)):
                            celldis = dist(cells[cell], finalXY)
                            if celldis <= cell_radius:
                                fiducial = True
                                break
                    
            if fiducial == False: # filter for non-fiducial
                nf_cut[event] = 1
            
            if fiducial == True: # filter for fiducial
                f_cut[event] = 1
            
            if (event % 10000 == 0):
                print('    Finished Event ' + str(event))

        # perform the cuts on the fiducial and nonfiducial dictionaries
        nf_table = {}
        f_table = {}
        table_temp2 = t.arrays(expressions=target_branches, interpretation_executor=executor)
        
        for k in target_branches:
            nf_table[k] = table_temp2[k][nf_cut]
        
        for k in target_branches:
            f_table[k] = table_temp2[k][f_cut]

        print('     -> Finished.')
              
        print('    2. Retrieving angles')
        print('        a. Nonfiducial')
        
        nf_events += len(nf_table['TargetScoringPlaneHits_v12.z_'])

        for event in range(len(nf_table['TargetScoringPlaneHits_v12.z_'])):
            
            for hit in range(len(nf_table['TargetScoringPlaneHits_v12.z_'][event])):
                
                if (nf_table['TargetScoringPlaneHits_v12.z_'][event][hit] < 0.1777 and
                nf_table['TargetScoringPlaneHits_v12.z_'][event][hit] > 0.1757 and
                nf_table['TargetScoringPlaneHits_v12.trackID_'][event][hit] == 1 and
                nf_table['TargetScoringPlaneHits_v12.pdgID_'][event][hit] == 11):
                    
                    # Position and Momentum values
                    X = nf_table['TargetScoringPlaneHits_v12.x_'][event][hit]
                    Y = nf_table['TargetScoringPlaneHits_v12.y_'][event][hit]
                    Z = nf_table['TargetScoringPlaneHits_v12.z_'][event][hit]
                    pX = nf_table['TargetScoringPlaneHits_v12.px_'][event][hit]
                    pY = nf_table['TargetScoringPlaneHits_v12.py_'][event][hit]
                    pZ = nf_table['TargetScoringPlaneHits_v12.pz_'][event][hit]
                    
                    # Calculate recoil angle (in degrees)
                    theta = abs(np.arccos(pZ / np.sqrt(pX**2 + pY**2 + pZ**2)) * 180 / np.pi)
                    nf_Angles.append(theta)                    
                    break
    
            if (event % 10000 == 0):
                print('        Finished loading event number ' + str(event))
        
        print('    -> Finished.')
        print('        b. Fiducial')
        
        f_events += len(f_table['TargetScoringPlaneHits_v12.z_'])

        for event in range(len(f_table['TargetScoringPlaneHits_v12.z_'])):
            
            for hit in range(len(f_table['TargetScoringPlaneHits_v12.z_'][event])):
                
                if (f_table['TargetScoringPlaneHits_v12.z_'][event][hit] < 0.1777 and
                f_table['TargetScoringPlaneHits_v12.z_'][event][hit] > 0.1757 and
                f_table['TargetScoringPlaneHits_v12.trackID_'][event][hit] == 1 and
                f_table['TargetScoringPlaneHits_v12.pdgID_'][event][hit] == 11):
                    
                    # Position and Momentum values
                    X = f_table['TargetScoringPlaneHits_v12.x_'][event][hit]
                    Y = f_table['TargetScoringPlaneHits_v12.y_'][event][hit]
                    Z = f_table['TargetScoringPlaneHits_v12.z_'][event][hit]
                    pX = f_table['TargetScoringPlaneHits_v12.px_'][event][hit]
                    pY = f_table['TargetScoringPlaneHits_v12.py_'][event][hit]
                    pZ = f_table['TargetScoringPlaneHits_v12.pz_'][event][hit]
                    
                    # Calculate recoil angle (in degrees)
                    theta = abs(np.arccos(pZ / np.sqrt(pX**2 + pY**2 + pZ**2)) * 180 / np.pi)
                    f_Angles.append(theta)                    
                    break
    
            if (event % 10000 == 0):
                print('        Finished loading event number ' + str(event))
        
        print('    -> Finished.')

    return nf_Angles, f_Angles, nf_events, f_events

def getXYROC(signal, background, cut):
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0

    for angle in signal: 
        if (angle > cut): # True Positives
            truePos += 1
        if (angle < cut): # False Negatives
            falseNeg += 1
    
    for angle in background: 
        if (angle > cut): # False Positives
            falsePos += 1
        if (angle < cut): # True Negatives
            trueNeg += 1
    
    sensitivity = truePos / (truePos + falseNeg) # True Positive Rate = True Positives / (True Positives + False Negatives)
    specificity = falsePos / (falsePos + trueNeg) # False Positive Rate = False Positives / (False Positives + True Negatives)

    return specificity, sensitivity

if __name__ == '__main__':  
    print('--- ROC Curve Plotting Program ---')
    loadCellMap() # Load Cell Map
    nf_vals, f_vals, nf_num, f_num = getANGLES(files) # Get Recoil Angles
    
    # Print out details of events
    print()
    print('=== General Info ===')
    print('Total number of events: ' + str(nf_num + f_num))
    print('Total number of nonfiducial events: ' + str(nf_num))
    print('Total number of fiducial events: ' + str(f_num))

    # get True/False positive rate values and find the cut for the best ratio
    x_vals = []
    y_vals = []
    max_ratio = 0
    cut_value = 0
    for i in np.linspace(0,85,8500):
        x,y = getXYROC(nf_vals, f_vals, i) 
        x_vals.append(x)
        y_vals.append(y)
        if (x != 0 and y/x > max_ratio):
            cut_value = i
            max_ratio = y/x    
    print('The max ratio of True Positive Rate/False Positive Rate is: ' + str(max_ratio))
    print('The best recoil angle cut value is: ' + str(cut_value) + ' degrees')
    
    # Plot the ROC curve
    plt.figure() 
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Signal Efficiency for Electron Recoil Angle')
    plt.plot(x_vals, y_vals, marker='.')
    plt.savefig('/home/dgj1118/plotting/plots/ROC.png') # Save Image
