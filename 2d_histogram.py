# Plotting Script for Plotting 2-D Histograms
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

load_branches = ['TargetScoringPlaneHits_v12.x_', 'TargetScoringPlaneHits_v12.y_', 'TargetScoringPlaneHits_v12.z_', 'TargetScoringPlaneHits_v12.px_', 'TargetScoringPlaneHits_v12.py_', 'TargetScoringPlaneHits_v12.pz_', 'TargetScoringPlaneHits_v12.trackID_', 'TargetScoringPlaneHits_v12.pdgID_']

# Projection Functions 
def projectionX(x,y,z,px,py,pz):
    EcalSP = 240.5015
    if (px == 0):
        return x + (EcalSP - z)/99999
    else:
        return x + px/pz*(EcalSP - z)

def projectionY(x,y,z,px,py,pz):
    EcalSP = 240.5015
    if (py == 0):
        return y + (EcalSP - z)/99999
    else:
        return y + py/pz*(EcalSP - z)

# Function for getting X and Y values 
def getXY(filelist):

    print("Reading files")
    
    X = [] 
    Y = []
    
    for f in filelist:
        print("    Reading file {}".format(f))
        t = uproot.open(f)['LDMX_Events']
        if len(t.keys()) == 0:
            print("    File empty, skipping")
        table_temp = t.arrays(expressions=load_branches, interpretation_executor=executor)
        table = {}
        for k in load_branches:
            table[k] = table_temp[k]
        
        print('Starting selection')

        for event in range(len(table["TargetScoringPlaneHits_v12.pdgID_"])):
                            
            for hit in range(len(table["TargetScoringPlaneHits_v12.px_"][event])):
                
                # check if it's an electron with nonzero z-momentum
                if ((table['TargetScoringPlaneHits_v12.pdgID_'][event][hit] == 11) and
                   (table['TargetScoringPlaneHits_v12.trackID_'][event][hit] == 1) and 
                   (table['TargetScoringPlaneHits_v12.z_'][event][hit] > -1.7535) and
                   (table['TargetScoringPlaneHits_v12.z_'][event][hit] < 1.7535) and
                   (table['TargetScoringPlaneHits_v12.pz_'][event][hit] > 0)):
                    
                    x_ = table['TargetScoringPlaneHits_v12.x_'][event][hit]
                    y_ = table['TargetScoringPlaneHits_v12.y_'][event][hit]
                    z_ = table['TargetScoringPlaneHits_v12.z_'][event][hit]
                    px_ = table['TargetScoringPlaneHits_v12.px_'][event][hit]
                    py_ = table['TargetScoringPlaneHits_v12.py_'][event][hit]
                    pz_ = table['TargetScoringPlaneHits_v12.pz_'][event][hit]
                    
                    # calculate the projected x and y values
                    finalX = projectionX(x_,y_,z_,px_,py_,pz_)
                    finalY = projectionY(x_,y_,z_,px_,py_,pz_)
                    X.append(finalX)
                    Y.append(finalY)
                    break
                
            if (event % 10000 == 0):
                print('Finished Event ' + str(event))

    return X, Y

print('--- 2D Histogram Plotting Program ---')
X_vals, Y_vals, total = getXY(files)
print("Done. Plotting...")
my_cmap = copy.copy(plt.cm.get_cmap("jet"))
my_cmap.set_under('white', 1)
plt.figure()
plt.hist2d(X_vals, Y_vals, bins=500, range=([-300,300],[-300,300]), cmin = 1,  cmap=my_cmap, norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')

plt.savefig('/home/dgj1118/plotting/plots/EcalSPHits_XY.png')
