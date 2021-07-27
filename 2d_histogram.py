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

load_branches = ['EcalScoringPlaneHits_v12.pdgID_', 'EcalScoringPlaneHits_v12.trackID_', 'EcalScoringPlaneHits_v12.x_', 'EcalScoringPlaneHits_v12.y_', 'EcalScoringPlaneHits_v12.z_', 'EcalScoringPlaneHits_v12.px_', 'EcalScoringPlaneHits_v12.py_', 'EcalScoringPlaneHits_v12.pz_']

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

        for i in range(len(table["EcalScoringPlaneHits_v12.pdgID_"])):
                
            if (i % 10000 == 0):
                print('Finished Event ' + str(i)) 
            
           # if (i > 5000):
           #     break

            for j in range(len(table["EcalScoringPlaneHits_v12.px_"][i])):
            
                if (table['EcalScoringPlaneHits_v12.pdgID_'][i][j] == 11) and \
                   (table['EcalScoringPlaneHits_v12.z_'][i][j] > 240) and \
                   (table['EcalScoringPlaneHits_v12.z_'][i][j] < 241) and \
                   (table['EcalScoringPlaneHits_v12.trackID_'][i][j] == 1) and \
                   (table['EcalScoringPlaneHits_v12.pz_'][i][j] > 0):
                    
                    recoilX  = table['EcalScoringPlaneHits_v12.x_'][i][j]
                    recoilY  = table['EcalScoringPlaneHits_v12.y_'][i][j]
                   
                    X.append(recoilX)
                    Y.append(recoilY)
             
    return X, Y

X_vals, Y_vals = getXY(files)
print("Done. Plotting...")
my_cmap = copy.copy(plt.cm.get_cmap("jet"))
my_cmap.set_under('white', 1)
plt.figure()
plt.hist2d(X_vals, Y_vals, bins=500, range=([-300,300],[-300,300]), cmin = 1,  cmap=my_cmap, norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')

plt.savefig('/home/dgj1118/plotting/plots/EcalSPHits_XY.png')
