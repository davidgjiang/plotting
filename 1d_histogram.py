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

load_branches = ['TargetScoringPlaneHits_v12.x_', 'TargetScoringPlaneHits_v12.y_', 'TargetScoringPlaneHits_v12.z_', 'TargetScoringPlaneHits_v12.px_', 'TargetScoringPlaneHits_v12.py_', 'TargetScoringPlaneHits_v12.pz_', 'TargetScoringPlaneHits_v12.trackID_', 'TargetScoringPlaneHits_v12.pdgID_']

def getANGLES(filelist):
      
    print("Reading files")
    
    Angles = []
       
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
        
        for event in range(10): #range(len(table['TargetScoringPlaneHits_v12.z_'])):

            #if (event % 10000 == 0):
            #    print('Finished loading event number ' + str(event))

            #if (event == 10000):    
            #    break
            
            for hit in range(10): #range(len(table['TargetScoringPlaneHits_v12.z_'][event])):
                if (table['TargetScoringPlaneHits_v12.z_'][event][hit] > -1.7535 and \
                table['TargetScoringPlaneHits_v12.z_'][event][hit] < 1.7535 and \
                table['TargetScoringPlaneHits_v12.trackID_'][event][hit] == 1 and \
                table['TargetScoringPlaneHits_v12.pdgID_'][event][hit] == 11):
                    # Calculate recoil angle (in degrees)
                    X = table['TargetScoringPlaneHits_v12.x_'][event][hit]
                    Y = table['TargetScoringPlaneHits_v12.y_'][event][hit]
                    Z = table['TargetScoringPlaneHits_v12.z_'][event][hit]
                    pX = table['TargetScoringPlaneHits_v12.px_'][event][hit]
                    pY = table['TargetScoringPlaneHits_v12.py_'][event][hit]
                    pZ = table['TargetScoringPlaneHits_v12.pz_'][event][hit]
                    costheta = (np.sqrt(pX**2 + pY**2 + pZ**2) / pZ) * (180/3.1415926)
                    #Angles.append(theta)
                    print('event: ' + str(event))
                    print('X: ' + str(X))
                    print('Y: ' + str(Y))
                    print('Z: ' + str(Z))
                    print('pX: ' + str(pX))
                    print('pY: ' + str(pY))
                    print('pZ: ' + str(pZ))
                    print('cos(theta): ' + str(costheta))
                    print()
                    break
    
    #return Angles
    

    print('Finished selection')   


getANGLES(files)

'''
bin_list = list(range(-30,31))
print("Done. Plotting...")
plt.figure()
plt.hist(vals, bins=bin_list, range=(-30,30))
plt.xlabel('Recoil Angles (degrees)')
plt.savefig('/home/dgj1118/plotting/plots/TargetSP_Angles.png')
'''
                

        
    




