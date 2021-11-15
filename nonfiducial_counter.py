from LDMX.Framework import EventTree
import sys
import numpy as np

tree = EventTree.EventTree(sys.argv[1])                                         # tree = all of the events
total = 0                                                                       # total number of events
fiducials = 0                                                                   # total number of fiducial events
nonfiducials = 0                                                                # total number of non-fiducial events
leftovers = 0                                                                   # total number of events marked fiducial from filter, non-fiducial from calculation

# important constants (mm)
EcalSP = 240.5005                                                               # z position of the ECal Scoring Plane
EcalFace = 248.35                                                               # z position of the ECal Face
cell_radius = 5.0                                                               # radius of a single ECal cell module

# important functions 
def projectionX(x,y,z,px,py,pz,zFinal):                                         # x projection from z to zFinal
    if (px == 0):
        return x + (zFinal - z)/99999
    else:
        return x + px/pz*(zFinal - z)

def projectionY(x,y,z,px,py,pz,zFinal):                                         # y projection from z to zFinal
    if (py == 0):
        return y + (zFinal - z)/99999
    else:
        return y + py/pz*(zFinal - z)

def dist(p1, p2):                                                               # distance between two points
    return np.sqrt(np.sum( ( np.array(p1) - np.array(p2) )**2 ))

def loadCellMap():                                                              # load the v12 cell map (contains the x-y coordinates each cell center)
    cellMap = {}
    for i, x, y in np.loadtxt('cellmodule.txt'):
        cellMap[i] = (x, y)
    global cells
    cells = np.array(list(cellMap.values()))
    print("Loaded detector info")

def loadLayers():                                                               # load all the z-positions of each layer of the ECal
    layerMap = {}
    for i, x in np.loadtxt('layer.txt'):
        layerMap[i] = x
    global layers
    layers = np.array(list(layerMap.values()))
    print("Loaded layer info")

def loadRadii():                                                                # load all the radii of containment at each layer of the ECal
    radiiVals = {}
    for i, x in np.loadtxt('radii.txt'):
        radiiVals[i] = x 
        global radii
        radii = np.array(list(radiiVals.values()))
        print("Loaded Radii of Containment")

####################################################################################################################################################################################################################################################################

loadCellMap()                                                                   # load the cell map -> 'cells' is the global variable for the values

loadLayers()                                                                    # load the layer values -> 'layers' is the global variable for the values

loadRadii()                                                                     # load the radii of containment -> 'radii' is the global varaible for the values

for event in tree:                                                              # loop through each event in the root file
    total += 1                                                                  # increment each event by 1
    is_fiducial = (event.EventHeader.getIntParameter('is_fiducial') == 1)       # is_fiducial: if the event is fiducial, return 1

    if (is_fiducial):                                                           # if the event is fiducial...
        fiducials += 1                                                          # increment the fiducial count by 1
        
        fiducial = False                                                        # assume it is non-fiducial until proven fiducial

        for hit in event.EcalScoringPlaneHits:                                  # loop through each hit of each "filter fiducial event"
            
            pdgID =  hit.getPdgID()                                             # pdg id of the hit
            trackID = hit.getTrackID()                                          # track id of the hit
            Z = hit.getPosition()[2]                                            # z position of the hit
            Pz = hit.getMomentum()[2]                                           # z momentum of the hit

            # we want the track of the recoil electron that is within the ECal Scoring Plane
            if (pdgID == 11 and trackID == 1 and Z > 240.500 and Z < 240.501 and Pz > 0):
                X = hit.getPosition()[0]                                        # x position of the hit
                Y = hit.getPosition()[1]                                        # y position of the hit
                Px = hit.getMomentum()[0]                                       # x momentum of the hit
                Py = hit.getMomentum()[1]                                       # y momentum of the hit

                # we want the x-y coordinate of the projection at the ECal Face
                finalXY = (projectionX(X,Y,Z,Px,Py,Pz,EcalFace), projectionY(X,Y,Z,Px,Py,Pz,EcalFace)) 

                # make sure the x and y coordinates and momenta are not marked as -9999 (this value is set for missed hits)
                if not X == -9999 and not Y == -9999 and not Px == -9999 and not Py == -9999:
                    
                    for cell in range(len(cells)):                              # loop through each cell on the ECal Face
                        celldis = dist(cells[cell], finalXY)                    # calculate the distance from the cell center to the projected x-y coordinate
                        if celldis <= cell_radius:                              # if the distance from the cell center is within the cell radius the event is fiducial
                            fiducial = True
                            break                                               # stop looping over the cells

        if (fiducial == False):                                                 
            leftovers += 1                                                      # increment the amount of fiducial events from the filter -> non-fiducial after calculation
            electronLayerIntercepts = []                                        # store the primary electron's layer intercepts
            energies = []                                                       # store the energies of each hit 
            amplitudes = []                                                     # store the amplitudes of each hit

            for hit in event.EcalScoringPlaneHits:                              # loop through each hit in the event
                
                pdgID = hit.getPdgID()                                          # pdg id
                trackID = hit.getTrackID()                                      # track id
                Z = hit.getPosition()[2]                                        # z position
                Pz = hit.getMomentum()[2]                                       # z momentum

                if (pdgID == 11 and trackID == 1 and Z > 240.500 and Z < 240.501 and Pz > 0):           # start from the ECal SP: 
                    X = hit.getPosition()[0]                                                            # x position of the hit
                    Y = hit.getPosition()[1]                                                            # y position of the hit
                    Px = hit.getMomentum()[0]                                                           # x momentum of the hit
                    Py = hit.getMomentum()[1]                                                           # y momentum of the hit
                        
                    for i in range(len(layers)):                                                                            # loop through each layer of the ECal
                        zInitial = Z                                                                                        # initial z position is at the ECal SP ~240.5 mm
                        zFinal = layers[i]                                                                                  # final z position will increment to the next layer each time
                        xyFinal = (projectionX(X,Y,zInitial,Px,Py,Pz,zFinal),projectionX(X,Y,zInitial,Px,Py,Pz,zFinal))     # x-y position at the final z position (projected from ECal SP)
                        electronLayerIntercepts.append(xyFinal)                                                             # add the x-y position to the electronLayerIntercepts list

            for hit in event.EcalRecHits:                                                                                   # loop through each hit in the event
                zPos = hit.getZPos()                                                                                        # z position of the hit

                for i in range(len(layers)):                                                                                # loop through each layer
                    
                    if (zPos > (layers[i] - 0.005) and zPos < (layers[i] + 0.005)):                                         # select the event that appears in the ith layer
                        xPos = hit.getXPos()                                                                                # x position of the hit
                        yPos = hit.getYPos()                                                                                # y position of the hit
                        xyPos = (xPos,yPos)                                                                                 # (x,y) coordinate of the hit
                        
                        if (dist(xyPos,electronLayerIntercepts[i]) <= radii[i]):                                            # check if the hit is within the radius of containment for that layer w.r.t the electron (x,y) coordinate
                            energy = hit.getEnergy()                                                                        # energy of the hit
                            amplitude = hit.getAmplitude()                                                                  # amplitude of the hit
                            energies.append(energy)
                            amplitudes.append(amplitude)


            '''
            for hit2 in event.EcalSimHits:                                      # loop over every hit in EcalSimHits
                contrib = hit2.getContrib(0)                                    # contrib of the hit
                trackid = contrib.trackID                                       # track id of the hit
                pdgid = contrib.pdgCode                                          # pdg id of the hit

                if (pdgid == 11 and trackid == 1):                              # make sure it is the recoil electron
                    print("track id: " + str(trackid))
                    print("pdg id: " + str(pdgid))
                    eDep = hit2.getEdep()                                       # calculate the energy deposited in the ECal
                    print("Energy deposited in the ECal: " + str(eDep))         
            '''

    elif (is_fiducial != 1):                                                    # if the event is non-fiducial...
        nonfiducials += 1                                                       # increment the non-fiducial count by 1

print("=== Summary ===")
print("Total number of events: " + str(total))
print("Total number of fiducial events: " + str(fiducials))
print("Total number of non-fiducial events: " + str(nonfiducials))
print("Total number of fiducial events from filter but non-fiducial from calculation: " + str(leftovers))
