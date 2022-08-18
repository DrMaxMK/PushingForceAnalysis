import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import trackpy as tp
from numba import jit
import matplotlib.collections as collections


@jit
def dy_dx(t3, i, diffmode=3, filtr_small=0, loud = False, neutral_frames=[0,0,0]):
    #diffmode 1: running average calculation
    #diffmode 2: against previous frame calculation
    #diffmode 3: against some neutral position calculation
    #diffmode 4: against some neutral position calculation with drift correction
    #diffmode 5: against 3 neutral position calculation with drift correction
    #filtr_small is length in pixels under which u filter out dx dy. i.e. filtr_small=1 filters our every movement under 1 pixel 
    
    part = t3.iloc[i]["particle"]
    y_old = t3.iloc[i]["y"]
    x_old = t3.iloc[i]["x"]
    frame_old = int(t3.iloc[i]["frame"]) #otherwise stuff like 0.0 
    neutral_frame = neutral_frames[0]
    
    
    if loud:
        print(frame_old, t3["frame"].max())
        
    if frame_old < t3["frame"].max()-0.5:
        y_new = t3[((t3["particle"]== part) & (t3["frame"]== frame_old+1))]["y"]
        x_new = t3[((t3["particle"]== part) & (t3["frame"]== frame_old+1))]["x"]
        #print(y_new, x_new)
        if loud:
            1+1
            #print("\n yold: \n", y_old,"\n xold: \n", x_old,"\n ynew: \n", y_new,"\n xnew: \n", x_new, "\n da lengths: ", len(y_new),len(x_new))

        if (len(y_new)!=0) & (len(x_new)!=0):
            if diffmode==1: #running avg
                y_avg = t3[((t3["particle"]== part) & (t3["frame"]<= 100))]["y"].mean()
                x_avg = t3[((t3["particle"]== part) & (t3["frame"]<= 100))]["x"].mean()
                dyObj = (y_new - y_avg) 
                dxObj = (x_new - x_avg)
            elif diffmode==2: #against old frame
                y_old = t3[((t3["particle"]== part) & (t3["frame"]== frame_old))]["y"]#.mean()
                x_old = t3[((t3["particle"]== part) & (t3["frame"]== frame_old))]["x"]#.mean()
                
                dyObj = (y_new.iloc[0] - y_old.iloc[0]) 
                dxObj = (x_new.iloc[0] - x_old.iloc[0])
                #dyObj = (y_new - y_avg) 
                #dxObj = (x_new - x_avg)
            elif diffmode==3: #against neutral position
                neutral_frame = neutral_frame
                y_dif = t3[((t3["particle"]== part) & (t3["frame"]== neutral_frame))]["y"]
                x_dif = t3[((t3["particle"]== part) & (t3["frame"]== neutral_frame))]["x"]
                #print(y_new, y_dif)
                if (len(y_dif)!=0) & (len(x_dif)!=0):
                    #iloc[0] or what used to work was dyObj[frame_old+1]
                    #print(y_new)
                    dyObj = (y_new[frame_old+1] - y_dif.iloc[0]) 
                    dxObj = (x_new[frame_old+1] - x_dif.iloc[0])
                else:
                    return 0
            elif diffmode==4: #against neutral position with drift correction
                # returns the true xy coordinates but dx dy are calculated minus the drift
                drift = tp.compute_drift(t3)
                t_drift = tp.subtract_drift(t3.copy(), drift)
                neutral_frame = neutral_frame
                #TODO change t3 to drift corrected t3 here and use diffmode 3 code for rest:)
                y_new = t_drift[((t_drift["particle"]== part) & (t_drift["frame"]== frame_old+1))]["y"]
                x_new = t_drift[((t_drift["particle"]== part) & (t_drift["frame"]== frame_old+1))]["x"]
                y_dif = t_drift[((t_drift["particle"]== part) & (t_drift["frame"]== neutral_frame))]["y"]
                x_dif = t_drift[((t_drift["particle"]== part) & (t_drift["frame"]== neutral_frame))]["x"]
                #print(y_new, y_dif)
                if (len(y_dif)!=0) & (len(x_dif)!=0):
                    #iloc[0] or what used to work was dyObj[frame_old+1]
                    #print(y_new)
                    dyObj = (y_new[frame_old+1] - y_dif.iloc[0]) 
                    dxObj = (x_new[frame_old+1] - x_dif.iloc[0])
                else:
                    return 0
            elif diffmode==5: #against neutral position with drift correction and some more timepoints
                # returns the true xy coordinates but dx dy are calculated minus the drift
                drift = tp.compute_drift(t3)
                t_drift = tp.subtract_drift(t3.copy(), drift)
                t00, t01, t02 = neutral_frames 
                neutral_frame = t00
                if frame_old > t01:
                    neutral_frame = t01
                    if frame_old > t02:
                        #1+1
                        neutral_frame = t02
                #TODO change t3 to drift corrected t3 here and use diffmode 3 code for rest:)
                y_new = t_drift[((t_drift["particle"]== part) & (t_drift["frame"]== frame_old+1))]["y"]
                x_new = t_drift[((t_drift["particle"]== part) & (t_drift["frame"]== frame_old+1))]["x"]
                y_dif = t_drift[((t_drift["particle"]== part) & (t_drift["frame"]== neutral_frame))]["y"]
                x_dif = t_drift[((t_drift["particle"]== part) & (t_drift["frame"]== neutral_frame))]["x"]
                #print(y_new, y_dif)
                if (len(y_dif)!=0) & (len(x_dif)!=0):
                    #iloc[0] or what used to work was dyObj[frame_old+1]
                    #print(y_new)
                    dyObj = (y_new[frame_old+1] - y_dif.iloc[0]) 
                    dxObj = (x_new[frame_old+1] - x_dif.iloc[0])
                else:
                    return 0
                
                #iloc[frame_old] because iloc starts counting from 0 and not like frames from 1 so its frame_old+1-1
                #y_avg = t3[((t3["particle"]== part) & (t3["frame"]== neutral_frame))]["y"] 
                #x_avg = t3[((t3["particle"]== part) & (t3["frame"]== neutral_frame))]["x"]
                #dyObj = (y_new - y_avg) - drift.iloc[frame_old][0] 
                #dxObj = (x_new - x_avg) - drift.iloc[frame_old][1]
                #dyObj = (y_new[frame_old+1] - y_old)  #- drift.iloc[frame_old][0] 
                #dxObj = (y_new[frame_old+1] - y_old) #- drift.iloc[frame_old][1] 
            else: #should be the same as diffmode 2
                dyObj = (y_new - y_old) 
                dxObj = (x_new - x_old)

            index = len(t3.iloc[0])-1
            if  (dyObj**2 + dxObj**2 < filtr_small**2):#if u filter and its small then set to zero, 1µm/(0.3µm/px)=3px
                t3.iloc[i, index-1] = 0
                t3.iloc[i, index] = 0  
            else:
                t3.iloc[i, index-1] = dyObj #the difference of two pandas is some series shit so this [1] is needed
                t3.iloc[i, index] = dxObj
            if loud:
                print("dy, dx: ", dyObj, dxObj)

    else:
        1+1;

        
@jit        
def add_dy_dx(df, diffmode=2, filtr_small=0, loud=False, neutral_frames=[0,0,0]):
    t3 = df.copy()
    t3.insert(len(t3.iloc[0]), "dy", np.zeros(len(t3)))
    t3.insert(len(t3.iloc[0]), "dx", np.zeros(len(t3)))
    for i in range(len(t3)):
        dy_dx(t3, i, diffmode, filtr_small, loud, neutral_frames)
    print("done")
    return t3




def vector_clustering(vec):
    #get start and end of range where im in green 
    #save those ranges 
    #fit only in those?
    clustr_vec = np.zeros(len(vec))
    n = int #def as int so works as index afterwards
    if vec[0] == 1:
        n=1
    else:
        n=0
    diff_vec = np.diff(np.multiply(vec, 1))
    for i in range(len(vec)-1):
        if (diff_vec[i] == 1): #cluster starts
            n = n+1
            clustr_vec[i] = n
        elif (diff_vec[i] == 0) & (vec[i]!=0): #inside cluster
            clustr_vec[i] = n
        elif diff_vec[i] == -1: #clsuter ends
            clustr_vec[i] = 0 #not for readability but not neccessary bc called the vector as np zeros
            if  diff_vec[i-1] ==1: #dont want a cluster of one element
                n=n-1 
                clustr_vec[i-1] = 0
    return clustr_vec


def plot_cluster_fits(x, y_disps, in_bridge, daFile, loud=0):

    fig, ax = plt.subplots()
    #### Maximum
    ax.plot(x[:-1], y_disps[:-1,0], label="|dy| maximum", c="b", ls="-")

    # large timescale fit
    x_green = in_bridge * np.arange(0, len(in_bridge))
    linear_model=np.polyfit(x_green[x_green!=0], (y_disps[x_green])[x_green!=0,0], 1)
    linear_model_fn=np.poly1d(linear_model)
    if (loud): 
        print(f"Maximum Large Timescale slope: {linear_model[0]/2} um/min") #bc t axis is in 2min steps
    ax.plot(x,linear_model_fn(x), c="b", label="trend of maxima", ls="--")
    
    #short timescale fits
    clustrs = vector_clustering(in_bridge)
    n_max = clustrs.max()
    for i in np.arange(int(n_max))+1:
        linear_model=np.polyfit(x_green[clustrs==i], (y_disps[x_green])[clustrs==i, 0], 1)
        linear_model_fn=np.poly1d(linear_model)
        if (loud): 
            print(f"Maximum Short Timescale slope: {linear_model[0]/2} um/min")
        ax.plot((x[0:len(clustrs)])[clustrs==i],linear_model_fn((x[0:len(clustrs)])[clustrs==i]), c="b",ls="--")


    ##### Mean
    ax.plot(y_disps[:-1,1], label="|dy| mean", c="g", ls="-")
    
    linear_model=np.polyfit(x_green[x_green!=0], (y_disps[x_green])[x_green!=0,1], 1)
    linear_model_fn=np.poly1d(linear_model)
    if (loud): 
            print(f"Mean Long Timescale slope: {linear_model[0]/2} um/min")
    ax.plot(x,linear_model_fn(x), c="g", label="trend of means", ls="--")
    
    #short timescale fits
    for i in np.arange(int(n_max))+1:
        linear_model=np.polyfit(x_green[clustrs==i], (y_disps[x_green])[clustrs==i, 1], 1)
        linear_model_fn=np.poly1d(linear_model)
        if (loud): 
            print(f"Mean Short Timescale slope: {linear_model[0]/2} um/min")
        ax.plot((x[0:len(clustrs)])[clustrs==i],linear_model_fn((x[0:len(clustrs)])[clustrs==i]), c="g",ls="--")
    

    #ax.plot(x_disps[:,0], label="|dx| maximum")
    #ax.plot(x_disps[:,1], ls="--", label="|dx| mean")
    ax.legend()
    ax.set_ylabel("Displacement [µm]")


    collection = collections.BrokenBarHCollection.span_where(
        x, ymin=0, ymax=(ax.get_ylim()[1]), where=in_bridge, facecolor='green', alpha=0.2)
    ax.add_collection(collection)

    collection = collections.BrokenBarHCollection.span_where(
        x, ymin=0, ymax=(ax.get_ylim()[1]), where=in_bridge==False, facecolor='red', alpha=0.4)
    ax.add_collection(collection)
    
    ax.set_xticks(ax.get_xticks()[1:-2]) #this could be deleted but it gets rid of a warning
    ax.set_xticklabels(np.array(np.array(ax.get_xticks())*2).astype(int)) # to get away the 2min increment
    ax.set_xlabel("Time [min]")
    ax.set_title(str(daFile).split("/")[-1].split(".")[0]) 
    fig.savefig(daFile + "_E_of_t_whole_movie_aut.png", dpi=300)
    fig.show()
    
# only the maximum showing but give both in y_disps    
def plot_cluster_fits_max(x, y_disps, in_bridge, daFile, loud=0):

    fig, ax = plt.subplots()
    #### Maximum
    ax.plot(x[:-1], y_disps[:-1,0], label="|dy| maximum", c="b", ls="-")

    # large timescale fit
    x_green = in_bridge * np.arange(0, len(in_bridge))
    linear_model=np.polyfit(x_green[x_green!=0], (y_disps[x_green])[x_green!=0,0], 1)
    linear_model_fn=np.poly1d(linear_model)
    if (loud): 
        print(f"Maximum Large Timescale slope: {linear_model[0]/2} um/min") #bc t axis is in 2min steps
    ax.plot(x,linear_model_fn(x), c="black", label="trend of displacement", ls="--")
    
    #short timescale fits
    clustrs = vector_clustering(in_bridge)
    n_max = clustrs.max()
    for i in np.arange(int(n_max))+1:
        linear_model=np.polyfit(x_green[clustrs==i], (y_disps[x_green])[clustrs==i, 0], 1)
        linear_model_fn=np.poly1d(linear_model)
        if (loud): 
            print(f"Maximum Short Timescale slope: {linear_model[0]/2} um/min")
        ax.plot((x[0:len(clustrs)])[clustrs==i],linear_model_fn((x[0:len(clustrs)])[clustrs==i]), c="black",ls="--")


    ax.legend()
    ax.set_ylabel("Displacement [µm]")


    collection = collections.BrokenBarHCollection.span_where(
        x, ymin=0, ymax=(ax.get_ylim()[1]), where=in_bridge, facecolor='green', alpha=0.2)
    ax.add_collection(collection)

    collection = collections.BrokenBarHCollection.span_where(
        x, ymin=0, ymax=(ax.get_ylim()[1]), where=in_bridge==False, facecolor='red', alpha=0.4)
    ax.add_collection(collection)
    
    ax.set_xticks(ax.get_xticks()[1:-2]) #this could be deleted but it gets rid of a warning
    ax.set_xticklabels(np.array(np.array(ax.get_xticks())*2).astype(int)) # to get away the 2min increment
    ax.set_xlabel("Time [min]")
    ax.set_title(str(daFile).split("/")[-1].split(".")[0]) 
    fig.savefig(daFile + "_E_of_t_whole_movie_aut_max.png", dpi=300)
    fig.show()
    
    
def plot_cluster_fits_mean(x, y_disps, in_bridge, daFile, loud=0):

    fig, ax = plt.subplots()

    ##### Mean
    x_green = in_bridge * np.arange(0, len(in_bridge))
    ax.plot(y_disps[:-1,1], label="|dy| mean", c="b", ls="-")
    
    linear_model=np.polyfit(x_green[x_green!=0], (y_disps[x_green])[x_green!=0,1], 1)
    linear_model_fn=np.poly1d(linear_model)
    if (loud): 
            print(f"Mean Long Timescale slope: {linear_model[0]/2} um/min")
    ax.plot(x,linear_model_fn(x), c="black", label="trend of displacement", ls="--")
    
    #short timescale fits
    clustrs = vector_clustering(in_bridge)
    n_max = clustrs.max()
    for i in np.arange(int(n_max))+1:
        linear_model=np.polyfit(x_green[clustrs==i], (y_disps[x_green])[clustrs==i, 1], 1)
        linear_model_fn=np.poly1d(linear_model)
        if (loud): 
            print(f"Mean Short Timescale slope: {linear_model[0]/2} um/min")
        ax.plot((x[0:len(clustrs)])[clustrs==i],linear_model_fn((x[0:len(clustrs)])[clustrs==i]), c="black",ls="--")
    

    #ax.plot(x_disps[:,0], label="|dx| maximum")
    #ax.plot(x_disps[:,1], ls="--", label="|dx| mean")
    ax.legend()
    ax.set_ylabel("Displacement [µm]")


    collection = collections.BrokenBarHCollection.span_where(
        x, ymin=0, ymax=(ax.get_ylim()[1]), where=in_bridge, facecolor='green', alpha=0.2)
    ax.add_collection(collection)

    collection = collections.BrokenBarHCollection.span_where(
        x, ymin=0, ymax=(ax.get_ylim()[1]), where=in_bridge==False, facecolor='red', alpha=0.4)
    ax.add_collection(collection)
    
    ax.set_xticks(ax.get_xticks()[1:-2]) #this could be deleted but it gets rid of a warning
    ax.set_xticklabels(np.array(np.array(ax.get_xticks())*2).astype(int)) # to get away the 2min increment
    ax.set_xlabel("Time [min]")
    ax.set_title(str(daFile).split("/")[-1].split(".")[0]) 
    fig.savefig(daFile + "_E_of_t_whole_movie_aut_mean.png", dpi=300)
    fig.show()