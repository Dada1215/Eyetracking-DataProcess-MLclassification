"""Module GazeExtractViz: 
    to extract and visualize the gaze movements - fixation and saccades"""
# import basic modules 
import pandas as pd
import numpy as np
import os 
import random


## Fixations N0.1 : extract fixation by file
def extract_fixations_byfile(maxD,minT,datapath,datafile) : 
    """Function to extract the all the fixations for one file in a given folder path
    return : dataframe
    """
    ## introduction comme dataframe dans l'ordre de timestamp
    rawdata = pd.read_csv(datapath +'/'+ datafile, sep=',').sort_values(by='timestamp')
    ## Filte Only the points in the boundary of the screen
    insideScreen = (0<=rawdata['x']) & (rawdata['x']<=1920) & (0<=rawdata['y']) & (rawdata['y']<=1080)
    rawdata = rawdata[insideScreen].sort_values('timestamp').reset_index(drop=True) # reindex in order of time

    ## input de départ
    time = rawdata.timestamp
    x = rawdata.x
    y = rawdata.y

    ## étape-1 : iteration pour détécter tous les segment dont tous les deux point enchaînés ont une distance inférieur à 'maxdist'
    segment = [] ## contain 4 elements : (index,startT, endT , numberofpoints)
    # _inition_
    time_start = time[0] # First supposed point
    Nbpt = 1           # counter it as the first
    for i in range(1,len(time)) :
        dist = ((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)**0.5
        if dist <=  maxD : # si il satisfait le critère
            time_end = time[i] # 
            Nbpt = Nbpt + 1 # counte the points 
        else:
            # sinon terminer 
            time_end = time[i-1] # fermer le segment and rejeter le point actuel
            segment.append([i, time_start, time_end, Nbpt])
            # _réinition_
            time_start = time[i]
            Nbpt = 1

    ## étape-2 : filtrer les segment en fonction de la durée
    # output prévu
    data = [] # matrix
    for seg in segment :
        index = seg[0]
        duration =seg[2] - seg[1]
        Numberpoint = seg[3]
        if duration >= minT :
            x_centre = x[index]
            y_centre = y[index]
            data.append([seg[1], seg[2], duration, Numberpoint, x_centre, y_centre])
    
    ## mis en dataframe avec les nomes de colonnes 
    df = pd.DataFrame(data=data,columns=['time_start','time_end','fixaTime','Nbpt','x_centre','y_centre'])

    ## add 3 colomns at the end for user experience infos and then reorder them
    lst_keycols = rawdata.columns[:-3].to_list()
    for col in lst_keycols :
        # for each col, creat a new colomns in the df with correspondant value in rawdata
        df[col] = rawdata[col].unique()[0]
    # Specify the desired order
    colomns_neworder = lst_keycols + list(df.columns[:-3])  
    # Reorder columns
    df = df[colomns_neworder]

    ## output dataframe
    return df


## Functions N0.2 : for saccades extraction and addition to fixation dataframe
import numpy as np
def extractSaccads_byfile_andAddto_fixData(df_fix) :
    """Extraction of saccades feutures based on the extracted fixation data by file
    df_fix : extracted fixation dataframe by file """
    ## query of data for the input participant and activity key
    df_fix = df_fix.sort_values('time_start').reset_index(drop=True) # make sure: the index is in order of time

    ## Variables and 
    # creation of new columns for features values to extract
    df_fix[['saccTime','Amlplitude','vectorX','vectorY','angleAbs180','angleAbs360','angleRela180']] = None
    
    # extraire des colonnes spécifiques avant la boucle pour réduire les appels à 'df.loc'
    startTimes = df_fix['time_start']
    endTimes = df_fix['time_end']
    centresX = df_fix['x_centre']
    centresY = df_fix['y_centre']
    
    ## Extraction in boucle line by line
    pre_angle  = 0  # for a reference to calculate the first relative angle
    for idx in df_fix.index :
        # for the current line of index number 'idx'
        current_startTime = startTimes[idx]
        # find out next timestamp, which is the follozing value [time_start] in the sequence of timestamp
        next_startTime = find_next_timestamp(startTimes,current_startTime)
        if next_startTime != None : 
            ## if it is findable, which means this is not the last point 
            #1 find out the line where the value [time_start] is the next timestamp
            nextIdx = (df_fix[df_fix['time_start']==next_startTime].index)[0]
            #2 calculate the vector from the current pont to the next point
            X_current, Y_current = centresX[idx], centresY[idx]
            X_next, Y_next = centresX[nextIdx], centresY[nextIdx]
            vectorX,vectorY = (X_next - X_current), (Y_next - Y_current)
            #3 with the vector,caculate the amplitude et vilocity 
            Amlplitude = np.sqrt( vectorX**2 + vectorY**2 )
            saccTime = next_startTime - endTimes[idx]
            # saccVelocity = Amlplitude / saccTime
            #4 angle absolute 180/360
            cos_Abs = np.dot([vectorX,vectorY],[1,0]) / Amlplitude    
            angleAbs180 = np.rad2deg(np.arccos(cos_Abs)) 
            if vectorY < 0:
                angleAbs360 = 360 - angleAbs180
            else:
                angleAbs360 = angleAbs180
            #4 angle relative 180
            angleRela180 = abs(angleAbs360 - pre_angle) #% 360
            if angleRela180 > 180:
                angleRela180 = 360 - angleRela180
            # reinitiation 
            pre_angle = angleAbs360

            ## creation culumns and affectuation of values in the created culumns
            df_fix.loc[idx,'saccTime'] = saccTime
            df_fix.loc[idx,'Amlplitude'] = Amlplitude
            df_fix.loc[idx,'vectorX'] = vectorX 
            df_fix.loc[idx,'vectorY'] = vectorY
            df_fix.loc[idx,'angleAbs180'] = angleAbs180
            df_fix.loc[idx,'angleAbs360'] = angleAbs360
            df_fix.loc[idx,'angleRela180'] = angleRela180
    ## return of function: data frame
    return df_fix

# function for function
def find_next_timestamp(liste, value):
    """
    Automaticly find the next timestamp in a siquence of timestamp values
    """
    # Filtrer les valeurs strictement supérieures à la valeur donnée
    valeurs_superieures = [x for x in liste if x > value]
    # Si aucune valeur ne correspond, retourner None
    if not valeurs_superieures:
        return None
    # Retourner la plus petite valeur parmi les valeurs supérieures
    return min(valeurs_superieures)


## 
## Fixations N0.3 : extraction in total
def Extract_FixaSacca_ofAllfiles(maxD,minT,path) :
    """Function to output a dataframe that contains all the gaze data"""
    # generate the list of filenames from the folder path
    import os 
    file_list = os.listdir(path)
    # input of the for circle
    df_combined = None
    for file in file_list :
        # fixation extraction in the file
        fixation_file = extract_fixations_byfile(maxD, minT, path, datafile= file)
        # saccades extraction in the file and concatenation of colomns
        addSacca_file = extractSaccads_byfile_andAddto_fixData(fixation_file)
        # Concatenate the combined df to the previous combined dataframe
        df_combined = pd.concat([df_combined,addSacca_file])

    return df_combined.dropna() # drop lignes with Nan values

## end of function extractation



"""Visualisation of Fixation and Saccades 
"""
import matplotlib.pyplot as plt
import seaborn as sns

## Viz01 : fixaSaccadViz_standalone
def fixaSaccadViz_standalone(participant,activity,data):
    """visualisation of the fixations of certain activity by certain participant"""
    import matplotlib.pyplot as plt
    # data input 
    df = data[ (data['participant']==participant) & 
                   (data['activity']==activity) ].sort_values('time_start').reset_index(drop=True)
    longscreen, widthscreen = 1920,1080
    size = df['fixaTime'] / df['fixaTime'].median() *(widthscreen/40)
    colors = df['fixaTime'] 

    fig,ax = plt.subplots(figsize =(7,4))

    ax.scatter('x_centre', 'y_centre', s=size, c=colors, data=df, alpha=0.9)

    for i in range(len(df.index)-1) :
        if(df.index[i+1] != 0) : 
            ax.arrow(df.x_centre[i], df.y_centre[i], 
                     df.x_centre[i+1]-df.x_centre[i], df.y_centre[i+1]-df.y_centre[i], 
                     width=0.02,color='darkgreen',head_length=10.0,head_width=10.0,alpha=0.2)

    ax.set(xlim=[0, longscreen], ylim=[0,widthscreen],xlabel='x_centre', ylabel='y_centre',
       title='{} / {}' .format(participant,activity))
    ax.grid(True)
    plt.show()


## Viz01 : fixaSaccadViz_standalone
def VizCompare_1activity_btwn3person(activity,data): 
    """Compared visualisation of one given activity between 3 randam participants
    
    Used to observe the similar patterns of one activity """
    import random
    import matplotlib.pyplot as plt
    sujets = list(data.participant.unique())
    sujets_rd = random.sample(sujets,3)

    fig, axs = plt.subplots(1, 3, subplot_kw=dict(box_aspect=1),
                             sharex=True, sharey=True,layout='constrained',figsize=(19,6))

    longscreen, widthscreen = 1920,1080
    for suj, ax in enumerate(axs.flat):
        df = data[ (data['participant']== sujets_rd[suj]) 
                     & (data['activity']==activity) ].sort_values('time_start').reset_index(drop=True)

        size = df['fixaTime'] / df['fixaTime'].median() *(widthscreen/40)
        colors = df['fixaTime']  / df['fixaTime'].median() 
        ax.scatter('x_centre', 'y_centre', s=size, c =colors, data=df, alpha=0.9)

        for i in range(len(df.index)-1) :
            if(df.index[i+1] != 0) : 
                ax.arrow(df.x_centre[i], df.y_centre[i], 
                         df.x_centre[i+1]-df.x_centre[i], df.y_centre[i+1]-df.y_centre[i], 
                         width=0.02,color='darkgreen',head_length=20.0,head_width=20.0,alpha=0.2)

        ax.set(xlim=[0, longscreen], ylim=[0,widthscreen],title='{}'.format(sujets_rd[suj]))
        ax.margins(x=0,y=0)
        ax.grid(True)
        fig.tight_layout()
    print("presse the button again to refresh and choose anothers 3 participant")
    plt.show()


## Viz 03: 
def Viz_FixaSacc_compare8actis(gazedata): 
    """Comparing 8 different activities by one random participant, help to observe the different patterns between activities
    
    return : 8 Scatter Plot of fixation and saccade """
    import random
    import matplotlib.pyplot as plt
    #sujet randomly choosed
    sujets = list(gazedata.participant.unique())
    sujet_radom = random.choice(sujets)
    # list af all 8 activity
    activities = list(gazedata.activity.unique())
    # create plot and subplot
    fig, axs = plt.subplots(2, 4, subplot_kw=dict(box_aspect=1),
                             sharex=True, sharey=True,layout='constrained',figsize=(20,6))

    ## Boucle for each of the 8 activities
    longscreen, widthscreen = 1920,1080
    for j, ax in enumerate(axs.flat):
        # Query of dataframe
        df = gazedata[ (gazedata['participant']== sujet_radom) & (gazedata['activity']==activities[j])]
        # points of Fixation
        size = df['fixaTime'] / df['fixaTime'].median() *(widthscreen/80)
        colors = df['fixaTime'] 
        ax.scatter('x_centre', 'y_centre', s=size, c =colors, data=df, alpha=0.9)
        # vectors of Saccades
        for i in range(len(df.index)-1) :
            if(df.index[i+1] != 0) : 
                ax.arrow(df.x_centre[i], df.y_centre[i], 
                         df.x_centre[i+1]-df.x_centre[i], df.y_centre[i+1]-df.y_centre[i], 
                         width=0.02,color='red',head_length=20.0,head_width=20.0,alpha=0.1)
        # ax configurations
        ax.set(xlim=[0, 1900], ylim=[0,1400],title='{}'.format(activities[j]))
        ax.margins(x=0,y=0)
        ax.grid(True)
        fig.tight_layout()

    plt.axis([0, longscreen, 0, widthscreen])
    plt.show()