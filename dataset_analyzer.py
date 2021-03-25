import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from enum import IntEnum

FPS = 35

class GameStage(IntEnum):
    EXPLORATION = 0
    COMBAT      = 1
    MENU        = 2
    CONSOLE     = 3
    SCOREBOARD  = 4

class DatasetAnalyzer():
    def __init__(self, pathCSV, smooth=None, windowLength=None):
        self.pathCSV      = pathCSV
        self.pathOutput   = '/'.join(pathCSV.split('/')[:-1])
        self.windowLength = windowLength

        self.smooth       = smooth
        if not smooth:
            self.fileName = pathCSV.split('.')[1].split('.')[0].split('/')[-1]
        else:
            self.fileName = "%s_smooth_w%i" % (pathCSV.split('.')[1].split('.')[0].split('/')[-1], self.windowLength)
            self.exp_moving_average()

        self.videoData = {}
        self.kbpsOverall = 0

    def read_CSV(self):

        with open("%s/%s.csv" % (self.pathOutput, self.fileName), 'r') as csvInput:
            csvReader = csv.reader(csvInput)
            #skip the first row 
            next(csvReader)

            # Fetch the first row seperately for stage comparison
            firstRow = next(csvReader)
            stage    = firstRow[3]
            secStageFirst = float(firstRow[0])

            # To keep the last row before new game stage
            tmpRow = deepcopy(firstRow)
            
            for row in csvReader:
                
                if row[3] != stage:
                    sec   = float(row[0])
                    stage = int(tmpRow[3])
                    self.videoData.update({round(sec, 6): stage})

                    # Update
                    secStageFirst = float(row[0])
                    stage = row[3]
                    
                tmpRow = deepcopy(row)
            
            # For the last game stage seconds. They are not saved with the code block above because they won't see different stage afterward.
            sec   = float(tmpRow[0])
            stage = int(tmpRow[3])
            self.videoData.update({round(sec, 6): stage})
        
        self.get_statistics()
        self.plot()

    def exp_moving_average(self):

        with open("%s/%s.csv" % (self.pathOutput, self.fileName), 'w') as csvOutput:

            csvWriter = csv.writer(csvOutput, lineterminator='\n')

            dfInput = pd.read_csv(self.pathCSV)
            newRows = []
            newRows.append(dfInput.columns)

            dfInput['EMA'] = dfInput['size'].ewm(span=self.windowLength, adjust=False).mean()
            
            for _, row in dfInput.iterrows():
                newRows.append([str(round(row['time'], 6)), str(round(row['EMA'])), row['pict_type'], row['gamestage']])

            csvWriter.writerows(newRows)

        """ Calculation without using Pandas method (faster approach) """
        # with open(self.pathCSV, 'r') as csvInput, \
        #         open("%s/%s.csv" % (self.pathOutput, self.fileName), 'w') as csvOutput:

        #     csvWriter = csv.writer(csvOutput, lineterminator='\n')
        #     csvReader = csv.reader(csvInput)
        #     newRows = []

        #     #skip the first row 
        #     header = next(csvReader)
        #     newRows.append(header)

        #     # Smoothing factor
        #     alpha = 2 / (self.windowLength + 1)  

        #     # Fetch the first row seperately
        #     firstRow = next(csvReader)
        #     sec   = firstRow[0]
        #     type  = firstRow[2]
        #     stage = firstRow[3]
        #     formerEMA = int(float(firstRow[1]))
        #     newRows.append([sec, str(formerEMA), type, stage])

        #     for row in csvReader: 
        #         formerEMA = alpha * (float(row[1]) - formerEMA) + formerEMA
        #         newRows.append([row[0], str(round(formerEMA)), row[2], row[3]])    

        #     csvWriter.writerows(newRows)

    def get_statistics(self):
        
        dfInput = pd.read_csv("%s/%s.csv" % (self.pathOutput, self.fileName))

        self.kbpsOverall = round((dfInput['size'].sum() / sorted(list(self.videoData.keys()))[-1]) * 8 // 1000, 2)          

        count_total = dfInput.shape[0]

        df_exploration = dfInput[(dfInput['gamestage'] == GameStage.EXPLORATION)]
        df_combat      = dfInput[(dfInput['gamestage'] == GameStage.COMBAT)]
        df_menu        = dfInput[(dfInput['gamestage'] == GameStage.MENU)]
        df_console     = dfInput[(dfInput['gamestage'] == GameStage.CONSOLE)]
        df_scoreboard  = dfInput[(dfInput['gamestage'] == GameStage.SCOREBOARD)]

        min_exploration   = round(df_exploration['size'].min() * 8 / 1000, 5)
        max_exploration   = df_exploration['size'].max() * 8 // 1000
        cv_exploration    = round((df_exploration['size'].std() / df_exploration['size'].mean()), 2)
        mean_exploration  = round(df_exploration['size'].mean() * 8 / 1000, 2)
        count_exploration = df_exploration.shape[0]
        dom_exploration   = round(count_exploration * 100 / count_total , 2)

        min_combat   = round(df_combat['size'].min() * 8 / 1000, 5)
        max_combat   = df_combat['size'].max() * 8 // 1000
        cv_combat    = round((df_combat['size'].std() / df_combat['size'].mean()), 2)
        mean_combat  = round(df_combat['size'].mean() * 8 / 1000, 2)
        count_combat = df_combat.shape[0]
        dom_combat   = round(count_combat * 100 / count_total , 2)

        min_menu     = round(df_menu['size'].min() * 8 / 1000, 5)
        max_menu     = df_menu['size'].max() * 8 // 1000
        cv_menu      = round((df_menu['size'].std() / df_menu['size'].mean()), 2)
        mean_menu    = round(df_menu['size'].mean() * 8 / 1000, 2)
        count_menu   = df_menu.shape[0]
        dom_menu     = round(count_menu * 100 / count_total , 2)

        min_console   = round(df_console['size'].min() * 8 / 1000, 5)
        max_console   = df_console['size'].max() * 8 // 1000
        cv_console    = round((df_console['size'].std() / df_console['size'].mean()), 2)
        mean_console  = round(df_console['size'].mean() * 8 / 1000, 2)
        count_console = df_console.shape[0]
        dom_console   = round(count_console * 100 / count_total , 2)
        
        min_scoreboard   = round(df_scoreboard['size'].min() * 8 / 1000, 5)
        max_scoreboard   = df_scoreboard['size'].max() * 8 // 1000
        cv_scoreboard    = round((df_scoreboard['size'].std() / df_scoreboard['size'].mean()), 2)
        mean_scoreboard  = round(df_scoreboard['size'].mean() * 8 / 1000, 2)
        count_scoreboard = df_scoreboard.shape[0]
        dom_scoreboard   = round(count_scoreboard * 100 / count_total , 2)

        data = {"Gamestage":        ["Exploration", "Combat", "Menu", "Console", "Scoreboard"],
                "Min (kilobit)":    [min_exploration, min_combat, min_menu, min_console, min_scoreboard],
                "Max (kilobit)":    [max_exploration, max_combat, max_menu, max_console, max_scoreboard],
                "Mean (kilobit)":   [mean_exploration,mean_combat, mean_menu, mean_console, mean_scoreboard],
                "Coeff. Variation": [cv_exploration, cv_combat, cv_menu, cv_console, cv_scoreboard],
                "Fraction (%)":     [dom_exploration, dom_combat, dom_menu, dom_console, dom_scoreboard]}

        dfOutput = pd.DataFrame(data) 

        fig,ax = self.render_table(dfOutput)
        fig.savefig("%s/%s_stats.png" % (self.pathOutput, self.fileName))
        plt.clf()

    def render_table(self, data, col_width=3.5, row_height=0.6, font_size=12, header_color='#40466e', edge_color='w',
                           row_colors=['#f1f1f2', 'w'], bbox=[0, 0, 1, 1], header_columns=0, ax=None, **kwargs):
        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize=size)
            ax.axis('off')
        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)

        for k, cell in mpl_table._cells.items():
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0]%len(row_colors) ])

        return ax.get_figure(), ax
    
    def plot(self):

        sns.set(rc={'figure.figsize': (18, 6)})
        sns.set_theme()

        legendMap = {GameStage.EXPLORATION: ["blue", "Exploration"],
                     GameStage.COMBAT:      ["red", "Combat"],
                     GameStage.MENU:        ["gold", "Menu"],
                     GameStage.CONSOLE:     ["lime", "Console"],
                     GameStage.SCOREBOARD:  ["cyan", "Scoreboard"]}

        # In order to avoid using same legend multiple times
        usedLegends = []  
                   
        dfInput = pd.read_csv("%s/%s.csv" % (self.pathOutput, self.fileName))

        dfInput['bitrate_kbps']    = dfInput['size'].rolling(window=FPS).sum() * 8 // 1000
        dfInput['accumulatedBits'] = dfInput['size'].cumsum()
        
        """ Bitrate/Time Graph """
        
        fig = sns.lineplot(data=dfInput, x="time", y="bitrate_kbps", color='black', linewidth=1)

        formerSec = 0
        for sec, stage in self.videoData.items():

            color, label = legendMap[stage]
            if stage in usedLegends:
                label = None
            else:
                usedLegends.append(stage)
            
            plt.axvspan(formerSec, sec, facecolor=color, alpha=0.15, label=label)
            formerSec = sec
        
        if self.smooth:
            plt.title("Bitrate/Time per Game Stage [E.M.A with alpha: {:.3f}] - [Overall kbps: {:.3f}]".format(2 / (self.windowLength+1), self.kbpsOverall))
        else:
            plt.title("Bitrate/Time per Game Stage [Overall kbps: {:.3f}]".format(self.kbpsOverall))

        plt.legend(facecolor='white', framealpha=1, loc="upper right") 
        plt.margins(x=0)
        fig.figure.savefig("%s/%s.png" % (self.pathOutput, self.fileName), bbox_inches = "tight")
        plt.clf()

        """ Average Bitrate for Each Game Stage """

        avgStageBitrates = {0.0: 0}

        for index, row in dfInput.iterrows():

            if index == 0:
                firstRow = deepcopy(row)
                continue

            if row['gamestage'] != firstRow['gamestage']:
                timeDiff = tmpRow['time'] - firstRow['time']
                accBitrateStage = int((tmpRow['accumulatedBits'] - firstRow['accumulatedBits']) // timeDiff) * 8 // 1000
                avgStageBitrates[round(firstRow['time'] + timeDiff/2.0, 3)] = (accBitrateStage, tmpRow['gamestage'])
                #print("Between sec. {} - {} -> Gamestage: {} -> dB/dt: {}".format(round(firstRow['time'],2), round(tmpRow['time'],2), tmpRow['gamestage'], accBitrateStage))
                firstRow = deepcopy(row) 

            tmpRow = deepcopy(row) 
                
        # for the last game stage
        if tmpRow['time'] != firstRow['time']:
            timeDiff = tmpRow['time'] - firstRow['time']
            accBitrateStage = int((tmpRow['accumulatedBits'] - firstRow['accumulatedBits']) // timeDiff) * 8 // 1000
            avgStageBitrates[round(firstRow['time'] + timeDiff/2.0, 3)] = (accBitrateStage, tmpRow['gamestage'])
            #print("Between sec. {} - {} -> Gamestage: {} -> dB/dt: {}".format(round(firstRow['time'],2), round(tmpRow['time'],2), tmpRow['gamestage'], accBitrateStage))

        del avgStageBitrates[0.0]

        fig = plt.figure()
        ax  = fig.add_subplot(111, facecolor='white')
        
        usedLegends.clear()
        for sec, (kbps, stage) in avgStageBitrates.items():

            color, label = legendMap[stage]
            if stage in usedLegends:
                label = None
            else:
                usedLegends.append(stage)
            
            ax.scatter(x=sec, y=kbps, c=color, label=label, s=8) 
        
        if self.smooth:
            plt.title("Average Bitrate per Game Stage [E.M.A with alpha: {:.3f}] - [Overall kbps: {:.3f}]".format(2 / (self.windowLength+1), self.kbpsOverall))
        else:
            plt.title("Average Bitrate per Game Stage [Overall kbps: {:.3f}]".format(self.kbpsOverall))

        ax.grid(b=True, which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.1)
        plt.margins(x=0.01, y=0.01)
        plt.legend()
        #plt.xticks(list(avgStageBitrates.keys()), rotation=90, fontsize=6)
        plt.xlabel("sec")
        plt.ylabel("kbits/sec")
        fig.canvas.draw()
        plt.savefig("%s/%s_avgBperStage.png" % (self.pathOutput, self.fileName), bbox_inches = "tight")
        plt.clf()

        """ Variation of Average Bitrate: B(n) - B(n-1) """

        fig = plt.figure()
        ax  = fig.add_subplot(111, facecolor='white')

        usedLegends.clear()
        formerKbps = 0
        for sec, (kbps, stage) in avgStageBitrates.items():
            
            if not formerKbps:
                formerKbps = list(avgStageBitrates.values())[0][0]
                continue

            color, label = legendMap[stage]
            if stage in usedLegends:
                label = None
            else:
                usedLegends.append(stage)
            
            ax.scatter(x=sec, y=kbps-formerKbps, c=color, label=label, s=9)   
            formerKbps = kbps
        
        if self.smooth:
            plt.title("Variation of Average Bitrate: B(n) - B(n-1) [E.M.A with alpha: {:.3f}] - [Overall kbps: {:.3f}]".format(2 / (self.windowLength+1), self.kbpsOverall))
        else:
            plt.title("Variation of Average Bitrate: B(n) - B(n-1) [Overall kbps: {:.3f}]".format(self.kbpsOverall))

        ax.grid(b=True, which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.1)
        plt.margins(x=0.01, y=0.01)
        plt.legend()
        #plt.xticks(list(avgStageBitrates.keys()), rotation=90, fontsize=6)
        plt.xlabel("sec")
        plt.ylabel("kbits/sec")
        fig.canvas.draw()
        plt.savefig("%s/%s_varB.png" % (self.pathOutput, self.fileName), bbox_inches = "tight")
        plt.clf()

        """ Bitrate Change During 2 Seconds around Stage Changes """

        bitrateVariances = {}
        tmpRow = deepcopy(dfInput.iloc[0])
        totalRows = dfInput.shape[0]

        for index, row in dfInput.iterrows():

            # Game stage change
            if row['gamestage'] != tmpRow['gamestage']:

                # Lowerbound and upperbound must be at the same distance from the index of change
                counterBackward = counterForward = 0

                while counterBackward < FPS and \
                        index-(counterBackward+1) > 0 and \
                            dfInput.iloc[index-(counterBackward+1)]['gamestage'] == tmpRow['gamestage']:
                    counterBackward += 1

                while counterForward < FPS and \
                        index+(counterForward+1) < totalRows and \
                            dfInput.iloc[index+(counterForward+1)]['gamestage'] == row['gamestage']:
                    counterForward += 1

                bound = counterBackward if counterBackward < counterForward else counterForward
                bitrate_kbps = int((dfInput.iloc[index+bound]['accumulatedBits'] - dfInput.iloc[index-bound]['accumulatedBits']) /
                                  (dfInput.iloc[index+bound]['time'] - dfInput.iloc[index-bound]['time'])) * 8 // 1000

                bitrateVariances[round(row['time'], 3)] = (bitrate_kbps, row['gamestage'])
            
            tmpRow = deepcopy(row)

        usedLegends.clear()

        fig = plt.figure()
        ax  = fig.add_subplot(111, facecolor='white')

        for sec, (kbps, stage) in bitrateVariances.items():

            color, label = legendMap[stage]
            if stage in usedLegends:
                label = None
            else:
                usedLegends.append(stage)
            
            ax.scatter(x=sec, y=kbps, c=color, label=label, s=8) 
        
        if self.smooth:
            plt.title("Bitrate Change During 2 Seconds around Stage Changes [E.M.A with alpha: {:.3f}] - [Overall kbps: {:.3f}]".format(2 / (self.windowLength+1), self.kbpsOverall))
        else:
            plt.title("Bitrate Change During 2 Seconds around Stage Changes [Overall kbps: {:.3f}]".format(self.kbpsOverall))

        ax.grid(b=True, which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.1)
        plt.margins(x=0.01, y=0.01)
        plt.legend()
        #plt.xticks(list(avgStageBitrates.keys()), rotation=90, fontsize=6)
        plt.xlabel("sec")
        plt.ylabel("kbits/sec")
        fig.canvas.draw()
        plt.savefig("%s/%s_2secChangeB.png" % (self.pathOutput, self.fileName), bbox_inches = "tight")
        plt.clf()

        """ Accumulated kbit Change on Exploration <-> Combat """

        def clearCollections(*collections):
            for collection in collections:  collection.clear()

        kbitChangeOn_exp_to_combat = {}
        kbitChangeOn_combat_to_exp = {}
        # Skip the first 34 frames, because there is no bitrate value for them.
        tmpRow = deepcopy(dfInput.iloc[34])

        for index, row in dfInput.iterrows():

            if index < 35:  continue

            # Game stage change
            if row['gamestage'] != tmpRow['gamestage']:
                
                # Change from Exploration to Combat
                if tmpRow['gamestage'] == GameStage.EXPLORATION and row['gamestage'] == GameStage.COMBAT:

                    difference = row['accumulatedBits'] - tmpRow['accumulatedBits']

                    if int(tmpRow['bitrate_kbps']) not in kbitChangeOn_exp_to_combat:    
                        kbitChangeOn_exp_to_combat[int(tmpRow['bitrate_kbps'])] = difference
                    else:   
                        kbitChangeOn_exp_to_combat[int(tmpRow['bitrate_kbps'])] = (kbitChangeOn_exp_to_combat[tmpRow['bitrate_kbps']] + difference) // 2

                # Change from Combat to Exploration
                elif tmpRow['gamestage'] == GameStage.COMBAT and row['gamestage'] == GameStage.EXPLORATION:

                    difference = row['accumulatedBits'] - tmpRow['accumulatedBits']

                    if int(tmpRow['bitrate_kbps']) not in kbitChangeOn_combat_to_exp:    
                        kbitChangeOn_combat_to_exp[int(tmpRow['bitrate_kbps'])] = difference
                    else:   
                        kbitChangeOn_combat_to_exp[int(tmpRow['bitrate_kbps'])] = (kbitChangeOn_combat_to_exp[int(tmpRow['bitrate_kbps'])] + difference) // 2
            
            tmpRow = deepcopy(row)

        bandwidth_0_100   = []
        bandwidth_100_200 = []
        bandwidth_200_300 = []
        bandwidth_300_400 = []
        bandwidth_400_500 = []
        bandwidth_500_inf = []

        for bitrate, difference in kbitChangeOn_exp_to_combat.items():
            if   bitrate < 100:                     bandwidth_0_100.append(difference)
            elif bitrate >= 100 and bitrate < 200:  bandwidth_100_200.append(difference)
            elif bitrate >= 200 and bitrate < 300:  bandwidth_200_300.append(difference)
            elif bitrate >= 300 and bitrate < 400:  bandwidth_300_400.append(difference)
            elif bitrate >= 400 and bitrate < 500:  bandwidth_400_500.append(difference)
            else:                                   bandwidth_500_inf.append(difference)
        
        def mean(nums): 
            if nums:    return int(sum(nums)/len(nums))
            else:       return '-'

        data = {"Exploration->Combat":   ["BW 0-100 kbps", "BW 100-200 kbps", "BW 200-300 kbps", "BW 300-400 kbps", "BW 400-500 kbps", "BW > 500 kbps"],
                "Number of Change":      [len(bandwidth_0_100), len(bandwidth_100_200), len(bandwidth_200_300), len(bandwidth_300_400), len(bandwidth_400_500), len(bandwidth_500_inf)],
                "Min Variation (kbits)": [min(bandwidth_0_100, default='-'), min(bandwidth_100_200, default='-'), min(bandwidth_200_300, default='-'), 
                                          min(bandwidth_300_400, default='-'), min(bandwidth_400_500, default='-'), min(bandwidth_500_inf, default='-')],
                "Max Variation (kbits)": [max(bandwidth_0_100, default='-'), max(bandwidth_100_200, default='-'), max(bandwidth_200_300, default='-'),
                                          max(bandwidth_300_400, default='-'), max(bandwidth_400_500, default='-'), max(bandwidth_500_inf, default='-')],
                "Mean Variation (kbits)":[mean(bandwidth_0_100), mean(bandwidth_100_200), mean(bandwidth_200_300), mean(bandwidth_300_400), mean(bandwidth_400_500), mean(bandwidth_500_inf)]
               }
        
        dfOutput = pd.DataFrame(data) 
        fig,ax = self.render_table(dfOutput)
        fig.savefig("%s/%s_stats_expToCombat_varKbits.png" % (self.pathOutput, self.fileName))
        plt.clf()

        clearCollections(bandwidth_0_100, bandwidth_100_200, bandwidth_200_300, bandwidth_300_400, bandwidth_400_500, bandwidth_500_inf)

        for bitrate, difference in kbitChangeOn_combat_to_exp.items():
            if   bitrate < 100:                     bandwidth_0_100.append(difference)
            elif bitrate >= 100 and bitrate < 200:  bandwidth_100_200.append(difference)
            elif bitrate >= 200 and bitrate < 300:  bandwidth_200_300.append(difference)
            elif bitrate >= 300 and bitrate < 400:  bandwidth_300_400.append(difference)
            elif bitrate >= 400 and bitrate < 500:  bandwidth_400_500.append(difference)
            else:                                   bandwidth_500_inf.append(difference)

        data = {"Combat->Exploration":   ["BW 0-100 kbps", "BW 100-200 kbps", "BW 200-300 kbps", "BW 300-400 kbps", "BW 400-500 kbps", "BW > 500 kbps"],
                "Number of Change":      [len(bandwidth_0_100), len(bandwidth_100_200), len(bandwidth_200_300), len(bandwidth_300_400), len(bandwidth_400_500), len(bandwidth_500_inf)],
                "Min Variation (kbits)": [min(bandwidth_0_100, default='-'), min(bandwidth_100_200, default='-'), min(bandwidth_200_300, default='-'), 
                                          min(bandwidth_300_400, default='-'), min(bandwidth_400_500, default='-'), min(bandwidth_500_inf, default='-')],
                "Max Variation (kbits)": [max(bandwidth_0_100, default='-'), max(bandwidth_100_200, default='-'), max(bandwidth_200_300, default='-'),
                                          max(bandwidth_300_400, default='-'), max(bandwidth_400_500, default='-'), max(bandwidth_500_inf, default='-')],
                "Mean Variation (kbits)":[mean(bandwidth_0_100), mean(bandwidth_100_200), mean(bandwidth_200_300), mean(bandwidth_300_400), mean(bandwidth_400_500), mean(bandwidth_500_inf)]
               }
        
        dfOutput = pd.DataFrame(data) 
        fig,ax = self.render_table(dfOutput)
        fig.savefig("%s/%s_stats_combatToExp_varKbits.png" % (self.pathOutput, self.fileName))
        plt.clf()

        """ Bitrate Change on Exploration <-> Combat """

        def mean(nums): 
            if nums:                return round(sum(nums) / len(nums), 2)
            else:                   return '-'
        def ratio_zeros(nums):
            if nums:                return round(nums.count(0) * 100 / len(nums), 2)
            else:                   return '-'
        def ratio_negatives(nums): 
            if nums:                return round(len([num for num in nums if num < 0]) * 100 / len(nums), 2)
            else:                   return '-'

        kbpsChangeOn_exp_to_combat = {}
        kbpsChangeOn_combat_to_exp = {}
        # Skip the first 34 frames, because there is no bitrate value for them.
        tmpRow = deepcopy(dfInput.iloc[34])

        for index, row in dfInput.iterrows():

            if index < 35:  continue

            # Game stage change
            if row['gamestage'] != tmpRow['gamestage']:
                
                # Change from Exploration to Combat
                if tmpRow['gamestage'] == GameStage.EXPLORATION and row['gamestage'] == GameStage.COMBAT:

                    difference = round((row['bitrate_kbps'] - tmpRow['bitrate_kbps']) * 100 / tmpRow['bitrate_kbps'], 2)

                    if int(tmpRow['bitrate_kbps']) not in kbpsChangeOn_exp_to_combat:    
                        kbpsChangeOn_exp_to_combat[int(tmpRow['bitrate_kbps'])] = difference
                    else:   
                        kbpsChangeOn_exp_to_combat[int(tmpRow['bitrate_kbps'])] = round((kbpsChangeOn_exp_to_combat[tmpRow['bitrate_kbps']] + difference) / 2, 2)

                # Change from Combat to Exploration
                elif tmpRow['gamestage'] == GameStage.COMBAT and row['gamestage'] == GameStage.EXPLORATION:

                    difference = round((row['bitrate_kbps'] - tmpRow['bitrate_kbps']) * 100 / tmpRow['bitrate_kbps'], 2)

                    if int(tmpRow['bitrate_kbps']) not in kbpsChangeOn_combat_to_exp:    
                        kbpsChangeOn_combat_to_exp[int(tmpRow['bitrate_kbps'])] = difference
                    else:   
                        kbpsChangeOn_combat_to_exp[int(tmpRow['bitrate_kbps'])] = round((kbpsChangeOn_combat_to_exp[int(tmpRow['bitrate_kbps'])] + difference) // 2, 2)
            
            tmpRow = deepcopy(row)

        clearCollections(bandwidth_0_100, bandwidth_100_200, bandwidth_200_300, bandwidth_300_400, bandwidth_400_500, bandwidth_500_inf)

        for bitrate, difference in kbpsChangeOn_exp_to_combat.items():
            if   bitrate < 100:                     bandwidth_0_100.append(difference)
            elif bitrate >= 100 and bitrate < 200:  bandwidth_100_200.append(difference)
            elif bitrate >= 200 and bitrate < 300:  bandwidth_200_300.append(difference)
            elif bitrate >= 300 and bitrate < 400:  bandwidth_300_400.append(difference)
            elif bitrate >= 400 and bitrate < 500:  bandwidth_400_500.append(difference)
            else:                                   bandwidth_500_inf.append(difference)

        data = {"Exploration->Combat": ["BW 0-100 kbps", "BW 100-200 kbps", "BW 200-300 kbps", "BW 300-400 kbps", "BW 400-500 kbps", "BW > 500 kbps"],
                "Number of Change":       [len(bandwidth_0_100), len(bandwidth_100_200), len(bandwidth_200_300), len(bandwidth_300_400), len(bandwidth_400_500), len(bandwidth_500_inf)],
                "Bitrate Unaffected (%)": [ratio_zeros(bandwidth_0_100), ratio_zeros(bandwidth_100_200), ratio_zeros(bandwidth_200_300),
                                           ratio_zeros(bandwidth_300_400), ratio_zeros(bandwidth_400_500), ratio_zeros(bandwidth_500_inf)],
                "- Variation (%)":     [ratio_negatives(bandwidth_0_100), ratio_negatives(bandwidth_100_200), ratio_negatives(bandwidth_200_300),
                                        ratio_negatives(bandwidth_300_400), ratio_negatives(bandwidth_400_500), ratio_negatives(bandwidth_500_inf)],
                "Min Variation (%)":   [min(bandwidth_0_100, default='-'), min(bandwidth_100_200, default='-'), min(bandwidth_200_300, default='-'), 
                                        min(bandwidth_300_400, default='-'), min(bandwidth_400_500, default='-'), min(bandwidth_500_inf, default='-')],
                "Max Variation (%)":   [max(bandwidth_0_100, default='-'), max(bandwidth_100_200, default='-'), max(bandwidth_200_300, default='-'),
                                        max(bandwidth_300_400, default='-'), max(bandwidth_400_500, default='-'), max(bandwidth_500_inf, default='-')],
                "Mean Variation (%)":  [mean(bandwidth_0_100), mean(bandwidth_100_200), mean(bandwidth_200_300), mean(bandwidth_300_400), mean(bandwidth_400_500), mean(bandwidth_500_inf)]
               }
       
        dfOutput = pd.DataFrame(data) 
        fig,ax = self.render_table(dfOutput)
        fig.savefig("%s/%s_stats_expToCombat_varKbps.png" % (self.pathOutput, self.fileName))
        plt.clf()

        clearCollections(bandwidth_0_100, bandwidth_100_200, bandwidth_200_300, bandwidth_300_400, bandwidth_400_500, bandwidth_500_inf)

        for bitrate, difference in kbpsChangeOn_combat_to_exp.items():
            if   bitrate < 100:                     bandwidth_0_100.append(difference)
            elif bitrate >= 100 and bitrate < 200:  bandwidth_100_200.append(difference)
            elif bitrate >= 200 and bitrate < 300:  bandwidth_200_300.append(difference)
            elif bitrate >= 300 and bitrate < 400:  bandwidth_300_400.append(difference)
            elif bitrate >= 400 and bitrate < 500:  bandwidth_400_500.append(difference)
            else:                                   bandwidth_500_inf.append(difference)

        data = {"Combat->Exploration": ["BW 0-100 kbps", "BW 100-200 kbps", "BW 200-300 kbps", "BW 300-400 kbps", "BW 400-500 kbps", "BW > 500 kbps"],
                "Number of Change":       [len(bandwidth_0_100), len(bandwidth_100_200), len(bandwidth_200_300), len(bandwidth_300_400), len(bandwidth_400_500), len(bandwidth_500_inf)],
                "Bitrate Unaffected (%)": [ratio_zeros(bandwidth_0_100), ratio_zeros(bandwidth_100_200), ratio_zeros(bandwidth_200_300),
                                           ratio_zeros(bandwidth_300_400), ratio_zeros(bandwidth_400_500), ratio_zeros(bandwidth_500_inf)],
                "- Variation (%)":     [ratio_negatives(bandwidth_0_100), ratio_negatives(bandwidth_100_200), ratio_negatives(bandwidth_200_300),
                                        ratio_negatives(bandwidth_300_400), ratio_negatives(bandwidth_400_500), ratio_negatives(bandwidth_500_inf)],
                "Min Variation (%)":   [min(bandwidth_0_100, default='-'), min(bandwidth_100_200, default='-'), min(bandwidth_200_300, default='-'), 
                                        min(bandwidth_300_400, default='-'), min(bandwidth_400_500, default='-'), min(bandwidth_500_inf, default='-')],
                "Max Variation (%)":   [max(bandwidth_0_100, default='-'), max(bandwidth_100_200, default='-'), max(bandwidth_200_300, default='-'),
                                        max(bandwidth_300_400, default='-'), max(bandwidth_400_500, default='-'), max(bandwidth_500_inf, default='-')],
                "Mean Variation (%)":  [mean(bandwidth_0_100), mean(bandwidth_100_200), mean(bandwidth_200_300), mean(bandwidth_300_400), mean(bandwidth_400_500), mean(bandwidth_500_inf)]
               }
        
        dfOutput = pd.DataFrame(data) 
        fig,ax = self.render_table(dfOutput)
        fig.savefig("%s/%s_stats_combatToExp_varKbps.png" % (self.pathOutput, self.fileName))
        plt.clf()

        """ Average Bitrate Change on Exploration <-> Combat """

        avgKbpsChangeOn_exp_to_combat = {}
        avgKbpsChangeOn_combat_to_exp = {}

        tuples_kbps_stage = list(avgStageBitrates.values())

        for i in range(len(tuples_kbps_stage)-1):

            if tuples_kbps_stage[i][1] == GameStage.EXPLORATION and tuples_kbps_stage[i+1][1] == GameStage.COMBAT:

                difference = round((tuples_kbps_stage[i+1][0] - tuples_kbps_stage[i][0]) * 100 / tuples_kbps_stage[i][0], 2)

                if tuples_kbps_stage[i][0] not in avgKbpsChangeOn_exp_to_combat:
                    avgKbpsChangeOn_exp_to_combat[tuples_kbps_stage[i][0]] = difference
                else:
                    avgKbpsChangeOn_exp_to_combat[tuples_kbps_stage[i][0]] = round((avgKbpsChangeOn_exp_to_combat[tuples_kbps_stage[i][0]] + difference) // 2, 2)

            if tuples_kbps_stage[i][1] == GameStage.COMBAT and tuples_kbps_stage[i+1][1] == GameStage.EXPLORATION:

                difference = round((tuples_kbps_stage[i+1][0] - tuples_kbps_stage[i][0]) * 100 / tuples_kbps_stage[i][0], 2)

                if tuples_kbps_stage[i][0] not in avgKbpsChangeOn_combat_to_exp:
                    avgKbpsChangeOn_combat_to_exp[tuples_kbps_stage[i][0]] = difference
                else:
                    avgKbpsChangeOn_combat_to_exp[tuples_kbps_stage[i][0]] = round((avgKbpsChangeOn_combat_to_exp[tuples_kbps_stage[i][0]] + difference) // 2, 2)
            

        clearCollections(bandwidth_0_100, bandwidth_100_200, bandwidth_200_300, bandwidth_300_400, bandwidth_400_500, bandwidth_500_inf)

        for bitrate, difference in avgKbpsChangeOn_exp_to_combat.items():
            if   bitrate < 100:                     bandwidth_0_100.append(difference)
            elif bitrate >= 100 and bitrate < 200:  bandwidth_100_200.append(difference)
            elif bitrate >= 200 and bitrate < 300:  bandwidth_200_300.append(difference)
            elif bitrate >= 300 and bitrate < 400:  bandwidth_300_400.append(difference)
            elif bitrate >= 400 and bitrate < 500:  bandwidth_400_500.append(difference)
            else:                                   bandwidth_500_inf.append(difference)

        data = {"Exploration->Combat": ["BW 0-100 kbps", "BW 100-200 kbps", "BW 200-300 kbps", "BW 300-400 kbps", "BW 400-500 kbps", "BW > 500 kbps"],
                "Number of Change":       [len(bandwidth_0_100), len(bandwidth_100_200), len(bandwidth_200_300), len(bandwidth_300_400), len(bandwidth_400_500), len(bandwidth_500_inf)],
                "Bitrate Unaffected (%)": [ratio_zeros(bandwidth_0_100), ratio_zeros(bandwidth_100_200), ratio_zeros(bandwidth_200_300),
                                           ratio_zeros(bandwidth_300_400), ratio_zeros(bandwidth_400_500), ratio_zeros(bandwidth_500_inf)],
                "- Variation (%)":     [ratio_negatives(bandwidth_0_100), ratio_negatives(bandwidth_100_200), ratio_negatives(bandwidth_200_300),
                                        ratio_negatives(bandwidth_300_400), ratio_negatives(bandwidth_400_500), ratio_negatives(bandwidth_500_inf)],
                "Min Variation (%)":   [min(bandwidth_0_100, default='-'), min(bandwidth_100_200, default='-'), min(bandwidth_200_300, default='-'), 
                                        min(bandwidth_300_400, default='-'), min(bandwidth_400_500, default='-'), min(bandwidth_500_inf, default='-')],
                "Max Variation (%)":   [max(bandwidth_0_100, default='-'), max(bandwidth_100_200, default='-'), max(bandwidth_200_300, default='-'),
                                        max(bandwidth_300_400, default='-'), max(bandwidth_400_500, default='-'), max(bandwidth_500_inf, default='-')],
                "Mean Variation (%)":  [mean(bandwidth_0_100), mean(bandwidth_100_200), mean(bandwidth_200_300), mean(bandwidth_300_400), mean(bandwidth_400_500), mean(bandwidth_500_inf)]
               }
        
        dfOutput = pd.DataFrame(data) 
        fig,ax = self.render_table(dfOutput)
        fig.savefig("%s/%s_stats_expToCombat_varAvgKbps.png" % (self.pathOutput, self.fileName))
        plt.clf()

        clearCollections(bandwidth_0_100, bandwidth_100_200, bandwidth_200_300, bandwidth_300_400, bandwidth_400_500, bandwidth_500_inf)

        for bitrate, difference in avgKbpsChangeOn_combat_to_exp.items():
            if   bitrate < 100:                     bandwidth_0_100.append(difference)
            elif bitrate >= 100 and bitrate < 200:  bandwidth_100_200.append(difference)
            elif bitrate >= 200 and bitrate < 300:  bandwidth_200_300.append(difference)
            elif bitrate >= 300 and bitrate < 400:  bandwidth_300_400.append(difference)
            elif bitrate >= 400 and bitrate < 500:  bandwidth_400_500.append(difference)
            else:                                   bandwidth_500_inf.append(difference)

        data = {"Combat->Exploration": ["BW 0-100 kbps", "BW 100-200 kbps", "BW 200-300 kbps", "BW 300-400 kbps", "BW 400-500 kbps", "BW > 500 kbps"],
                "Number of Change":       [len(bandwidth_0_100), len(bandwidth_100_200), len(bandwidth_200_300), len(bandwidth_300_400), len(bandwidth_400_500), len(bandwidth_500_inf)],
                "Bitrate Unaffected (%)": [ratio_zeros(bandwidth_0_100), ratio_zeros(bandwidth_100_200), ratio_zeros(bandwidth_200_300),
                                           ratio_zeros(bandwidth_300_400), ratio_zeros(bandwidth_400_500), ratio_zeros(bandwidth_500_inf)],
                "- Variation (%)":     [ratio_negatives(bandwidth_0_100), ratio_negatives(bandwidth_100_200), ratio_negatives(bandwidth_200_300),
                                        ratio_negatives(bandwidth_300_400), ratio_negatives(bandwidth_400_500), ratio_negatives(bandwidth_500_inf)],
                "Min Variation (%)":   [min(bandwidth_0_100, default='-'), min(bandwidth_100_200, default='-'), min(bandwidth_200_300, default='-'), 
                                        min(bandwidth_300_400, default='-'), min(bandwidth_400_500, default='-'), min(bandwidth_500_inf, default='-')],
                "Max Variation (%)":   [max(bandwidth_0_100, default='-'), max(bandwidth_100_200, default='-'), max(bandwidth_200_300, default='-'),
                                        max(bandwidth_300_400, default='-'), max(bandwidth_400_500, default='-'), max(bandwidth_500_inf, default='-')],
                "Mean Variation (%)":  [mean(bandwidth_0_100), mean(bandwidth_100_200), mean(bandwidth_200_300), mean(bandwidth_300_400), mean(bandwidth_400_500), mean(bandwidth_500_inf)]
               }
        
        dfOutput = pd.DataFrame(data) 
        fig,ax = self.render_table(dfOutput)
        fig.savefig("%s/%s_stats_combatToExp_varAvgKbps.png" % (self.pathOutput, self.fileName))
        plt.clf()