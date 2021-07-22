import csv
import numpy as np
import pandas as pd
from copy import deepcopy
from enum import IntEnum
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize': (18, 6)})
sns.set_theme()

FPS = 35

class GameStage(IntEnum):
    EXPLORATION = 0
    COMBAT      = 1
    MENU        = 2
    CONSOLE     = 3
    SCOREBOARD  = 4

legendMap = {GameStage.EXPLORATION: ["blue", "Exploration"],
             GameStage.COMBAT:      ["red", "Combat"],
             GameStage.MENU:        ["gold", "Menu"],
             GameStage.CONSOLE:     ["lime", "Console"],
             GameStage.SCOREBOARD:  ["cyan", "Scoreboard"]}

def mean(nums): 
    if nums:                return round(sum(nums) / len(nums), 2)
    else:                   return '-'
def ratio_negatives(nums): 
    if nums:                return round(len([num for num in nums if num < 0]) * 100 / len(nums), 2)
    else:                   return '-'

class DatasetAnalyzer():
    def __init__(self, pathCSV, smooth=None, windowLength=None):
        self.pathCSV      = pathCSV
        self.pathOutput   = '/'.join(pathCSV.split('/')[:-1])
        self.windowLength = windowLength
        self.smooth       = smooth

        self.dfInput     = pd.DataFrame()
        self.videoData   = {}
        self.kbpsOverall = 0
        self.avgStageBitrates = {}
        self.bitrateVariances = {}

        self.read_CSV()

    def read_CSV(self):

        self.dfInput = pd.read_csv(self.pathCSV)

        self.dfInput['bitrateKbps']      = self.dfInput['size'].rolling(window=FPS).sum() * 8 // 1000
        self.dfInput['accumulatedKbits'] = self.dfInput['size'].cumsum() * 8 // 1000

        if self.smooth:
            # ONLY FOR AN EXPERIMENT WITHOUT SYNTHETIC SCREENS
            self.dfInput['gamestageEMA']  = self.dfInput['gamestage'].ewm(span=self.windowLength, adjust=False).mean()
            for i in range(self.dfInput.shape[0]):  
                self.dfInput.loc[i, 'gamestageEMA'] = 1 if self.dfInput.loc[i, 'gamestageEMA'] > 0.5 else 0

            self.dfInput['bitrateKbpsEMA']  = self.dfInput['bitrateKbps'].ewm(span=self.windowLength, adjust=False).mean()
            self.kbpsOverall = round(self.dfInput['bitrateKbpsEMA'].mean(), 2)
            self.fileName = "%s_smooth_w%i" % (self.pathCSV.split('.')[1].split('.')[0].split('/')[-1], self.windowLength)
            columnStage = 'gamestageEMA'
        else:
            self.kbpsOverall = round(self.dfInput['bitrateKbps'].mean(), 2)
            self.fileName = self.pathCSV.split('.')[1].split('.')[0].split('/')[-1]
            columnStage = 'gamestage'

        for index, row in self.dfInput.iterrows():

            if index == 0:  
                firstStage    = self.dfInput.loc[0, columnStage]
                secfirstStage = self.dfInput.loc[0, 'time']
                tmpRow = deepcopy(self.dfInput.iloc[0])
                continue

            if row[columnStage] != firstStage:
                self.videoData[round(row['time'], 6)] = tmpRow[columnStage]
                # Update
                secfirstStage = row['time']
                firstStage    = row[columnStage]
            
            tmpRow = deepcopy(row)
        
        # For the last game stage seconds. They are not saved with the code block above because they won't see different stage afterward.
        self.videoData[round(self.dfInput.iloc[-1]['time'], 6)] = self.dfInput.iloc[-1][columnStage]

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
        
    def get_overall_stats(self):

        count_total = self.dfInput.shape[0]

        if self.smooth: columnStage = 'gamestageEMA'
        else:           columnStage = 'gamestage'

        df_exploration = self.dfInput[(self.dfInput[columnStage] == GameStage.EXPLORATION)]
        df_combat      = self.dfInput[(self.dfInput[columnStage] == GameStage.COMBAT)]
        df_menu        = self.dfInput[(self.dfInput[columnStage] == GameStage.MENU)]
        df_console     = self.dfInput[(self.dfInput[columnStage] == GameStage.CONSOLE)]
        df_scoreboard  = self.dfInput[(self.dfInput[columnStage] == GameStage.SCOREBOARD)]

        min_exploration   = df_exploration['size'].min()
        max_exploration   = df_exploration['size'].max()
        cv_exploration    = round((df_exploration['size'].std() / df_exploration['size'].mean()), 2)
        mean_exploration  = round(df_exploration['size'].mean(), 2)
        count_exploration = df_exploration.shape[0]
        dom_exploration   = round(count_exploration * 100 / count_total, 2)

        min_combat   = df_combat['size'].min()
        max_combat   = df_combat['size'].max()
        cv_combat    = round((df_combat['size'].std() / df_combat['size'].mean()), 2)
        mean_combat  = round(df_combat['size'].mean(), 2)
        count_combat = df_combat.shape[0]
        dom_combat   = round(count_combat * 100 / count_total , 2)

        min_menu     = df_menu['size'].min()
        max_menu     = df_menu['size'].max()
        cv_menu      = round((df_menu['size'].std() / df_menu['size'].mean()), 2)
        mean_menu    = round(df_menu['size'].mean(), 2)
        count_menu   = df_menu.shape[0]
        dom_menu     = round(count_menu * 100 / count_total , 2)

        min_console   = df_console['size'].min()
        max_console   = df_console['size'].max()
        cv_console    = round((df_console['size'].std() / df_console['size'].mean()), 2)
        mean_console  = round(df_console['size'].mean(), 2)
        count_console = df_console.shape[0]
        dom_console   = round(count_console * 100 / count_total , 2)
        
        min_scoreboard   = df_scoreboard['size'].min()
        max_scoreboard   = df_scoreboard['size'].max()
        cv_scoreboard    = round((df_scoreboard['size'].std() / df_scoreboard['size'].mean()), 2)
        mean_scoreboard  = round(df_scoreboard['size'].mean(), 2)
        count_scoreboard = df_scoreboard.shape[0]
        dom_scoreboard   = round(count_scoreboard * 100 / count_total , 2)

        data = {"Game Stage":       ["Exploration", "Combat", "Menu", "Console", "Scoreboard"],
                "Min (Bytes)":       [min_exploration, min_combat, min_menu, min_console, min_scoreboard],
                "Max (Bytes)":       [max_exploration, max_combat, max_menu, max_console, max_scoreboard],
                "Mean (Bytes)":      [mean_exploration,mean_combat, mean_menu, mean_console, mean_scoreboard],
                "Coeff. Variation": [cv_exploration, cv_combat, cv_menu, cv_console, cv_scoreboard],
                "Fraction (%)":     [dom_exploration, dom_combat, dom_menu, dom_console, dom_scoreboard]}

        dfOutput = pd.DataFrame(data) 

        fig,ax = self.render_table(dfOutput)
        fig.savefig("%s/%s_overall_stats.png" % (self.pathOutput, self.fileName))
        plt.clf()

    def plot_bitrate_time(self):
        # In order to avoid using same legend multiple times
        usedLegends = []

        if self.smooth:
            fig = sns.lineplot(data=self.dfInput, x="time", y="bitrateKbpsEMA", color='black', linewidth=1) 
        else:
            fig = sns.lineplot(data=self.dfInput, x="time", y="bitrateKbps", color='black', linewidth=1) 

        formerSec = 0
        for sec, stage in self.videoData.items():

            color, label = legendMap[stage]
            if stage in usedLegends:
                label = None
            else:
                usedLegends.append(stage)
            
            plt.axvspan(formerSec, sec, facecolor=color, alpha=0.15, label=label)
            formerSec = sec

        plt.legend(facecolor='white', framealpha=1, loc="upper right")
        plt.xlabel("Time [s]")
        plt.ylabel("Bitrate [kbps]") 
        plt.margins(x=0)
        fig.figure.savefig("%s/%s.png" % (self.pathOutput, self.fileName), bbox_inches = "tight")
        plt.clf()
    
    def plot_avg_bitrate_per_stage(self, plotVariationsBetweenStages=True):

        if self.smooth: columnStage = 'gamestageEMA'
        else:           columnStage = 'gamestage'

        self.avgStageBitrates = {0.0: 0}

        for index, row in self.dfInput.iterrows():

            if index == 0:
                firstRow = deepcopy(row)
                continue

            if row[columnStage] != firstRow[columnStage]:
                timeDiff = row['time'] - firstRow['time']
                accBitrateStage = int((row['accumulatedKbits'] - firstRow['accumulatedKbits']) // timeDiff)
                self.avgStageBitrates[round(firstRow['time'] + timeDiff/2.0, 3)] = (accBitrateStage, firstRow[columnStage])
                firstRow = deepcopy(row) 

            tmpRow = deepcopy(row)
                
        # for the last game stage
        if tmpRow['time'] != firstRow['time']:
            timeDiff = tmpRow['time'] - firstRow['time']
            accBitrateStage = int((tmpRow['accumulatedKbits'] - firstRow['accumulatedKbits']) // timeDiff)
            self.avgStageBitrates[round(firstRow['time'] + timeDiff/2.0, 3)] = (accBitrateStage, tmpRow[columnStage])

        del self.avgStageBitrates[0.0]

        fig = plt.figure()
        ax  = fig.add_subplot(111, facecolor='white')
        
        # In order to avoid using same legend multiple times
        usedLegends = []

        for sec, (kbps, stage) in self.avgStageBitrates.items():

            color, label = legendMap[stage]
            if stage in usedLegends:
                label = None
            else:
                usedLegends.append(stage)
            
            ax.scatter(x=sec, y=kbps, c=color, label=label) 

        ax.grid(b=True, which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.1)
        plt.margins(x=0.01, y=0.01)
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Bitrate [kbps]")
        fig.canvas.draw()
        plt.savefig("%s/%s_avgBperStage.png" % (self.pathOutput, self.fileName), bbox_inches = "tight")
        plt.clf()

        if plotVariationsBetweenStages:
            """ Variation of Average Bitrate: B(n) - B(n-1) """

            fig = plt.figure()
            ax  = fig.add_subplot(111, facecolor='white')

            usedLegends.clear()
            formerKbps = 0
            for sec, (kbps, stage) in self.avgStageBitrates.items():
                
                if not formerKbps:
                    formerKbps = list(self.avgStageBitrates.values())[0][0]
                    continue

                color, label = legendMap[stage]
                if stage in usedLegends:
                    label = None
                else:
                    usedLegends.append(stage)
                
                ax.scatter(x=sec, y=kbps-formerKbps, c=color, label=label)   
                formerKbps = kbps

            ax.grid(b=True, which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.1)
            plt.margins(x=0.01, y=0.01)
            plt.legend()
            plt.xlabel("Time [s]")
            plt.ylabel("Bitrate [kbps]")
            fig.canvas.draw()
            plt.savefig("%s/%s_varB.png" % (self.pathOutput, self.fileName), bbox_inches = "tight")
            plt.clf()

    def get_stats_avg_bitrate_var_exp_combat(self): 
        """ Variance of average bitrates from exploration to combat or vice-versa (Between an average kbps of former stage – average kbps of latter stage) """

        tuples_kbps_stage = list(self.avgStageBitrates.values())

        exp_combat_bandwidth_0_200,   combat_exp_bandwidth_0_200   = [], []
        exp_combat_bandwidth_200_300, combat_exp_bandwidth_200_300 = [], []
        exp_combat_bandwidth_300_350, combat_exp_bandwidth_300_350 = [], []
        exp_combat_bandwidth_350_400, combat_exp_bandwidth_350_400 = [], []
        exp_combat_bandwidth_400_450, combat_exp_bandwidth_400_450 = [], []
        exp_combat_bandwidth_450_500, combat_exp_bandwidth_450_500 = [], []
        exp_combat_bandwidth_500_inf, combat_exp_bandwidth_500_inf = [], []

        for i in range(len(tuples_kbps_stage)-1):

            baselineKbps = tuples_kbps_stage[i][0]
            varKbps = round((tuples_kbps_stage[i+1][0] - baselineKbps) * 100 / baselineKbps, 2)

            # Transition from Exploration to Combat
            if tuples_kbps_stage[i][1] == GameStage.EXPLORATION and tuples_kbps_stage[i+1][1] == GameStage.COMBAT:

                if   baselineKbps < 200:    exp_combat_bandwidth_0_200.append(varKbps)
                elif baselineKbps < 300:    exp_combat_bandwidth_200_300.append(varKbps)
                elif baselineKbps < 350:    exp_combat_bandwidth_300_350.append(varKbps)
                elif baselineKbps < 400:    exp_combat_bandwidth_350_400.append(varKbps)
                elif baselineKbps < 450:    exp_combat_bandwidth_400_450.append(varKbps)
                elif baselineKbps < 500:    exp_combat_bandwidth_450_500.append(varKbps)
                else:                       exp_combat_bandwidth_500_inf.append(varKbps)

            # Transition from Combat to Exploration
            elif tuples_kbps_stage[i][1] == GameStage.COMBAT and tuples_kbps_stage[i+1][1] == GameStage.EXPLORATION:

                if   baselineKbps < 200:    combat_exp_bandwidth_0_200.append(varKbps)
                elif baselineKbps < 300:    combat_exp_bandwidth_200_300.append(varKbps)
                elif baselineKbps < 350:    combat_exp_bandwidth_300_350.append(varKbps)
                elif baselineKbps < 400:    combat_exp_bandwidth_350_400.append(varKbps)
                elif baselineKbps < 450:    combat_exp_bandwidth_400_450.append(varKbps)
                elif baselineKbps < 500:    combat_exp_bandwidth_450_500.append(varKbps)
                else:                       combat_exp_bandwidth_500_inf.append(varKbps)

        data = {"Exploration->Combat":    ["BW < 200 kbps", "BW 200-300 kbps", "BW 300-350 kbps", "BW 350-400 kbps", "BW 400-450 kbps", "BW 450-500 kbps", "BW > 500 kbps"],
                "Number of Transitions":  [len(exp_combat_bandwidth_0_200), len(exp_combat_bandwidth_200_300), len(exp_combat_bandwidth_300_350), len(exp_combat_bandwidth_350_400),
                                           len(exp_combat_bandwidth_400_450), len(exp_combat_bandwidth_450_500), len(exp_combat_bandwidth_500_inf)],
                "Negative Variance (%)":  [ratio_negatives(exp_combat_bandwidth_0_200), ratio_negatives(exp_combat_bandwidth_200_300), ratio_negatives(exp_combat_bandwidth_300_350),
                                           ratio_negatives(exp_combat_bandwidth_350_400), ratio_negatives(exp_combat_bandwidth_400_450), ratio_negatives(exp_combat_bandwidth_450_500), ratio_negatives(exp_combat_bandwidth_500_inf)],
                "Min. Variance (%)":   [min(exp_combat_bandwidth_0_200, default='-'), min(exp_combat_bandwidth_200_300, default='-'), min(exp_combat_bandwidth_300_350, default='-'), 
                                        min(exp_combat_bandwidth_350_400, default='-'), min(exp_combat_bandwidth_400_450, default='-'), min(exp_combat_bandwidth_450_500, default='-'), min(exp_combat_bandwidth_500_inf, default='-')],
                "Max. Variance (%)":   [max(exp_combat_bandwidth_0_200, default='-'), max(exp_combat_bandwidth_200_300, default='-'), max(exp_combat_bandwidth_300_350, default='-'),
                                        max(exp_combat_bandwidth_350_400, default='-'), max(exp_combat_bandwidth_400_450, default='-'), max(exp_combat_bandwidth_450_500, default='-'), max(exp_combat_bandwidth_500_inf, default='-')],
                "Avg. Variance (%)":   [mean(exp_combat_bandwidth_0_200), mean(exp_combat_bandwidth_200_300), mean(exp_combat_bandwidth_300_350), mean(exp_combat_bandwidth_350_400), mean(exp_combat_bandwidth_400_450), mean(exp_combat_bandwidth_450_500), mean(exp_combat_bandwidth_500_inf)]
               }
        
        dfOutput = pd.DataFrame(data) 
        fig,ax = self.render_table(dfOutput)
        fig.savefig("%s/%s_stats_varAvgKbps_expToCombat.png" % (self.pathOutput, self.fileName))
        plt.clf()

        data = {"Combat->Exploration":    ["BW < 200 kbps", "BW 200-300 kbps", "BW 300-350 kbps", "BW 350-400 kbps", "BW 400-450 kbps", "BW 450-500 kbps", "BW > 500 kbps"],
                "Number of Transitions":  [len(combat_exp_bandwidth_0_200), len(combat_exp_bandwidth_200_300), len(combat_exp_bandwidth_300_350), len(combat_exp_bandwidth_350_400),
                                           len(combat_exp_bandwidth_400_450), len(combat_exp_bandwidth_450_500), len(combat_exp_bandwidth_500_inf)],
                "Negative Variance (%)":  [ratio_negatives(combat_exp_bandwidth_0_200), ratio_negatives(combat_exp_bandwidth_200_300), ratio_negatives(combat_exp_bandwidth_300_350),
                                           ratio_negatives(combat_exp_bandwidth_350_400), ratio_negatives(combat_exp_bandwidth_400_450), ratio_negatives(combat_exp_bandwidth_450_500), ratio_negatives(combat_exp_bandwidth_500_inf)],
                "Min. Variance (%)":   [min(combat_exp_bandwidth_0_200, default='-'), min(combat_exp_bandwidth_200_300, default='-'), min(combat_exp_bandwidth_300_350, default='-'), 
                                        min(combat_exp_bandwidth_350_400, default='-'), min(combat_exp_bandwidth_400_450, default='-'), min(combat_exp_bandwidth_450_500, default='-'), min(combat_exp_bandwidth_500_inf, default='-')],
                "Max. Variance (%)":   [max(combat_exp_bandwidth_0_200, default='-'), max(combat_exp_bandwidth_200_300, default='-'), max(combat_exp_bandwidth_300_350, default='-'),
                                        max(combat_exp_bandwidth_350_400, default='-'), max(combat_exp_bandwidth_400_450, default='-'), max(combat_exp_bandwidth_450_500, default='-'), max(combat_exp_bandwidth_500_inf, default='-')],
                "Avg. Variance (%)":   [mean(combat_exp_bandwidth_0_200), mean(combat_exp_bandwidth_200_300), mean(combat_exp_bandwidth_300_350), mean(combat_exp_bandwidth_350_400), mean(combat_exp_bandwidth_400_450), mean(combat_exp_bandwidth_450_500), mean(combat_exp_bandwidth_500_inf)]
               }
        
        dfOutput = pd.DataFrame(data) 
        fig,ax = self.render_table(dfOutput)
        fig.savefig("%s/%s_stats_varAvgKbps_combatToExp.png" % (self.pathOutput, self.fileName))
        plt.clf()

    def plot_bitrate_var_on_stage_changes(self, frameRange):
        """ Bitrate Variance between 'frameRange' frames before a change and 'frameRange' frames after a change (maximum)"""

        if self.smooth: 
            columnStage    = 'gamestageEMA'
            columnBitrate  = 'bitrateKbpsEMA'
        else:           
            columnStage    = 'gamestage'
            columnBitrate  = 'bitrateKbps'

        # There is no calculated bitrate value before 35th frame.
        tmpRow = deepcopy(self.dfInput.iloc[FPS-1])
        totalRows = self.dfInput.shape[0]

        for index, row in self.dfInput.iterrows():

            # There is no calculated bitrate value before 35th frame.
            if index < FPS:   continue

            # Game stage change
            if row[columnStage] != tmpRow[columnStage]:

                # Lowerbound and upperbound must be at the same distance from the index of change
                counterBackward = counterForward = 0

                while counterBackward < frameRange and \
                        index-(counterBackward+1) >= FPS-1 and \
                            self.dfInput.iloc[index-(counterBackward+1)][columnStage] == tmpRow[columnStage]:
                    counterBackward += 1

                while counterForward < frameRange and \
                        index+(counterForward+1) < totalRows and \
                            self.dfInput.iloc[index+(counterForward+1)][columnStage] == row[columnStage]:
                    counterForward += 1

                # if counterBackward is not zero (to avoid cases like: 0001000)
                if counterBackward == 0 or counterForward == 0:
                    baselineKbps = self.dfInput.iloc[index-1][columnBitrate]
                    varKbps = int((self.dfInput.iloc[index][columnBitrate] - baselineKbps) * 100 / baselineKbps)
                else:
                    bound = counterBackward if counterBackward < counterForward else counterForward
                    baselineKbps = self.dfInput.iloc[index-bound][columnBitrate]
                    varKbps = int((self.dfInput.iloc[index+bound][columnBitrate] - baselineKbps) * 100 / baselineKbps)

                self.bitrateVariances[(round(row['time'], 3), baselineKbps)] = (varKbps, tmpRow[columnStage], row[columnStage])
            
            tmpRow = deepcopy(row)

        # In order to avoid using same legend multiple times
        usedLegends = []

        fig = plt.figure()
        ax  = fig.add_subplot(111, facecolor='white')

        for (sec, baselineKbps), (varKbps, formerStage, latterStage) in self.bitrateVariances.items():

            color, label = legendMap[latterStage]
            if latterStage in usedLegends:
                label = None
            else:
                usedLegends.append(latterStage)
            
            ax.scatter(x=sec, y=varKbps, c=color, label=label) 

        ax.grid(b=True, which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.1)
        plt.margins(x=0.01, y=0.01)
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Bitrate [kbps]")
        fig.canvas.draw()
        plt.savefig("%s/%s_kbps_var_range%i.png" % (self.pathOutput, self.fileName, frameRange), bbox_inches = "tight")
        plt.clf()

    def get_stats_bitrate_var_on_stage_changes(self, frameRange): 
        """ Bitrate Variances between 'frameRange' frames before change and 'frameRange' frames after change (maximum)"""

        exp_combat_bandwidth_0_200,   combat_exp_bandwidth_0_200   = [], []
        exp_combat_bandwidth_200_300, combat_exp_bandwidth_200_300 = [], []
        exp_combat_bandwidth_300_350, combat_exp_bandwidth_300_350 = [], []
        exp_combat_bandwidth_350_400, combat_exp_bandwidth_350_400 = [], []
        exp_combat_bandwidth_400_450, combat_exp_bandwidth_400_450 = [], []
        exp_combat_bandwidth_450_500, combat_exp_bandwidth_450_500 = [], []
        exp_combat_bandwidth_500_inf, combat_exp_bandwidth_500_inf = [], []

        for (sec, baselineKbps), (varKbps, formerStage, latterStage) in self.bitrateVariances.items():
            # Transition from Exploration to Combat
            if formerStage == GameStage.EXPLORATION and latterStage == GameStage.COMBAT:

                if   baselineKbps < 200:    exp_combat_bandwidth_0_200.append(varKbps)
                elif baselineKbps < 300:    exp_combat_bandwidth_200_300.append(varKbps)
                elif baselineKbps < 350:    exp_combat_bandwidth_300_350.append(varKbps)
                elif baselineKbps < 400:    exp_combat_bandwidth_350_400.append(varKbps)
                elif baselineKbps < 450:    exp_combat_bandwidth_400_450.append(varKbps)
                elif baselineKbps < 500:    exp_combat_bandwidth_450_500.append(varKbps)
                else:                       exp_combat_bandwidth_500_inf.append(varKbps)

            # Transition from Combat to Exploration
            if formerStage == GameStage.COMBAT and latterStage == GameStage.EXPLORATION:

                if   baselineKbps < 200:    combat_exp_bandwidth_0_200.append(varKbps)
                elif baselineKbps < 300:    combat_exp_bandwidth_200_300.append(varKbps)
                elif baselineKbps < 350:    combat_exp_bandwidth_300_350.append(varKbps)
                elif baselineKbps < 400:    combat_exp_bandwidth_350_400.append(varKbps)
                elif baselineKbps < 450:    combat_exp_bandwidth_400_450.append(varKbps)
                elif baselineKbps < 500:    combat_exp_bandwidth_450_500.append(varKbps)
                else:                       combat_exp_bandwidth_500_inf.append(varKbps)


        data = {"Exploration->Combat":    ["BW < 200 kbps", "BW 200-300 kbps", "BW 300-350 kbps", "BW 350-400 kbps", "BW 400-450 kbps", "BW 450-500 kbps", "BW > 500 kbps"],
                "Number of Transitions":  [len(exp_combat_bandwidth_0_200), len(exp_combat_bandwidth_200_300), len(exp_combat_bandwidth_300_350), len(exp_combat_bandwidth_350_400),
                                           len(exp_combat_bandwidth_400_450), len(exp_combat_bandwidth_450_500), len(exp_combat_bandwidth_500_inf)],
                "Negative Variance (%)":  [ratio_negatives(exp_combat_bandwidth_0_200), ratio_negatives(exp_combat_bandwidth_200_300), ratio_negatives(exp_combat_bandwidth_300_350),
                                           ratio_negatives(exp_combat_bandwidth_350_400), ratio_negatives(exp_combat_bandwidth_400_450), ratio_negatives(exp_combat_bandwidth_450_500), ratio_negatives(exp_combat_bandwidth_500_inf)],
                "Min. Variance (%)":   [min(exp_combat_bandwidth_0_200, default='-'), min(exp_combat_bandwidth_200_300, default='-'), min(exp_combat_bandwidth_300_350, default='-'), 
                                        min(exp_combat_bandwidth_350_400, default='-'), min(exp_combat_bandwidth_400_450, default='-'), min(exp_combat_bandwidth_450_500, default='-'), min(exp_combat_bandwidth_500_inf, default='-')],
                "Max. Variance (%)":   [max(exp_combat_bandwidth_0_200, default='-'), max(exp_combat_bandwidth_200_300, default='-'), max(exp_combat_bandwidth_300_350, default='-'),
                                        max(exp_combat_bandwidth_350_400, default='-'), max(exp_combat_bandwidth_400_450, default='-'), max(exp_combat_bandwidth_450_500, default='-'), max(exp_combat_bandwidth_500_inf, default='-')],
                "Avg. Variance (%)":   [mean(exp_combat_bandwidth_0_200), mean(exp_combat_bandwidth_200_300), mean(exp_combat_bandwidth_300_350), mean(exp_combat_bandwidth_350_400), mean(exp_combat_bandwidth_400_450), mean(exp_combat_bandwidth_450_500), mean(exp_combat_bandwidth_500_inf)]
               }
        
        dfOutput = pd.DataFrame(data) 
        fig,ax = self.render_table(dfOutput)
        fig.savefig("%s/%s_kbps_var_range%i_exp_combat.png" % (self.pathOutput, self.fileName, frameRange))
        plt.clf()

        data = {"Combat->Exploration":    ["BW < 200 kbps", "BW 200-300 kbps", "BW 300-350 kbps", "BW 350-400 kbps", "BW 400-450 kbps", "BW 450-500 kbps", "BW > 500 kbps"],
                "Number of Transitions":  [len(combat_exp_bandwidth_0_200), len(combat_exp_bandwidth_200_300), len(combat_exp_bandwidth_300_350), len(combat_exp_bandwidth_350_400),
                                           len(combat_exp_bandwidth_400_450), len(combat_exp_bandwidth_450_500), len(combat_exp_bandwidth_500_inf)],
                "Negative Variance (%)":  [ratio_negatives(combat_exp_bandwidth_0_200), ratio_negatives(combat_exp_bandwidth_200_300), ratio_negatives(combat_exp_bandwidth_300_350),
                                           ratio_negatives(combat_exp_bandwidth_350_400), ratio_negatives(combat_exp_bandwidth_400_450), ratio_negatives(combat_exp_bandwidth_450_500), ratio_negatives(combat_exp_bandwidth_500_inf)],
                "Min. Variance (%)":   [min(combat_exp_bandwidth_0_200, default='-'), min(combat_exp_bandwidth_200_300, default='-'), min(combat_exp_bandwidth_300_350, default='-'), 
                                        min(combat_exp_bandwidth_350_400, default='-'), min(combat_exp_bandwidth_400_450, default='-'), min(combat_exp_bandwidth_450_500, default='-'), min(combat_exp_bandwidth_500_inf, default='-')],
                "Max. Variance (%)":   [max(combat_exp_bandwidth_0_200, default='-'), max(combat_exp_bandwidth_200_300, default='-'), max(combat_exp_bandwidth_300_350, default='-'),
                                        max(combat_exp_bandwidth_350_400, default='-'), max(combat_exp_bandwidth_400_450, default='-'), max(combat_exp_bandwidth_450_500, default='-'), max(combat_exp_bandwidth_500_inf, default='-')],
                "Avg. Variance (%)":   [mean(combat_exp_bandwidth_0_200), mean(combat_exp_bandwidth_200_300), mean(combat_exp_bandwidth_300_350), mean(combat_exp_bandwidth_350_400), mean(combat_exp_bandwidth_400_450), mean(combat_exp_bandwidth_450_500), mean(combat_exp_bandwidth_500_inf)]
               }
        
        dfOutput = pd.DataFrame(data) 
        fig,ax = self.render_table(dfOutput)
        fig.savefig("%s/%s_kbps_var_range%i_combat_exp.png" % (self.pathOutput, self.fileName, frameRange))
        plt.clf()

    def get_transition_matrix_exp_combat(self):

        if self.smooth: columnStage = 'gamestageEMA'
        else:           columnStage = 'gamestage'

        countTransitionExpToCombat = countTransitionCombatToExp = countTransitionExpToExp = countTransitionCombatToCombat = 0

        for i in range(self.dfInput.shape[0] - 1):

            formerStage, latterStage = self.dfInput.loc[i, columnStage], self.dfInput.loc[i+1, columnStage]

            if   formerStage == GameStage.EXPLORATION and latterStage == GameStage.COMBAT:      countTransitionExpToCombat    += 1
            elif formerStage == GameStage.COMBAT and latterStage == GameStage.EXPLORATION:      countTransitionCombatToExp    += 1
            elif formerStage == GameStage.EXPLORATION and latterStage == GameStage.EXPLORATION: countTransitionExpToExp       += 1
            elif formerStage == GameStage.COMBAT and latterStage == GameStage.COMBAT:           countTransitionCombatToCombat += 1

        transitionExpToExp       = round(countTransitionExpToExp/(countTransitionExpToCombat+countTransitionExpToExp), 2)
        transitionExpToCombat    = round(countTransitionExpToCombat/(countTransitionExpToCombat+countTransitionExpToExp), 2)
        transitionCombatToCombat = round(countTransitionCombatToCombat/(countTransitionCombatToCombat+countTransitionCombatToExp), 2)
        transitionCombatToExp    = round(countTransitionCombatToExp/(countTransitionCombatToCombat+countTransitionCombatToExp), 2)
        
        data = {"Game Stage":  ["Exploration", "Combat"],
                "Exploration": [transitionExpToExp, transitionCombatToExp],
                "Combat":      [transitionExpToCombat, transitionCombatToCombat]
               }
        
        dfOutput = pd.DataFrame(data) 
        fig,ax = self.render_table(dfOutput)
        fig.savefig("%s/%s_transitionMatrix.png" % (self.pathOutput, self.fileName))
        plt.clf()

    def get_stats_bitrate_on_exp_combat(self):
        """ For each phase, what are the min-max-avg bitrate? How is the distribution on different bandwidths? """

        if self.smooth: 
            columnStage    = 'gamestageEMA'
            columnBitrate = 'bitrateKbpsEMA'
        else:           
            columnStage    = 'gamestage'
            columnBitrate = 'bitrateKbps'

        # Trim the first 34 rows, since those are nan in bitrate
        self.dfInput.drop(range(FPS-1), inplace=True)
        self.dfInput.reset_index(drop=True, inplace=True)

        df_exploration = self.dfInput[(self.dfInput[columnStage] == GameStage.EXPLORATION)]
        df_combat      = self.dfInput[(self.dfInput[columnStage] == GameStage.COMBAT)]

        df_exploration_0_100   = df_exploration[(df_exploration[columnBitrate] < 100)]
        df_exploration_100_200 = df_exploration[(df_exploration[columnBitrate] >= 100) & (df_exploration[columnBitrate] < 200)]
        df_exploration_200_300 = df_exploration[(df_exploration[columnBitrate] >= 200) & (df_exploration[columnBitrate] < 300)]
        df_exploration_300_400 = df_exploration[(df_exploration[columnBitrate] >= 300) & (df_exploration[columnBitrate] < 400)]
        df_exploration_400_500 = df_exploration[(df_exploration[columnBitrate] >= 400) & (df_exploration[columnBitrate] < 500)]
        df_exploration_500_inf = df_exploration[(df_exploration[columnBitrate] > 500)]

        df_combat_0_100   = df_combat[(df_combat[columnBitrate] < 100)]
        df_combat_100_200 = df_combat[(df_combat[columnBitrate] >= 100) & (df_combat[columnBitrate] < 200)]
        df_combat_200_300 = df_combat[(df_combat[columnBitrate] >= 200) & (df_combat[columnBitrate] < 300)]
        df_combat_300_400 = df_combat[(df_combat[columnBitrate] >= 300) & (df_combat[columnBitrate] < 400)]
        df_combat_400_500 = df_combat[(df_combat[columnBitrate] >= 400) & (df_combat[columnBitrate] < 500)]
        df_combat_500_inf = df_combat[(df_combat[columnBitrate] > 500)]

        data = [["< 100",  df_exploration_0_100.shape[0],   df_combat_0_100.shape[0]],
                ["100-200",df_exploration_100_200.shape[0], df_combat_100_200.shape[0]],
                ["200-300",df_exploration_200_300.shape[0], df_combat_200_300.shape[0]],
                ["300-400",df_exploration_300_400.shape[0], df_combat_300_400.shape[0]],
                ["400-500",df_exploration_400_500.shape[0], df_combat_400_500.shape[0]],
                ["> 500",  df_exploration_500_inf.shape[0], df_combat_500_inf.shape[0]]
               ]

        dfOutput = pd.DataFrame(data, columns=["Kbps", "Exploration", "Combat"])
        ax = dfOutput.plot(x="Kbps", y=["Exploration", "Combat"], kind="bar", figsize=(10,6), rot=0)
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.015), fontsize=8)

        plt.ylabel("Number of Frames") 
        plt.xlabel("Bitrate [kbps]")
        plt.savefig("%s/%s_countHistogram.png" % (self.pathOutput, self.fileName), bbox_inches = "tight")
        plt.clf()
        
        data = {"Game Stage":   ["Exploration", "Combat"],
                "Min. Bitrate": [round(df_exploration[columnBitrate].min(), 2), round(df_combat[columnBitrate].min(), 2)],
                "Max. Bitrate": [round(df_exploration[columnBitrate].max(), 2), round(df_combat[columnBitrate].max(), 2)],
                "Avg. Bitrate": [round(df_exploration[columnBitrate].mean(), 2), round(df_combat[columnBitrate].mean(), 2)]
               }
        dfOutput = pd.DataFrame(data) 
        fig,ax = self.render_table(dfOutput)
        fig.savefig("%s/%s_bitrateStats.png" % (self.pathOutput, self.fileName))
        plt.clf()

    def get_stats_bitrate_var_exp_combat(self):
        """ What are the min-avg-max variation on bitrate when you transition from E->C - C->E - E->E - C->C (on different baseline bitrates) """

        if self.smooth: 
            columnStage   = 'gamestageEMA'
            columnBitrate = 'bitrateKbpsEMA'
        else:           
            columnStage   = 'gamestage'
            columnBitrate = 'bitrateKbps'

        bitrateVariance_exp_combat = {}
        bitrateVariance_combat_exp = {}
        bitrateVariance_exp_exp = {}
        bitrateVariance_combat_combat = {}

        for i in range(self.dfInput.shape[0]-1):

            formerStage, latterStage = self.dfInput.loc[i, columnStage], self.dfInput.loc[i+1, columnStage]
            baseBitrate = self.dfInput.loc[i, columnBitrate]
            variance = round((self.dfInput.loc[i+1, columnBitrate] - baseBitrate) * 100 / baseBitrate, 2)

            if formerStage == GameStage.EXPLORATION and latterStage == GameStage.COMBAT:

                if baseBitrate not in bitrateVariance_exp_combat:
                    bitrateVariance_exp_combat[baseBitrate] = variance
                else:
                    bitrateVariance_exp_combat[baseBitrate] = round((bitrateVariance_exp_combat[baseBitrate] + variance) / 2, 2)

            elif formerStage == GameStage.COMBAT and latterStage == GameStage.EXPLORATION:

                if baseBitrate not in bitrateVariance_combat_exp:
                    bitrateVariance_combat_exp[baseBitrate] = variance
                else:
                    bitrateVariance_combat_exp[baseBitrate] = round((bitrateVariance_combat_exp[baseBitrate] + variance) / 2, 2)
                
            elif formerStage == GameStage.EXPLORATION and latterStage == GameStage.EXPLORATION:

                if baseBitrate not in bitrateVariance_exp_exp:
                    bitrateVariance_exp_exp[baseBitrate] = variance
                else:
                    bitrateVariance_exp_exp[baseBitrate] = round((bitrateVariance_exp_exp[baseBitrate] + variance) / 2, 2)

            elif formerStage == GameStage.COMBAT and latterStage == GameStage.COMBAT:           

                if baseBitrate not in bitrateVariance_combat_combat:
                    bitrateVariance_combat_combat[baseBitrate] = variance
                else:
                    bitrateVariance_combat_combat[baseBitrate] = round((bitrateVariance_combat_combat[baseBitrate] + variance) / 2, 2)

        exp_combat_bandwidth_0_200 = []
        exp_combat_bandwidth_200_300 = []
        exp_combat_bandwidth_300_350 = []
        exp_combat_bandwidth_350_400 = []
        exp_combat_bandwidth_400_450 = []
        exp_combat_bandwidth_450_500 = []
        exp_combat_bandwidth_500_inf = []
        combat_exp_bandwidth_0_200 = []
        combat_exp_bandwidth_200_300 = []
        combat_exp_bandwidth_300_350 = []
        combat_exp_bandwidth_350_400 = []
        combat_exp_bandwidth_400_450 = []
        combat_exp_bandwidth_450_500 = []
        combat_exp_bandwidth_500_inf = []
        exp_exp_bandwidth_0_200 = []
        exp_exp_bandwidth_200_300 = []
        exp_exp_bandwidth_300_350 = []
        exp_exp_bandwidth_350_400 = []
        exp_exp_bandwidth_400_450 = []
        exp_exp_bandwidth_450_500 = []
        exp_exp_bandwidth_500_inf = []
        combat_combat_bandwidth_0_200 = []
        combat_combat_bandwidth_200_300 = []
        combat_combat_bandwidth_300_350 = []
        combat_combat_bandwidth_350_400 = []
        combat_combat_bandwidth_400_450 = []
        combat_combat_bandwidth_450_500 = []
        combat_combat_bandwidth_500_inf = []
        
        for (b, var) in bitrateVariance_exp_combat.items():
            if b < 200:
                exp_combat_bandwidth_0_200.append(var)
            elif b < 300:
                exp_combat_bandwidth_200_300.append(var)
            elif b < 350:
                exp_combat_bandwidth_300_350.append(var)
            elif b < 400:
                exp_combat_bandwidth_350_400.append(var)
            elif b < 450:
                exp_combat_bandwidth_400_450.append(var)
            elif b < 500:
                exp_combat_bandwidth_450_500.append(var)
            else:
                exp_combat_bandwidth_500_inf.append(var)

        for (b, var) in bitrateVariance_combat_exp.items():
            if b < 200:
                combat_exp_bandwidth_0_200.append(var)
            elif b < 300:
                combat_exp_bandwidth_200_300.append(var)
            elif b < 350:
                combat_exp_bandwidth_300_350.append(var)
            elif b < 400:
                combat_exp_bandwidth_350_400.append(var)
            elif b < 450:
                combat_exp_bandwidth_400_450.append(var)
            elif b < 500:
                combat_exp_bandwidth_450_500.append(var)
            else:
                combat_exp_bandwidth_500_inf.append(var)

        for (b, var) in bitrateVariance_exp_exp.items():
            if b < 200:
                exp_exp_bandwidth_0_200.append(var)
            elif b < 300:
                exp_exp_bandwidth_200_300.append(var)
            elif b < 350:
                exp_exp_bandwidth_300_350.append(var)
            elif b < 400:
                exp_exp_bandwidth_350_400.append(var)
            elif b < 450:
                exp_exp_bandwidth_400_450.append(var)
            elif b < 500:
                exp_exp_bandwidth_450_500.append(var)
            else:
                exp_exp_bandwidth_500_inf.append(var)

        for (b, var) in bitrateVariance_combat_combat.items():
            if b < 200:
                combat_combat_bandwidth_0_200.append(var)
            elif b < 300:
                combat_combat_bandwidth_200_300.append(var)
            elif b < 350:
                combat_combat_bandwidth_300_350.append(var)
            elif b < 400:
                combat_combat_bandwidth_350_400.append(var)
            elif b < 450:
                combat_combat_bandwidth_400_450.append(var)
            elif b < 500:
                combat_combat_bandwidth_450_500.append(var)
            else:
                combat_combat_bandwidth_500_inf.append(var)
        
        data = {"Exploration->Combat": ["BW 0 < 200 kbps", "BW 200-300 kbps", "BW 300-350 kbps", "BW 350-400 kbps", "BW 400-450 kbps", "BW 450-500 kbps", "BW > 500 kbps"],
                "Min. Variance (%)": [min(exp_combat_bandwidth_0_200, default='-'), min(exp_combat_bandwidth_200_300, default='-'), min(exp_combat_bandwidth_300_350, default='-'), 
                                      min(exp_combat_bandwidth_350_400, default='-'), min(exp_combat_bandwidth_400_450, default='-'), min(exp_combat_bandwidth_450_500, default='-'), min(exp_combat_bandwidth_500_inf, default='-')],
                "Max. Variance (%)": [max(exp_combat_bandwidth_0_200, default='-'), max(exp_combat_bandwidth_200_300, default='-'), max(exp_combat_bandwidth_300_350, default='-'), 
                                      max(exp_combat_bandwidth_350_400, default='-'), max(exp_combat_bandwidth_400_450, default='-'), max(exp_combat_bandwidth_450_500, default='-'), max(exp_combat_bandwidth_500_inf, default='-')],
                "Avg. Variance (%)": [mean(exp_combat_bandwidth_0_200), mean(exp_combat_bandwidth_200_300), mean(exp_combat_bandwidth_300_350), 
                                      mean(exp_combat_bandwidth_350_400), mean(exp_combat_bandwidth_400_450), mean(exp_combat_bandwidth_450_500), mean(exp_combat_bandwidth_500_inf)]
               }
        
        dfOutput = pd.DataFrame(data) 
        fig,ax = self.render_table(dfOutput)
        fig.savefig("%s/%s_bitrateVar_exp_combat.png" % (self.pathOutput, self.fileName))
        plt.clf()

        data = {"Combat->Exploration": ["BW 0 < 200 kbps", "BW 200-300 kbps", "BW 300-350 kbps", "BW 350-400 kbps", "BW 400-450 kbps", "BW 450-500 kbps", "BW > 500 kbps"],
                "Min. Variance (%)": [min(combat_exp_bandwidth_0_200, default='-'), min(combat_exp_bandwidth_200_300, default='-'), min(combat_exp_bandwidth_300_350, default='-'), 
                                      min(combat_exp_bandwidth_350_400, default='-'), min(combat_exp_bandwidth_400_450, default='-'), min(combat_exp_bandwidth_450_500, default='-'), min(combat_exp_bandwidth_500_inf, default='-')],
                "Max. Variance (%)": [max(combat_exp_bandwidth_0_200, default='-'), max(combat_exp_bandwidth_200_300, default='-'), max(combat_exp_bandwidth_300_350, default='-'), 
                                      max(combat_exp_bandwidth_350_400, default='-'), max(combat_exp_bandwidth_400_450, default='-'), max(combat_exp_bandwidth_450_500, default='-'), max(combat_exp_bandwidth_500_inf, default='-')],
                "Avg. Variance (%)": [mean(combat_exp_bandwidth_0_200), mean(combat_exp_bandwidth_200_300), mean(combat_exp_bandwidth_300_350), 
                                      mean(combat_exp_bandwidth_350_400), mean(combat_exp_bandwidth_400_450), mean(combat_exp_bandwidth_450_500), mean(combat_exp_bandwidth_500_inf)]
               }
        
        dfOutput = pd.DataFrame(data) 
        fig,ax = self.render_table(dfOutput)
        fig.savefig("%s/%s_bitrateVar_combat_exp.png" % (self.pathOutput, self.fileName))
        plt.clf()

        data = {"Exploration->Exploration": ["BW 0 < 200 kbps", "BW 200-300 kbps", "BW 300-350 kbps", "BW 350-400 kbps", "BW 400-450 kbps", "BW 450-500 kbps", "BW > 500 kbps"],
                "Min. Variance (%)": [min(exp_exp_bandwidth_0_200, default='-'), min(exp_exp_bandwidth_200_300, default='-'), min(exp_exp_bandwidth_300_350, default='-'), 
                                      min(exp_exp_bandwidth_350_400, default='-'), min(exp_exp_bandwidth_400_450, default='-'), min(exp_exp_bandwidth_450_500, default='-'), min(exp_exp_bandwidth_500_inf, default='-')],
                "Max. Variance (%)": [max(exp_exp_bandwidth_0_200, default='-'), max(exp_exp_bandwidth_200_300, default='-'), max(exp_exp_bandwidth_300_350, default='-'), 
                                      max(exp_exp_bandwidth_350_400, default='-'), max(exp_exp_bandwidth_400_450, default='-'), max(exp_exp_bandwidth_450_500, default='-'), max(exp_exp_bandwidth_500_inf, default='-')],
                "Avg. Variance (%)": [mean(exp_exp_bandwidth_0_200), mean(exp_exp_bandwidth_200_300), mean(exp_exp_bandwidth_300_350), 
                                      mean(exp_exp_bandwidth_350_400), mean(exp_exp_bandwidth_400_450), mean(exp_exp_bandwidth_450_500), mean(exp_exp_bandwidth_500_inf)]
               }
        
        dfOutput = pd.DataFrame(data) 
        fig,ax = self.render_table(dfOutput)
        fig.savefig("%s/%s_bitrateVar_exp_exp.png" % (self.pathOutput, self.fileName))
        plt.clf()

        data = {"Combat->Combat": ["BW 0 < 200 kbps", "BW 200-300 kbps", "BW 300-350 kbps", "BW 350-400 kbps", "BW 400-450 kbps", "BW 450-500 kbps", "BW > 500 kbps"],
                "Min. Variance (%)": [min(combat_combat_bandwidth_0_200, default='-'), min(combat_combat_bandwidth_200_300, default='-'), min(combat_combat_bandwidth_300_350, default='-'), 
                                      min(combat_combat_bandwidth_350_400, default='-'), min(combat_combat_bandwidth_400_450, default='-'), min(combat_combat_bandwidth_450_500, default='-'), min(combat_combat_bandwidth_500_inf, default='-')],
                "Max. Variance (%)": [max(combat_combat_bandwidth_0_200, default='-'), max(combat_combat_bandwidth_200_300, default='-'), max(combat_combat_bandwidth_300_350, default='-'), 
                                      max(combat_combat_bandwidth_350_400, default='-'), max(combat_combat_bandwidth_400_450, default='-'), max(combat_combat_bandwidth_450_500, default='-'), max(combat_combat_bandwidth_500_inf, default='-')],
                "Avg. Variance (%)": [mean(combat_combat_bandwidth_0_200), mean(combat_combat_bandwidth_200_300), mean(combat_combat_bandwidth_300_350), 
                                      mean(combat_combat_bandwidth_350_400), mean(combat_combat_bandwidth_400_450), mean(combat_combat_bandwidth_450_500), mean(combat_combat_bandwidth_500_inf)]
               }
        
        dfOutput = pd.DataFrame(data) 
        fig,ax = self.render_table(dfOutput)
        fig.savefig("%s/%s_bitrateVar_combat_combat.png" % (self.pathOutput, self.fileName))
        plt.clf()