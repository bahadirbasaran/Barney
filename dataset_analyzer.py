import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from enum import IntEnum

class GameStage(IntEnum):
    EXPLORATION = 0
    COMBAT      = 1
    MENU        = 2
    CONSOLE     = 3
    SCOREBOARD  = 4
    DEATH       = 5

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


    def plot(self):

        sns.set(rc={'figure.figsize': (18, 6)})
        sns.set_theme()

        usedLegends = []  
        legendMap = {GameStage.EXPLORATION: ["darkblue", "Exploration"],
                     GameStage.COMBAT:      ["red", "Combat"],
                     GameStage.MENU:        ["darkorange", "Menu"],
                     GameStage.CONSOLE:     ["lime", "Console"],
                     GameStage.SCOREBOARD:  ["cyan", "Scoreboard"],
                     GameStage.DEATH:       ["blue", "Death"]}  
                   
        dfInput = pd.read_csv("%s/%s.csv" % (self.pathOutput, self.fileName))
        dfInput['bitrate_kbps'] = dfInput['size'].rolling(window=35).sum() * 8 // 1000    

        fig = sns.lineplot(data=dfInput, x="time", y="bitrate_kbps", color='black', linewidth=1)

        formerSec = 0
        for sec, stage in self.videoData.items():

            if stage != GameStage.DEATH:
            
                color, label = legendMap[stage]
                if stage in usedLegends:
                    label = None
                else:
                    usedLegends.append(stage)
                
                plt.axvspan(formerSec, sec, facecolor=color, alpha=0.15, label=label)
                formerSec = sec  
                
            else:
                color, label = legendMap[GameStage.DEATH]
                if label in usedLegends:
                    label = None
                else:
                    usedLegends.append(label)

                plt.axvline(sec, color=color, linestyle="--", linewidth=0.5, label=label)
        
        if self.smooth:
            plt.title("Bitrate/Time per Game Stage [E.M.A with alpha: {:.3f}] - [Overall kbps: {:.3f}]".format(2 / (self.windowLength+1), self.kbpsOverall))
        else:
            plt.title("Bitrate/Time per Game Stage [Overall kbps: {:.3f}]".format(self.kbpsOverall))

        plt.legend(facecolor='white', framealpha=1, loc="upper right") 
        plt.margins(x=0)
        fig.figure.savefig("%s/%s.png" % (self.pathOutput, self.fileName), bbox_inches = "tight")
        plt.clf()

    def get_statistics(self):
        
        dfInput = pd.read_csv("%s/%s.csv" % (self.pathOutput, self.fileName))

        self.kbpsOverall = round((dfInput['size'].sum() / sorted(list(self.videoData.keys()))[-1]) * 8 / 1000, 2)          

        count_total = dfInput.shape[0]

        df_exploration = dfInput[(dfInput['gamestage'] == 0)]
        df_combat      = dfInput[(dfInput['gamestage'] == 1)]
        df_menu        = dfInput[(dfInput['gamestage'] == 2)]
        df_console     = dfInput[(dfInput['gamestage'] == 3)]
        df_scoreboard  = dfInput[(dfInput['gamestage'] == 4)]

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

        def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14, header_color='#40466e', edge_color='w',
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

        fig,ax = render_mpl_table(dfOutput)
        fig.savefig("%s/%s_stats.png" % (self.pathOutput, self.fileName))
        plt.clf()
   