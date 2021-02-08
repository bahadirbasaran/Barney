# Dependency: plotbitrate

import csv
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from itertools import islice
from copy import deepcopy

class DatasetAnalyzer():
    def __init__(self, pathCSV, smooth=None, windowLength=None):
        self.pathCSV      = pathCSV
        self.pathOutput   = '/'.join(pathCSV.split('/')[:-1])
        self.windowLength = windowLength

        self.smooth       = smooth
        if not smooth:
            self.fileName = pathCSV.split('.')[1].split('.')[0].split('/')[-1]
        elif smooth == "exp":
            self.fileName = "%s_smooth_exp_w%i" % (pathCSV.split('.')[1].split('.')[0].split('/')[-1], self.windowLength)
            self.moving_average(exponential=True)
        elif smooth == "simple":
            self.fileName = "%s_smooth_simple_w%i" % (pathCSV.split('.')[1].split('.')[0].split('/')[-1], self.windowLength)
            self.moving_average(exponential=False)

        self.videoData    = {}

    def read_CSV(self):

        with open("%s/%s.csv" % (self.pathOutput, self.fileName), 'r') as csvInput:
            csvReader = csv.reader(csvInput)
            #skip the first row 
            next(csvReader)

            # Fetch the first row seperately for stage comparison
            firstRow  = next(csvReader)
            chunkSize = float(firstRow[1])
            stage     = firstRow[3]
            secStageFirst = float(firstRow[0])

            # To keep the last row before new game stage
            tmpRow = deepcopy(firstRow)
            
            for row in csvReader:
                
                if row[3] == stage:
                    chunkSize += int(row[1])
                
                else:
                    sec      = float(row[0])
                    timeDiff = sec - secStageFirst
                    chunkSize_kb = chunkSize * 8 / 1000
                    kbps     = int(chunkSize_kb / timeDiff)
                    stage    = int(tmpRow[3])
                    self.videoData.update({round(sec, 2): [kbps, stage]})

                    # Update
                    secStageFirst = float(row[0])
                    chunkSize = float(row[1])
                    stage = row[3]
                    
                tmpRow = deepcopy(row)
            
            # For the last game stage seconds. They are not saved with the code block above because they won't see different stage afterward.
            sec = float(tmpRow[0])
            timeDiff = sec - secStageFirst
            chunkSize_kb = chunkSize * 8 / 1000
            kbps     = int(chunkSize_kb / timeDiff)
            stage    = int(tmpRow[3])
            self.videoData.update({round(sec, 2): [kbps, stage]})
        
        self.plot(self.fileName)
        self.get_statistics(self.fileName)

    def moving_average(self, exponential):

        with open(self.pathCSV, 'r') as csvInput, \
                open("%s/%s.csv" % (self.pathOutput, self.fileName), 'w') as csvOutput:

            csvWriter = csv.writer(csvOutput, lineterminator='\n')
            csvReader = csv.reader(csvInput)
            newRows = []

            #skip the first row 
            header = next(csvReader)
            newRows.append(header)

            # Simple Moving Average                                         
            if not exponential:
                pass
                # cursor, shiftCursor, windowSize, n = 0, 0, 0, 0
                # isFirstRow, isSameStage = True, True
                # while True:
                #     # keep the window elements in the list "rows"
                #     rows = [row for row in islice(csvReader, cursor, cursor+self.windowLength)]
                #     if not rows:
                #         break

                #     # first frame of the current window
                #     if isFirstRow:
                #         stage = rows[0][3]
                #         isFirstRow = False
                    
                #     for indice, row in enumerate(rows, start=1):
                #         if row[3] == stage:
                #             windowSize += int(row[1])
                #             n += 1
                #             windowAvg = windowSize // n
                #             newRows.append([row[0], str(windowAvg), row[2], row[3]])
                #             isSameStage = True
                        
                #         # Frame belonging to different game stage
                #         else:
                #             stage = row[3]
                #             isSameStage = False
                #             shiftCursor = indice-1
                #             break
                    
                #     if isSameStage:
                #         cursor += 1
                #     else:
                #         cursor += shiftCursor
                    
                #     windowSize, n = 0, 0 
                           
            # Exponential Moving Average
            else:
                # Smoothing factor
                alpha = 2 / (self.windowLength + 1)

                # Fetch the first row seperately for stage comparison
                firstRow = next(csvReader)
                sec   = firstRow[0]
                type  = firstRow[2]
                stage = firstRow[3]
                formerEMA = int(float(firstRow[1]))

                newRows.append([sec, str(formerEMA), type, stage])

                for row in csvReader:
                    if row[3] == stage:
                        formerEMA = alpha * (float(row[1]) - formerEMA) + formerEMA
                    
                    # Different game stage than the former one
                    else:
                        stage = row[3]
                        formerEMA = float(row[1])

                    newRows.append([row[0], str(round(formerEMA)), row[2], row[3]])    

            csvWriter.writerows(newRows)


    def plot(self, figureName):
        plt.style.use('seaborn')

        if self.smooth == "exp":
            plt.figure(figsize=[28, 8]).canvas
            plt.title("Bitrate/Time per Game Stage [E.M.A with alpha: {:.3f}]".format(2 / (self.windowLength+1)))
        elif self.smooth == "simple":
            plt.figure(figsize=[28, 8]).canvas
            plt.title("Bitrate/Time per Game Stage [Simple Moving Average with Window Length: %i]" % self.windowLength)
        else:
            plt.figure(figsize=[28, 8]).canvas
            plt.title("Bitrate/Time per Game Stage")

        legendMap = {0: ["darkblue", "Exploration"],
                     1: ["yellow", "Combat"],
                     2: ["darkorange", "Menu"],
                     3: ["lime", "Console"],
                     4: ["cyan", "Scoreboard"]}

        x = sorted(list(self.videoData.keys()))     # sorted in case of older Python versions
        y = [self.videoData[xi][0] for xi in x]
        #plt.step([0] + x, [y[0]] + y, color='black', ls=':', lw=0.5, where='pre')
        usedLegendIDs = []        
        
        for xi0, xi1, yi in zip([0] + x[:-1], x, y):
            legendID = self.videoData[xi1][1]
            color, label = legendMap[legendID]
            if legendID in usedLegendIDs:
                label = None
            else:
                usedLegendIDs.append(legendID)
            
            # In case of plotting both bar and line graphs, leave one "label" parameter out.
            #plt.plot([xi0, xi1], [yi] * 2, color=color, label=label)
            plt.bar(xi0, yi, width=xi1-xi0, align='edge', color=color, label=label, alpha=0.4)
            plt.margins(x=0)

        plt.xticks(x)
        plt.gca().xaxis.set_major_locator(MultipleLocator(60))
        plt.gca().yaxis.set_major_locator(MultipleLocator(100))
        plt.xlabel('Time (sec)')
        plt.ylabel("Bitrate (kbit/sec)")
        plt.legend()
        plt.grid(True, axis="y")
        plt.savefig("%s/%s.png" % (self.pathOutput, figureName))

    def get_statistics(self, figureName):
        
        dfInput = pd.read_csv("%s/%s.csv" % (self.pathOutput, figureName))

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
                "Min (kB)":         [min_exploration, min_combat, min_menu, min_console, min_scoreboard],
                "Max (kB)":         [max_exploration, max_combat, max_menu, max_console, max_scoreboard],
                "Mean (kB)":        [mean_exploration,mean_combat, mean_menu, mean_console, mean_scoreboard],
                "Coeff. Variation": [cv_exploration, cv_combat, cv_menu, cv_console, cv_scoreboard],
                "Dominance (%)":    [dom_exploration, dom_combat, dom_menu, dom_console, dom_scoreboard]}

        #dfOutput = pd.DataFrame(data, index =['Exploration', 'Combat', 'Menu', 'Console', 'Scoreboard']) 
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
        fig.savefig("%s/%s_stats.png" % (self.pathOutput, figureName))
   