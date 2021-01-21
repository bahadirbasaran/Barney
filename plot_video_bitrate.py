# Dependency: plotbitrate

import csv
import subprocess
import numpy as np
import matplotlib.pyplot as plt

VIDEO_FPS = 35

class PlotVideoBitrate():
    def __init__(self, listVideoPaths, outputPath, windowLength=VIDEO_FPS, shift=None):
        self.listVideoPaths  = listVideoPaths
        self.outputPath = outputPath
        self.windowLength = windowLength
        self.shift = shift
        self.dictStageData = {}

    def generate_CSV(self):   
        for path in self.listVideoPaths:
            videoName_format = path.split('/')[-1]              # chunk0_combat_35fps.mp4
            videoName    = videoName_format.split('.')[0]       # chunk0_combat_35fps
            pathVideoDir = path.rstrip(videoName_format)
            subprocess.Popen("plotbitrate -f csv_raw -o %s.csv %s" % (videoName, videoName_format), shell=True, cwd=pathVideoDir).wait()
            print("%s%s.csv has been generated!" % (pathVideoDir, videoName))
    
    def read_CSV(self):
        for path in self.listVideoPaths:
            videoName_format = path.split('/')[-1]
            videoName    = videoName_format.split('.')[0]
            pathVideoDir = path.rstrip(videoName_format)
            dictKbps = {}
            chunkSize_bytes, nFrame = 0, 0
            pathCSV = pathVideoDir + "/%s.csv" % videoName
            
            with open(pathCSV, 'r') as fptrCSV:
                rows = csv.reader(fptrCSV)
                next(rows)    #skip the first row    
                
                # chunk-based sum by default
                if not self.shift:
                    for row in rows:
                        frameSize_bytes = int(row[1])
                        chunkSize_bytes += frameSize_bytes
                        nFrame += 1
                        if nFrame % self.windowLength == 0:
                            chunkSize_kbits = chunkSize_bytes * 8 / 1000 
                            #avg = round(chunkSize_kbits / float(self.windowLength), 2)
                            if self.windowLength == VIDEO_FPS:
                                sec = nFrame // self.windowLength
                            else:
                                sec = round(float(row[0]), 4)
                            #dictKbps[sec] = avg
                            dictKbps[sec] = chunkSize_kbits
                            chunkSize_bytes = 0

                # Moving average otherwise
                else:
                    tuples = []
                    for row in rows:
                        sec = round(float(row[0]), 4)
                        frameSize_bytes = int(row[1])
                        tuples.append( (sec, frameSize_bytes) )

                    cursor = 0
                    while cursor < len(tuples):
                        scope    = tuples[cursor : cursor+self.windowLength]
                        scopeSec = round(scope[0][0], 3)
                        scopeAvg_bytes = sum([b for (_ , b) in scope]) / len(scope)
                        scopeAvg_kbits = round(scopeAvg_bytes * 8 / 1000 , 2)
                        dictKbps[scopeSec] = scopeAvg_kbits
                        cursor += self.shift    

            stageName = videoName.split('_')[1]
            mean = round(sum(dictKbps.values()) / float(len(dictKbps.keys())) , 2)
            stageName_mean = stageName + '_' + str(mean)
            self.dictStageData.update({stageName_mean: dictKbps})    

    def plot(self):
        plt.style.use('seaborn')
        if not self.shift:
            plt.figure(figsize=[20, 8]).canvas.set_window_title("Bitrate/Time per Game Stage [%i frames per sec]" % self.windowLength)
        else:
            plt.figure(figsize=[20, 8]).canvas.set_window_title("Bitrate Moving Average per Game Stage [Window Length: %i - Shifted by: %i]" % (self.windowLength, self.shift))
        legends = []
        for stage, data in self.dictStageData.items():
            plt.plot(data.values())
            legends.append("%s - Mean: %s" % (stage.split('_')[0], stage.split('_')[1]))
        plt.legend(legends)
        #pltRange = np.arange(len(next(iter(self.dictStageData.values()))))
        #plt.xticks(pltRange, next(iter(self.dictStageData.values())).keys())
        #plt.xticks(pltRange)
        #plt.yticks([x for x in range(0,50000, 2000)])
        plt.xticks([])
        plt.xlabel('Time (sec)')
        plt.ylabel("Bitrate (kbit/s)")
        plt.grid(True, axis="y")
        plt.savefig(self.outputPath)
   