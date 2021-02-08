import os
import sys
import csv
import subprocess
import numpy as np
from PIL import Image
from itertools import product
from dataset_analyzer import DatasetAnalyzer

FPS = 35

class SceneLabels():
    EXPLORATION = 0
    COMBAT      = 1
    MENU        = 2
    CONSOLE     = 3
    SCOREBOARD  = 4
    CUTSCENE    = 5

def build_dataset():

    if not os.path.exists("./experiment"):
        print("There is no experiment to be used to generate a dataset!")
        return
    
    for id_map in range(1,33):       # for each existing one of 32 maps

        path_exp = "./experiment/map_id_%i" % id_map
        
        while os.path.exists(path_exp):
            
            experiments = os.listdir(path_exp)
            
            for exp in experiments:

                path_dataset = path_exp + "/" + exp + "/dataset"
                
                j = 0
                while os.path.isfile("%s/chunk%i.npz" %(path_dataset, j)):

                    path_chunk = path_dataset + "/chunk%i" % j
                    
                    if os.path.exists(path_chunk):
                        break
                    else:
                        os.mkdir(path_chunk)

                    data = np.load("%s.npz" % path_chunk)
                    
                    n = 0
                    while data.get('frame%i' % n) is not None:
                        arr   = data.get('frame%i' % n)
                        arr   = np.moveaxis(arr , 0, -1) 
                        img   = Image.fromarray(arr)
                        frame = "frame{0:05d}.png".format(n)
                        img.save("%s/%s" % (path_chunk, frame))
                        print(frame) 
                        n += 1
                    print("%s has been created!" % path_chunk)

                    videoName = "chunk%i_400kbps.mp4" % j
                    #subprocess.Popen("ffmpeg -r %i -f image2 -i %s -vcodec libx264 -crf 24 -loglevel quiet %s" % (FPS, "frame%05d.png", videoName), shell=True, cwd=path_chunk).wait()
                    subprocess.Popen("ffmpeg -r %i -f image2 -i %s -vcodec libx264 -b:v 400k -loglevel quiet %s" % (FPS, "frame%05d.png", videoName), shell=True, cwd=path_chunk).wait()
                    subprocess.Popen("plotbitrate -f csv_raw -o raw_400kbps.csv %s" % videoName, shell=True, cwd=path_chunk).wait()
                    print("%s has been recorded!" % videoName)
                    videoName = "chunk%i_500kbps.mp4" % j
                    subprocess.Popen("ffmpeg -r %i -f image2 -i %s -vcodec libx264 -b:v 500k -loglevel quiet %s" % (FPS, "frame%05d.png", videoName), shell=True, cwd=path_chunk).wait()
                    subprocess.Popen("plotbitrate -f csv_raw -o raw_500kbps.csv %s" % videoName, shell=True, cwd=path_chunk).wait()
                    print("%s has been recorded!" % videoName)
                    videoName = "chunk%i_600kbps.mp4" % j
                    subprocess.Popen("ffmpeg -r %i -f image2 -i %s -vcodec libx264 -b:v 600k -loglevel quiet %s" % (FPS, "frame%05d.png", videoName), shell=True, cwd=path_chunk).wait()
                    subprocess.Popen("plotbitrate -f csv_raw -o raw_600kbps.csv %s" % videoName, shell=True, cwd=path_chunk).wait()
                    print("%s has been recorded!" % videoName)

                    print("Frames inside [%s] are being destroyed..." % path_chunk)
                    subprocess.Popen("rm -f *.png", shell=True, cwd=path_chunk).wait()

                    # Append corresponding game stage into each frame in csv
                    for csvName, outputName in zip(["raw_400kbps.csv", "raw_500kbps.csv", "raw_600kbps.csv"],
                                                   ["frames_400kbps.csv", "frames_500kbps.csv", "frames_600kbps.csv"]):

                        with open("%s/%s" % (path_chunk, csvName), 'r') as csvInput, \
                                open("%s/%s" % (path_chunk, outputName), 'w') as csvOutput:

                            csvWriter = csv.writer(csvOutput, lineterminator='\n')
                            csvReader = csv.reader(csvInput)
                            rows = []

                            header = next(csvReader)
                            header.append("gamestage")
                            rows.append(header)

                            for n, row in enumerate(csvReader):
                                #append the game stage
                                stage = int(data.get('label%i' % n))
                                row.append(stage)
                                rows.append(row)

                            csvWriter.writerows(rows)
                    
                    subprocess.Popen("rm -f raw*.csv", shell=True, cwd=path_chunk).wait()

                    for inputName in ["frames_400kbps.csv", "frames_500kbps.csv", "frames_600kbps.csv"]:
                    
                        pathCSV = "%s/%s" % (path_chunk, inputName)
                        
                        obj = DatasetAnalyzer(pathCSV) 
                        obj.read_CSV()

                        for length in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                            obj = DatasetAnalyzer(pathCSV, smooth="exp", windowLength=length)
                            obj.read_CSV()
                    
                    j += 1

            break

if __name__ == "__main__":

    build_dataset()
        
