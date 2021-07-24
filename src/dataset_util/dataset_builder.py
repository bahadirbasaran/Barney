import os
import csv
import subprocess
import numpy as np
from PIL import Image
from dataset_analyzer import DatasetAnalyzer

FPS = 35
NUM_MAPS = 32

def build_dataset():

    if not os.path.exists("./experiment"):
        print("There is no experiment to be used to generate a dataset!")
        return
    
    for id_map in range(1, NUM_MAPS):       

        path_exp = "./experiment/map_id_%i" % id_map
        
        if os.path.exists(path_exp):
            
            experiments = os.listdir(path_exp)
            
            for exp in experiments:

                path_dataset = path_exp + "/" + exp + "/dataset"
                
                labels = []

                j = n = 0
                while os.path.isfile("%s/chunk%i.npz" %(path_dataset, j)):

                    path_chunk = path_dataset + "/chunk%i" % j

                    data = np.load("%s.npz" % path_chunk)
                    
                    while data.get('frame%i' % n) is not None:
                        labels.append(data.get('label%i' % n))
                        arr   = data.get('frame%i' % n)
                        arr   = np.moveaxis(arr , 0, -1) 
                        img   = Image.fromarray(arr)
                        frame = "frame{0:06d}.png".format(n)
                        img.save("%s/%s" % (path_dataset, frame))
                        print(frame) 
                        n += 1

                    print("%s has been extracted!" % path_chunk)   # To avoid processing the "_500/600kbps" csv files.
                    
                    j += 1

                videoName = "video_400kbps.mp4"
                subprocess.Popen("ffmpeg -r %i -f image2 -i %s -vcodec libx264 -b:v 400k -loglevel quiet %s" % (FPS, "frame%06d.png", videoName), shell=True, cwd=path_dataset).wait()
                subprocess.Popen("plotbitrate -f csv_raw -o raw_400kbps.csv %s" % videoName, shell=True, cwd=path_dataset).wait()
                print("%s has been recorded!" % videoName)

                print("Frames inside [%s] are being destroyed..." % path_dataset)
                subprocess.Popen("for f in *.png; do rm \"$f\"; done", shell=True, cwd=path_dataset).wait()

                with open("%s/%s" % (path_dataset, "raw_400kbps.csv"), 'r') as csvInput, \
                        open("%s/%s" % (path_dataset, "frames_400kbps.csv"), 'w') as csvOutput:

                    csvWriter = csv.writer(csvOutput, lineterminator='\n')
                    csvReader = csv.reader(csvInput)
                    rows = []

                    header = next(csvReader)
                    header.append("gamestage")
                    rows.append(header)

                    for index, row in enumerate(csvReader):
                        row.append(labels[index])
                        rows.append(row)

                    csvWriter.writerows(rows)
                
                # Delete the raw dataset
                subprocess.Popen("rm -f raw_400kbps.csv", shell=True, cwd=path_dataset).wait() 
                
                pathCSV = "%s/%s" % (path_dataset, "frames_400kbps.csv")
                
                analyzer = DatasetAnalyzer(pathCSV)
                analyzer.plot_bitrate_time()
                
                for length in [9, 18]:  # corresponding to 0.25, 0.5 seconds
                    analyzer = DatasetAnalyzer(pathCSV, smooth=True, windowLength=length)
                    analyzer.get_overall_stats()
                    analyzer.plot_bitrate_time()
                    analyzer.plot_avg_bitrate_per_stage(plotVariationsBetweenStages=True)
                    analyzer.get_stats_avg_bitrate_var_exp_combat()

                    for frameRange in [9, 18]:
                        analyzer.plot_bitrate_var_on_stage_changes(frameRange)
                        analyzer.get_stats_bitrate_var_on_stage_changes(frameRange)

                    analyzer.get_transition_matrix_exp_combat()
                    analyzer.get_stats_bitrate_on_exp_combat()
                    analyzer.get_stats_bitrate_var_exp_combat()
                

if __name__ == "__main__":

    build_dataset()
        
