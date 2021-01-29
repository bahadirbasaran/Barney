import os
import sys
import subprocess
import numpy as np
from PIL import Image
from itertools import product
from plot_video_bitrate import PlotVideoBitrate

FPS = 35

class SceneLabels():
    EXPLORATION = 0
    COMBAT      = 1
    MENU        = 2
    CONSOLE     = 3
    SCOREBOARD  = 4
    CUTSCENE    = 5

def build_dataset(categorical=False, videoOnly=False):

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

                    path_chunk  = path_dataset + "/chunk%i" % j
                    path_frames = path_chunk + "/frames"
                    
                    if os.path.exists(path_chunk):
                        break
                    else:
                        os.mkdir(path_chunk)
                        os.mkdir(path_frames)

                    data = np.load("%s.npz" % path_chunk)
                    
                    n_frame = 0
                    while data.get('frame%i' %n_frame) is not None: 

                        arr = data.get('frame%i' %n_frame)
                        arr = np.moveaxis(arr , 0, -1) 
                        im = Image.fromarray(arr)
                        frame = "frame{0:04d}.png".format(n_frame)
                        im.save("%s/%s" % (path_frames, frame))
                        print(frame) 
                        n_frame += 1

                    print("%s has been created!" % path_chunk)

                    # 'fps' is calculated based on time difference between the first and last labels in the chunk. 
                    # t_firstFrame_ms = data.get('label0')[1]
                    # t_lastFrame_ms  = data.get('label%i' % (n_frame - 1))[1]
                    # t_elapsed_s = (t_lastFrame_ms - t_firstFrame_ms) / 1000
                    # fps = int(round(n_frame / t_elapsed_s))
                
                    subprocess.Popen("ffmpeg -r %i -f image2 -i %s -vcodec libx264 -crf 1 -loglevel quiet ../chunk%i_%ifps.mp4" % (FPS, "frame%04d.png", j, FPS), shell=True, cwd=path_frames).wait()
                    print("Chunk%i.mp4 has been recorded!" % j)
                    subprocess.Popen("plotbitrate -o chunk%i_%ifps.svg chunk%i_%ifps.mp4" % (j, FPS, j, FPS), shell=True, cwd=path_chunk).wait()
                    print("Chunk%i_%ifps.svg has been created!" % (j, FPS))
                    if videoOnly:
                        print("Frames inside [%s] are being destroyed..." % path_frames)
                        subprocess.Popen("rm -f *.png", shell=True, cwd=path_frames).wait()

                    if categorical:
                        
                        path_exploration = path_frames + "/exploration"
                        path_combat      = path_frames + "/combat"
                        path_menu        = path_frames + "/menu"
                        path_console     = path_frames + "/console"
                        path_scoreboard  = path_frames + "/scoreboard"
                        videoPaths = []

                        n_label = n_exploration = n_combat = n_menu = n_console = n_scoreboard = 0
                        while data.get('label%i' %n_label):

                            arr = data.get('frame%i' %n_label)
                            arr = np.moveaxis(arr , 0, -1) 
                            im = Image.fromarray(arr)
                            
                            stageCategory = data.get('label%i' %n_label)[0]

                            if stageCategory == SceneLabels.EXPLORATION:

                                if not os.path.exists(path_exploration):
                                    os.mkdir(path_exploration)
                                            
                                frame = "frame{0:04d}.png".format(n_exploration)
                                im.save("%s/%s" % (path_exploration, frame))
                                print(path_exploration, frame)
                                n_exploration += 1

                            elif stageCategory == SceneLabels.COMBAT:

                                if not os.path.exists(path_combat):
                                    os.mkdir(path_combat)
                                            
                                frame = "frame{0:04d}.png".format(n_combat)
                                im.save("%s/%s" % (path_combat, frame))
                                print(path_combat, frame)
                                n_combat += 1

                            elif stageCategory == SceneLabels.MENU:

                                if not os.path.exists(path_menu):
                                    os.mkdir(path_menu)
                                            
                                frame = "frame{0:04d}.png".format(n_menu)
                                im.save("%s/%s" % (path_menu, frame))
                                print(path_menu, frame)
                                n_menu += 1
                            
                            elif stageCategory == SceneLabels.CONSOLE:

                                if not os.path.exists(path_console):
                                    os.mkdir(path_console)
                                            
                                frame = "frame{0:04d}.png".format(n_console)
                                im.save("%s/%s" % (path_console, frame))
                                print(path_console, frame)
                                n_console += 1

                            elif stageCategory == SceneLabels.SCOREBOARD:

                                if not os.path.exists(path_scoreboard):
                                    os.mkdir(path_scoreboard)
                                            
                                frame = "frame{0:04d}.png".format(n_scoreboard)
                                im.save("%s/%s" % (path_scoreboard, frame))
                                print(path_scoreboard, frame)
                                n_scoreboard += 1

                            n_label += 1

                        for path in [path_exploration, path_combat, path_menu, path_console, path_scoreboard]:
                            if os.path.exists(path):
                                subprocess.Popen("ffmpeg -r %i -f image2 -i %s -vcodec libx264 -crf 1 -loglevel quiet chunk%i_%s_%ifps.mp4" 
                                    % (FPS, "frame%04d.png", j, path.split('/')[-1], FPS), shell=True, cwd=path).wait()
                                #subprocess.Popen("plotbitrate -o chunk%i.svg chunk%i_%ifps.mp4" % (j, j, FPS), shell=True, cwd=path).wait()
                                videoPaths.append(path + "/chunk%i_%s_%ifps.mp4" % (j, path.split('/')[-1], FPS))
                                if videoOnly:
                                    print("Frames inside [%s] are being destroyed..." % path)
                                    subprocess.Popen("rm -f *.png", shell=True, cwd=path).wait()
                        
                        for window, shift in list(product([25, 30, 35, 40, 45], [None, 1, 2, 3, 4, 5])):
                            if shift is None:
                                objPlot = PlotVideoBitrate(videoPaths, path_chunk + "/chunk%s_w%i.png" % (j, window), window, shift)
                            else:
                                objPlot = PlotVideoBitrate(videoPaths, path_chunk + "/chunk%s_w%i_s%i.png" % (j, window, shift), window, shift)
                            objPlot.generate_CSV()
                            objPlot.read_CSV()
                            objPlot.plot()
                    
                    j += 1

            break

if __name__ == "__main__":

    if(all(arg in sys.argv for arg in ["--categorical", "--videoOnly"])):
        build_dataset(categorical=True, videoOnly=True)
    elif("--categorical" in sys.argv):
        build_dataset(categorical=True)
    elif("--videoOnly" in sys.argv):
        build_dataset(videoOnly=True)
    else:
        build_dataset()
        
