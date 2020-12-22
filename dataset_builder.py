import os
import subprocess
import numpy as np
from PIL import Image

for id_map in range(1,33):       # for each one of 32 maps

    path_exp = "./experiment/map_id_%i" % id_map
    
    while os.path.exists(path_exp):
        
        experiments = os.listdir(path_exp)
        
        for exp in experiments:

            path_dataset = path_exp + "/" + exp + "/dataset"
            
            j = 0
            while os.path.isfile("%s/chunk%i.npz" %(path_dataset, j)):

                path_chunk  = path_dataset + "/chunk%i" % j
                path_frames = path_chunk + "/frames"
                
                if not os.path.exists(path_chunk):
                    os.mkdir(path_chunk)
                if not os.path.exists(path_frames):
                    os.mkdir(path_frames)

                data = np.load("%s.npz" % path_chunk)
                
                k = 0
                while data.get('frame%i' %k) is not None: 
                    arr = data.get('frame%i' %k)
                    arr = np.moveaxis(arr , 0, -1) 
                    im = Image.fromarray(arr)
                    frame = "frame{0:04d}.png".format(k)
                    im.save("%s/%s" % (path_frames, frame))
                    print(frame) 
                    k += 1

                print("Chunk%i has been recorded!" % j)

                subprocess.Popen("ffmpeg -r 10 -f image2 -i %s -vcodec libx264 -crf 1 ../chunk%i.mp4" % ("frame%04d.png", j), shell=True, cwd=path_frames).wait()
                print("Chunk%i.mp4 has been recorded!" % j)
                subprocess.Popen("plotbitrate -o chunk%i.svg chunk%i.mp4" % (j, j), shell=True, cwd=path_chunk).wait()
                print("Chunk%i.svg has been created!" % j)

                j += 1

        break

'''
Alternative approach:
ffmpeg -r 10 -f image2 -i frame%04d.png -vcodec libx264 -crf 1 -pix_fmt yuv420p -vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2" chunk0.mp4
'''