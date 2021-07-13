import os

path = "./data/imgs/from_video/Biceps_curl/img_0-99"
#dirs = os.listdir(path)
#for file in os.listdir(path):
#    fullpath = os.path.join(path,file)
#    if os.path.isfile(fullpath):
#        print("file",file)
#    elif os.path.isdir(fullpath):
#        print("dir",file)

for r,d,f in os.walk(path):
    print("path",r)
    print("dir",d)
    print("file",f[0])
