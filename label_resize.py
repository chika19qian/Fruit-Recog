import os
import random
from PIL import Image
root = 'data'

# for dir in os.listdir(root):
#     print(dir)
#     for files in os.listdir(root+dir):
#         try:
#             img=Image.open(root+dir+'/'+files)
#             img=img.resize((224,224))
#             print(root+dir+'/'+files)
#             img.save(root+dir+'/'+files)
#         except:
#             os.remove(root+dir+'/'+files)

counter=0
for dir in os.listdir(root):
    print(dir)
    for files in os.listdir(root+dir):
        os.rename(root+dir+'/'+str(files),root+dir+'/'+str(counter)+'_'+str(files))
    counter+=1