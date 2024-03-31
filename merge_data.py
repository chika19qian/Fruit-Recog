import shutil,os

files='data/'
for dir in os.listdir(files):
    for file in os.listdir(files+dir):
        try:
            shutil.move('data/'+dir+'/'+file,'data/all/')
        except OSError:
            pass
