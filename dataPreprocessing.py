import os, shutil


def moveImages():
    for foldername in os.listdir('cats/'):
        for filename in os.listdir('cats/' + foldername):
            if filename.endswith('.jpg'):
                shutil.copy('cats/' + foldername + '/' + filename, 'images/' + filename)


if __name__ == '__main__':
    moveImages()
