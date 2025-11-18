# brain-to-text-model
AIT Budapest Data Science Fall 2025, final project

# KEY FINDING
What makes the setup work is using plank! However, the plank home directory has a cap of 5G. To get around this, I've been putting everything in tmp (like data, extra packages). I think the condition of tmp is that it deletes every so often so my hope is that it doesn't delete often so we can use it.

# Initial change to setup instructinos
I used something called **micromamba** instead of conda because I couldn't determine if conda would work on plank. Installing the miniconda package they have in the instructions takes up too much space in our home directory. Instead, I think I just put all of my stuff for micromomba in tmp/ or it's also possible I was able to put it in my home directory I don't remember.

# First setup.sh
I can't remember whether or not I used their setup. I think I didn't because I am using micromamba so I think I just created the b2txt25 environment with micromamba and then I copy and pasted all of the pip install commands they have in the folder to get everything downloaded.

# Next setup_lm.sh
This one is completely runnable! I modified it for micromamba so all you have to do is `./setup.sh`.

# Data
So the data as we know is 10G so this will not fit in our home directory. I basically put everything in a folder called data on my Desktop and then ran `rsync -avh --progress ~/Desktop/data/ [YOUR ID]@plank.thayer.dartmouth.edu:/tmp/data/`. This will give you progress updates as it's downloading and will put it in tmp/data. I then created a symlink to it in the home directory so it was easily accesible.

# Model Changes
I made some minor adjustments to the models as well so check the `model_training` folder. We just had to change some stuff to make it compatible with a cpu. When I ran the baseline, the RT was pretty quick but if we make more advanced changes, we could brainstorm on how to use a GPU.

# Note abot redis
I think I had some issues with redis and manually installed it but it was easy.

# Finally we are ready
So now you should be ready to run the baseline model. Basically, you have two terminals open with b2txt25 and b2txt25_lm and the redis server running to connect them. I only ran 1gram I don't think 3gram or 5gram are necessary. All the instructions for this should be the same as on the original github I just changed some paths to match my directory. So now the baseline model is runnable!