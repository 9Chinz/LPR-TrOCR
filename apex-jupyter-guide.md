# How to run steps in the Jupyter guide

## ❗ important info ❗
- dip_M is the username that login
- prism-N is the name of the machine that jupyter get gpu
- use command ```squeue -u $USER``` to check if the job is running

## 📝 edit your example-run-jupyter.sh
can edit job name in example-run-jupyter.sh at part 
- ```#SBATCH -A <your name>```  change to your project name
- ```#SBATCH --gres=gpu:<N>``` can change to gpu:N if you want more gpu
- ```#SBATCH --job-name=<your job name>``` change to your job name 
- ```source activate <CONDA ENV>``` change to your conda environment name

## 🅰️ on apex

- run ```sbatch example-run-jupyter.sh```
- look for token in file err_%j.txt
    - token look something like this ```http://prism-1:6789/lab?token=732bc7b09482ac...```
    - copy only token part ```732bc7b09482ac...```
- run ```squeue -u $USER``` to check if the job is running for example like 
```sh
JOBID   PARTITION      NAME    USER  ST      TIME  NODES  NODELIST(REASON)
 1234       batch   jupyter   dip_M   R   0:01:10      1  prism-1
```

## 🧑‍💻 on local terminal

- run ```ssh -NL localhost:6789:prism-N:6789 dip_M@apex-logi.cmkl.ac.th```

- open local browser and go to ```localhost:6789```

- paste token and login