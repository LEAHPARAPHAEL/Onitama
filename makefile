## For remote work

JOB ?= se_net_6_96_no_aug
TIME_LIMIT ?= 31:00:00

generate_job_script:
	python remote/generate_script.py --job ${JOB} --time ${TIME_LIMIT}

launch_job:
	sbatch remote/scripts/${JOB}.sh

check_job:
	squeue

view_job_output:
	cat logs/${JOB}.out

play:
	python -m gui.gui

## TODO
# train le no_aug
# comparaison temps sur la version default
# comparaison de deux modeles l'un entrainé sur toutes les cartes l'autre sur la moitié
# etre clair dans les configs
# dans la config le modele ait un nom reconnaissable
# nombre de total positions? - "shallow"
