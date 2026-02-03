## For remote work

CURRENT_RUN := template

launch_job:
	sbatch remote/scripts/${CURRENT_RUN}.sh

check_job:
	squeue

view_job_output:
	cat logs/${CURRENT_RUN}.out
