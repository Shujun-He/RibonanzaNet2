for job in grid_search_jobs/*.sh; do
    sbatch "$job"
    #echo "$job"
done