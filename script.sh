# !/bin/csh
#PBS -N myjob
#PBS -q course
#PBS -j oe
#PBS -l nodes=12:ppn=8

cd /home/rahuljr/DS295/parallel-online-knn/

for j in 10 12
do
    mpiexec -np $j python3 -m knn.experiments2
done
