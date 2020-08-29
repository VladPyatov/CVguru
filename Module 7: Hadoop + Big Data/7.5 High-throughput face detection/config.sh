# set the base directory
BASE=$(pwd)

# update the PATH to include the 'pyimagesearch' library
PYTHONPATH="${PYTHONPATH}:${BASE}"
export PYTHONPATH

# configure service path
export HADOOP_HOME="/home/guru/services/hadoop"

