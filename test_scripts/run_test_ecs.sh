#!/bin/sh
#$ -S /bin/sh
#$ -wd /vol/grid-solar/sgeusers/sarathkash
#$ -o /vol/grid-solar/sgeusers/sarathkash/system_out/$JOB_NAME.o$JOB_ID
#$ -e /vol/grid-solar/sgeusers/sarathkash/system_out/$JOB_NAME.e$JOB_ID



# Move into job-local temp dir, as recommended by ECS Grid
if [ -d /local/tmp/sarathkash/$JOB_ID ]; then
    cd /local/tmp/sarathkash/$JOB_ID
else
    echo "Job temp directory not found"
    echo "Contents of /local/tmp:"
    ls -la /local/tmp
    echo "Contents of /local/tmp/sarathkash:"
    ls -la /local/tmp/sarathkash
    exit 1
fi

# Copy required input files into local temp
cp /vol/grid-solar/sgeusers/sarathkash/test_ecs.py .

# Only copy data if your script actually uses it
if [ -d /vol/grid-solar/sgeusers/sarathkash/data ]; then
    cp -r /vol/grid-solar/sgeusers/sarathkash/data .
fi

echo "Host: $(hostname)"
command -v nvidia-smi || echo "nvidia-smi not found"
nvidia-smi -L || true

# Run the Python script
/vol/grid-solar/sgeusers/sarathkash/venvs/myenv/bin/python test_ecs.py --r "$1"
if [ $? -ne 0 ]; then
    echo "Python script failed."
    exit 1
else
    echo "python test_ecs.py --r $1 completed."
fi

# Copy only the useful outputs back to shared storage
mkdir -p /vol/grid-solar/sgeusers/sarathkash/results/$JOB_ID
cp -r results /vol/grid-solar/sgeusers/sarathkash/results/$JOB_ID/

echo "Job done"