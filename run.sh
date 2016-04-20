if [ $# -lt 2 ]
then
    echo "Usage: ./run.sh <Epochs> <BatchSize> [cpu|gpu]"
    exit
fi

epochs=$1
batchsize=$2

if [ $# -lt 3 ] || [ $3 = "cpu" ]
then
    echo "CPU: 16 threads"
    THEANO_FLAGS=mode=FAST_RUN,device=cpu,force_device=True OMP_NUM_THREADS=16 python autoencoder.py $epochs $batchsize
    echo "CPU: 12 threads"
    THEANO_FLAGS=mode=FAST_RUN,device=cpu,force_device=True OMP_NUM_THREADS=12 python autoencoder.py $epochs $batchsize
    echo "CPU: 8 threads"
    THEANO_FLAGS=mode=FAST_RUN,device=cpu,force_device=True OMP_NUM_THREADS=8 python autoencoder.py $epochs $batchsize
    echo "CPU: 4 threads"
    THEANO_FLAGS=mode=FAST_RUN,device=cpu,force_device=True OMP_NUM_THREADS=4 python autoencoder.py $epochs $batchsize
    echo "CPU: 2 threads"
    THEANO_FLAGS=mode=FAST_RUN,device=cpu,force_device=True OMP_NUM_THREADS=2 python autoencoder.py $epochs $batchsize
    echo "CPU: 1 thread"
    THEANO_FLAGS=mode=FAST_RUN,device=cpu,force_device=True OMP_NUM_THREADS=1 python autoencoder.py $epochs $batchsize
fi

if [ $# -lt 3 ] || [ $3 = "gpu" ]
then
    echo "GPU"
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,force_device=True,lib.cnmem=0.75,allow_gc=False,floatX=float32 python autoencoder.py $epochs $batchsize
fi
