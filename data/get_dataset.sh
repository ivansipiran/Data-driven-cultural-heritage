SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

cd $SCRIPTPATH/
wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip --no-check-certificate
mkdir -p shapenet_part
mv shapenetcore_partanno_segmentation_benchmark_v0.zip shapenet_part/
cd $SCRIPTPATH/shapenet_part/
unzip shapenetcore_partanno_segmentation_benchmark_v0.zip
rm shapenetcore_partanno_segmentation_benchmark_v0.zip
cd shapenetcore_partanno_segmentation_benchmark_v0/
mv * ..
cd ..
rm -r shapenetcore_partanno_segmentation_benchmark_v0/
cd $SCRIPTPATH
