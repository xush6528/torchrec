for i in {1..23}
do
    echo "download day_$i now"
    cd /data/criteo/ && sudo curl -O https://storage.googleapis.com/criteo-cail-datasets/day_$i.gz
    cd /data/criteo/ && sudo gzip -d day_$i.gz
done 