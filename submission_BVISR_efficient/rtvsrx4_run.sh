
outputFile='./RTVSRx4/'
inputFile='./dataset/4K-ValidationSet-LR/ValidationSet/bitstreams/'
ffmpegpath='' #replace with your ffmpeg path

if [ ! -x "$outputFile" ];
then
mkdir -p "$outputFile"
fi

for fileName in ${inputFile}*.mp4
do
python ./vsr_runx4.py --input=$fileName --output=$outputFile --ffmpeg $ffmpegpath

done