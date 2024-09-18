
outputFile='./RTVSRx3/'
inputFile='./dataset/1080p-ValidationSet-LR/ValidationSet-1080p/bitstreams/'
ffmpegpath='' #replace with your ffmpeg path

if [ ! -x "$outputFile" ];
then
mkdir -p "$outputFile"
fi

for fileName in ${inputFile}*.mp4
do
python ./vsr_runx3.py --input=$fileName --output=$outputFile  --ffmpeg $ffmpegpath

done