source /cvmfs/sft.cern.ch/lcg/views/LCG_92/x86_64-centos7-gcc7-opt/setup.sh
echo woohoooooooooo1

cd /afs/cern.ch/work/s/sqasim/workspace_1/CMSSW_10_2_13
echo woohoooooooooo2
eval `scramv1 runtime -sh`
echo woohoooooooooo3
cd /afs/cern.ch/work/s/sqasim/workspace_1/ParticleFlowRegression
echo woohoooooooooo4
python ntuplizer.py --input $1 --outdir $2
echo woohoooooooooo5
