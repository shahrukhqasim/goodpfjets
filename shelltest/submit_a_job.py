import os
import argparse
import numpy as np
import glob
import os.path
import time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()    #Python tool that makes it easy to create an user-friendly command-line interface
    parser.add_argument('-o','--output', type=str, help="output  EOS directory", required=True)
    parser.add_argument('-if','--inputfile', type=str, help="input root file", required=True)
    parser.add_argument('-od','--outputdir', type=str, help="output directory to produce h5 file to", required=True)
    parser.add_argument('-q', '--queue', type=str, default = "1nw", help="LSFBATCH queue name")
#    parser.add_argument('-f', '--feature', type=str, default = "0", help="feature of the input daatset to be analyzed")
#     parser.add_argument('-t', '--toys', type=str, default = "1000", help="number of toys to be processed")
    args = parser.parse_args()

    mydir = args.output+"/"

    os.system("mkdir %s" %mydir)
    label = args.output.split("/")[-1]+'_5D_'+str(time.time())
    os.system("mkdir %s" %label)

    for i in range(1):
        joblabel = str(i)#fileIN.split("/")[-1].replace(".h5","")
        if not os.path.isfile("%s/%s_t.txt" %(mydir, joblabel)):
            # src file
            script_src = open("%s/%s.src" %(label, joblabel) , 'w')
            script_src.write("#!/bin/bash\n")
            script_src.write("echo woohoooooooooo1\n")
            script_src.write("export opath=`pwd`\n")
            script_src.write("cd /afs/cern.ch/work/s/sqasim/workspace_1/CMSSW_10_2_13\n")
            script_src.write("echo woohoooooooooo2\n")
            script_src.write("eval `scramv1 runtime -sh`\n")
            script_src.write("echo woohoooooooooo3\n")
            script_src.write("cd /afs/cern.ch/work/s/sqasim/workspace_1/ParticleFlowRegression\n")
            script_src.write("echo woohoooooooooo4\n")
            script_src.write("echo $opath\n")
            script_src.write("echo $opath\n")

            output_file_name = os.path.basename(args.inputfile).replace('.root', '.h5')
            print(output_file_name)
            0/0

            script_src.write("python ntuplizer.py --input %s --outdir $opath" %(args.inputfile)+'\n')
            script_src.write("cp $opath/%s %s/\n"%(output_file_name, args.outputdir))
            script_src.write("echo woohoooooooooo5"+'\n')
            script_src.close()
            os.system("chmod a+x %s/%s.src" %(label, joblabel))
#            os.system("bsub -q %s -o %s/%s.log -J %s_%s < %s/%s.src" %(args.queue, label, joblabel, label, joblabel, label, joblabel))
            # condor file
            script_condor = open("%s/%s.condor" %(label, joblabel) , 'w')
            script_condor.write("executable = %s/%s.src\n" %(label, joblabel))
            script_condor.write("universe = vanilla\n")
            script_condor.write("output = %s/%s.out\n" %(label, joblabel))
            script_condor.write("error =  %s/%s.err\n" %(label, joblabel))
            script_condor.write("log = %s/%s.log\n" %(label, joblabel))
            script_condor.write("+MaxRuntime = 72000\n")
            script_condor.write('requirements = (OpSysAndVer =?= "CentOS7")\n')
            script_condor.write("queue\n")

            script_condor.close()
            # condor file submission
            os.system("condor_submit %s/%s.condor" %(label, joblabel))

#            script_condor.write('requirements = (OpSysAndVer =?= "SLCern6")\n') #requirements = (OpSysAndVer =?= "CentOS7")