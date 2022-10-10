import matplotlib.pyplot as plt
import pickle
import argparse
import re
from coffea import processor, hist
from coffea.nanoevents import NanoAODSchema
from muon_analysis import MuonAnalysis

parser = argparse.ArgumentParser()
parser.add_argument('tag', help='tag that will be added to produced pkl files')
parser.add_argument('--result_dir', default='/Users/mascella/workspace/disp_muon_studies/Results/', help='folder where to save results')
args = parser.parse_args()

tag = args.tag

samples = {
    "HNL1": ["/Users/mascella/cernbox/Analysis_files/HeavyNeutrino_trilepton_M-1_V-0_0949736805647_mu_massiveAndCKM_LO/merged_0.root"],
    "2Mu2J": ["/Users/mascella/cernbox_shared_with_you/test/displaced_dimuon.root"]
}

result = processor.run_uproot_job(
    samples,
    "Events",
    MuonAnalysis(),
    processor.iterative_executor,
    {"schema": NanoAODSchema},
)

hist.plot1d(result["pt_dsa_1"][re.compile("2Mu2J*")])
plt.show()
plt.hist(result["den_dxy_gen"]["2Mu2J"].value, range=[0, 500], bins=50)
plt.hist(result["num_dxy_gen"]["2Mu2J"].value, range=[0, 500], bins=50)
plt.show()
with open("./Results/" + f'result_{args.tag}.pkl', 'wb') as f:
    pickle.dump(result, f)
