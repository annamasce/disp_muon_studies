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

# new samples
samples = {
    "HNL1": ["/Users/mascella/cernbox/disp_muons/disp_muons_HNL1_svFix.root"],
    "2Mu2J": ["/Users/mascella/cernbox/disp_muons/nanoTuple_230120_MH-1000_MFF-20_CTau-200mm.root"]
}

# # old samples - no SV fix
# samples = {
#     "HNL1": ["/Users/mascella/cernbox/disp_muons/disp_muons_HNL1_noSvFix.root"],
#     "2Mu2J": ["/Users/mascella/cernbox_shared_with_you/test/displaced_dimuon.root"]
# }

result = processor.run_uproot_job(
    samples,
    "Events",
    MuonAnalysis(),
    processor.iterative_executor,
    {"schema": NanoAODSchema},
)

hist.plot1d(result["lxy_significance"][re.compile("2Mu2J*")])
plt.show()
plt.hist(result["den_dimuon_pt"]["HNL1"].value, range=[0, 100], bins=100)
plt.hist(result["num_dimuon_pt"]["HNL1"].value, range=[0, 100], bins=100)
plt.show()
with open("./Results/" + f'result_{args.tag}.pkl', 'wb') as f:
    pickle.dump(result, f)
