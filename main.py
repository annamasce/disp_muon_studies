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

# old samples
samples = {
    "HNL1": ["/Users/mascella/cernbox/disp_muons/disp_muons_HNL1_svFix.root"],
    "2Mu2J": ["/Users/mascella/cernbox/disp_muons/disp_muons_2Mu2J_svFix.root"]
}

# new samples with SV fix
# samples = {
#     "HNL1": ["/Users/mascella/cernbox/disp_muons_HNL1.root"],
#     "2Mu2J": ["/Users/mascella/cernbox/disp_muons_2Mu2J.root"]
# }

result = processor.run_uproot_job(
    samples,
    "Events",
    MuonAnalysis(),
    processor.iterative_executor,
    {"schema": NanoAODSchema},
)

hist.plot1d(result["pt_dsa_1"][re.compile("2Mu2J*")])
plt.show()
plt.hist(result["den_dimuon_deltaR"]["HNL1"].value, range=[0, 2], bins=20)
plt.hist(result["num_dimuon_deltaR"]["HNL1"].value, range=[0, 2], bins=20)
plt.show()
with open("./Results/" + f'result_{args.tag}.pkl', 'wb') as f:
    pickle.dump(result, f)
