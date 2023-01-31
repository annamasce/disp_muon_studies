import pickle
import numpy as np
import hist
from hist import Hist, intervals
import matplotlib.pyplot as plt
import mplhep as hep
from argparse import ArgumentParser
from collections import namedtuple
from muon_analysis import MuonAnalysis
from matplotlib.backends.backend_pdf import PdfPages

# Declaring namedtuple()
Variable = namedtuple("Variable", ["name", "bins", "low_edge", "up_edge", "label"])
vars = [
    Variable("Lxy_gen", 50, 0., 500., r"gen $L_{xy}$ [cm]"),
    Variable("dimuon_pt_gen", 50, 0., 500., r"gen $p_{T}$ [GeV]"),
    Variable("dimuon_deltaR_gen", 50, 0., 2., r"gen $\Delta$R"),
    Variable("dimuon_deltaEta_gen", 50, -2.4, 2.4, r"gen $\Delta\eta$"),
    Variable("dimuon_deltaPhi_gen", 50, -np.pi, np.pi, r"gen $\Delta\phi$")
]

def plot_efficiency(result, plot_dir:str, plot_name:str, datasets:list):
    nice_colours = ["blue", "red", "orange", "green", "violet"]
    plt.style.use(hep.style.ROOT)

    plot_file_name = f"plots_eff_{plot_name}" if len(plot_name) > 0 else "plots_eff"
    with PdfPages(plot_dir + f"{plot_file_name}.pdf") as pdf:
        for var in vars:
            fig = plt.figure(figsize=(10, 8))
            plt.gca().set_prop_cycle(color=nice_colours)
            # ds_list = ['2Mu2J_MH-1000_MFF-20', '2Mu2J_MH-1000_MFF-150']
            for i, ds in enumerate(datasets):
                num = result[f"num_{var.name}"][ds].value
                den = result[f"den_{var.name}"][ds].value
                # c = ROOT.TCanvas("c2", "", 1)
                var_axis = hist.axis.Regular(var.bins, var.low_edge, var.up_edge, name=var.name, label=var.label)
                hist_num = Hist(var_axis, storage="weight")
                hist_den = Hist(var_axis, storage="weight")
                hist_num.fill(num)
                hist_den.fill(den)
                # Computing ratio and uncertainty
                ratio = hist_num.values() / hist_den.values()
                ratio_uncert = intervals.ratio_uncertainty(
                    num=hist_num.values(),
                    denom=hist_den.values(),
                    uncertainty_type="efficiency",
                )
                x_values = hist_num.axes[0].centers

                # Plotting
                plt.errorbar(
                    x_values,
                    ratio,
                    yerr=ratio_uncert,
                    marker="o",
                    linestyle="none",
                    label=ds
                )
                plt.ylim(0., 1.09)
                plt.xlabel(var.label)
                plt.ylabel("Efficiency")
                plt.legend()
            pdf.savefig()


def main():
    parser = ArgumentParser()
    parser.add_argument("tag", help="tag of result object name")
    parser.add_argument("--result_dir", default="/Users/mascella/workspace/disp_muon_studies/Results/",
                        help="folder where result is saved")
    parser.add_argument("--plot_dir", default="/Users/mascella/workspace/disp_muon_studies/Plots/",
                        help="folder where to save plots")
    parser.add_argument("--plot_name", default="",
                        help="name tag added to the plots")
    parser.add_argument("--ds_list", default="", help="list of datasets to analyse", type=str)
    args = parser.parse_args()
    tag = args.tag
    result_dir = args.result_dir

    with open(result_dir + f'result_{tag}.pkl', 'rb') as f:
        result = pickle.load(f)
    if len(args.ds_list) == 0:
        datasets = list(result[MuonAnalysis.get_var_axis_pairs()[0][0]].axes["ds"])
    else:
        datasets = [item for item in args.ds_list.split(',')]

    plot_efficiency(result, args.plot_dir, args.plot_name, datasets)




if __name__ == "__main__":
    main()