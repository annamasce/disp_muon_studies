from muon_analysis import MuonAnalysis
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import hist
from matplotlib.backends.backend_pdf import PdfPages
from argparse import ArgumentParser
import numpy as np

def plot_from_arrays(result, plot_dir, plot_name):
    with PdfPages(plot_dir + f"plots_from_arrays_{plot_name}.pdf") as pdf:
        for var in MuonAnalysis.get_array_vars():
            plt.hist(result[f"num_{var}"]["2Mu2J"].value, bins=50)
            plt.xlabel(var)
            pdf.savefig()
            plt.close()

def plot_histos(result, plot_dir, plot_name, datasets:list=[]):
    datasets_loc = [hist.loc(ds) for ds in datasets]
    with PdfPages(plot_dir + f"plots_histos_{plot_name}.pdf") as pdf:
        for var, _ in MuonAnalysis.get_var_axis_pairs():
            print(var)
            histo = result[var][datasets_loc, :]
            sel_norm = "converging_fit" if "dimuon" in var else "gen_matching"
            for i, name in enumerate(datasets):
                # print(result[f"n_ev_{sel_norm}"][name])
                # print(np.sum(histo.view(flow=True)[i]))
                histo.view(flow=True)[i] *= 1/result[f"n_ev_{sel_norm}"][name]
            histo.plot1d()
            plt.legend()
            pdf.savefig()
            plt.close()
        for var, _, _ in MuonAnalysis.get_var_axis_2d():
            print(var)
            for ds in datasets:
                histo = result[var][ds, :, :]
                histo.plot2d(norm=mpl.colors.LogNorm()) #cmap=mpl.colormaps["winter"])
                pdf.savefig()
                plt.close()

def main():
    parser = ArgumentParser()
    parser.add_argument("tag", help="tag of result object name")
    parser.add_argument("--result_dir", default="/Users/mascella/workspace/disp_muon_studies/Results/",
                        help="folder where result is saved")
    parser.add_argument("--ds_list", default="", help="list of datasets to analyse", type=str)
    args = parser.parse_args()
    tag = args.tag
    result_dir = args.result_dir
    with open(result_dir + f"result_{tag}.pkl", "rb") as f:
        result = pickle.load(f)
    if len(args.ds_list) == 0:
        datasets = list(result[MuonAnalysis.get_var_axis_pairs()[0][0]].axes["ds"])
    else:
        datasets = [item for item in args.ds_list.split(',')]
    # plot_from_arrays(result, "/Users/mascella/workspace/disp_muon_studies/Plots/", tag)
    plt.style.use(hep.style.ROOT)
    plot_histos(result, "/Users/mascella/workspace/disp_muon_studies/Plots/", tag, datasets=datasets)

if __name__ == "__main__":
    main()