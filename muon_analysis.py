from coffea import processor, hist
import numpy as np
import awkward as ak


def delta_r2(v1, v2):
    '''Calculates deltaR squared between two particles v1, v2 whose
    eta and phi methods return arrays
    '''
    dphi = (v1.phi - v2.phi + np.pi) % (2 * np.pi) - np.pi
    deta = v1.eta - v2.eta
    dr2 = dphi ** 2 + deta ** 2
    return dr2


def delta_r(v1, v2):
    '''Calculates deltaR between two particles v1, v2 whose
    eta and phi methods return arrays.

    Note: Prefer delta_r2 for cuts.
    '''
    return np.sqrt(delta_r2(v1, v2))

class MuonAnalysis(processor.ProcessorABC):
    def __init__(self):
        ds_axis = hist.Cat("ds", "Primary dataset")
        acc_dict = {var: hist.Hist("Counts", ds_axis, axis) for var, axis in self.get_var_axis_pairs()}
        for stage in self.get_selections():
            acc_dict[f'n_ev_{stage}'] = processor.defaultdict_accumulator(int)
            # acc_dict[f'sumw_{stage}'] = processor.defaultdict_accumulator(float)
        for var in self.get_array_vars():
            acc_dict[f'num_{var}'] = processor.dict_accumulator()
            acc_dict[f'den_{var}'] = processor.dict_accumulator()
        # acc_dict['sel_array'] = processor.dict_accumulator()
        self._accumulator = processor.dict_accumulator(acc_dict)

    @property
    def accumulator(self):
        return self._accumulator

    @staticmethod
    def get_array_vars():
        return [
            "dxy_gen"
        ]

    @staticmethod
    def get_selections():
        return [
            "two_muons_acc",
            "two_dsa",
            "dsa_selection",
            "gen_matching",
            "lxy_cut",
            "converging_fit",
            "sv_prob_cut"
        ]

    @staticmethod
    def get_var_axis_pairs():

        pt_dsa_axis = hist.Bin("pt_dsa", r"$p_{T}$ [GeV]", 20, 0., 600)
        pt_err_axis = hist.Bin("pt_err", r"$p_{T}$ error/$p_{T}$", 20, 0., 1.2)
        nhit_axis = hist.Bin('nhit', r'N(hits)', 71, -0.5, 70.5)
        chi2ndof_axis = hist.Bin('chi2ndof', r'$\chi^2$/ndof', 20, 0., 5.)

        v_a_pairs = [
            ('pt_dsa_1', pt_dsa_axis),
            ('pt_dsa_2', pt_dsa_axis),
            ('err_div_pt_dsa_1', pt_err_axis),
            ('err_div_pt_dsa_2', pt_err_axis),
            ('nhits_dsa_1', nhit_axis),
            ('nhits_dsa_2', nhit_axis),
            ('chi2ndof_dsa_1', chi2ndof_axis),
            ('chi2ndof_dsa_2', chi2ndof_axis),
        ]

        return v_a_pairs

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()
        ds = events.metadata["dataset"]

        if "HNL" in ds:
            events["ll_mother"] = events.GenPart[(events.GenPart.pdgId == 9900012) & (events.GenPart.statusFlags >= 2 ** 13)]
            events["ctau"] = ak.max(events.LHEPart.ctau, axis=-1)
        # To select gen muons from LL particle we need to first require distinctParentIdxG != -1 to avoide None pdgId
        gen_muons = events.GenPart[(np.abs(events.GenPart.pdgId) == 13) & (events.GenPart.status == 1) & (
                events.GenPart.distinctParentIdxG != -1)]
        events["gen_muon"] = gen_muons[gen_muons.distinctParent.pdgId > 1e4]

        # Select events with 2 gen muons in acceptance
        cut = (events.gen_muon.pt > 3) & (np.abs(events.gen_muon.eta) < 2.4)
        events["gen_muon"] = events.gen_muon[cut]
        events = events[ak.num(events.gen_muon.pt, axis=-1) == 2]
        print("Events with 2 gen muons in acceptance:", len(events))
        out["n_ev_two_muons_acc"][ds] += len(events)

        # save gen_dxy info
        if "HNL" in ds:
            # Currently SV info is not saved for gen muons in HNL samples, so gen_dxy is derived from ctau
            events["gen_dxy"] = events.ctau / 10 * ak.flatten(events.ll_mother.pt, axis=-1) / ak.flatten(
                events.ll_mother.mass, axis=-1)  # cm
        else:
            gen_vx = events.gen_muon[:, 0].vx
            gen_vy = events.gen_muon[:, 0].vy
            events["gen_dxy"] = np.sqrt(gen_vx ** 2 + gen_vy ** 2)
        print(events.gen_dxy)
        self.process_ds(events, out, ds)
        if "2Mu2J" in ds:
            self.process_ds(events, out, ds + " high Lxy", mode="high_Lxy")
            self.process_ds(events, out, ds + " low Lxy", mode="low_Lxy")

        return out

    def process_ds(self, events, out, ds, mode=""):
        do_dsa_sel = True
        for var in self.get_array_vars():
            out[f'num_{var}'][ds] = processor.column_accumulator(np.array([]))
            out[f'den_{var}'][ds] = processor.column_accumulator(np.array([]))
        if mode == "high_Lxy":
            print("Here I do stuff for high Lxy")
            events = events[events.gen_dxy > 330]
        elif mode == "low_Lxy":
            print("Here I do stuff for low Lxy")
            # events = events[(events.gen_dxy > 200) & (events.gen_dxy <= 330)]
            events = events[events.gen_dxy <= 330]
        else:
            print("Here I do stuff for whole ds")
        out["n_ev_lxy_cut"][ds] += len(events)
        events = self.process_den(events, out, ds, do_sel_dsa=do_dsa_sel)
        events, dimuons = self.process_num(events, out, ds, do_sel=do_dsa_sel)


    def process_den(self, events, out, ds, do_sel_dsa=True):
        print("Total den events:", len(events))
        # Select events with at least 2 DSA muons
        events = events[ak.num(events.DSAMuon, axis=-1) >= 2]
        print("Den events with 2 DSA muons:", len(events))
        out["n_ev_two_dsa"][ds] += len(events)
        # Take all possible pairs of DSA muons
        l1_idx, l2_idx = ak.unzip(ak.argcombinations(events.DSAMuon, 2))
        dsa_1 = events.DSAMuon[l1_idx]
        dsa_2 = events.DSAMuon[l2_idx]
        if do_sel_dsa:
            dimuon_cut = self.dsa_selection(dsa_1) & self.dsa_selection(dsa_2)
            event_cut = (ak.sum(dimuon_cut, axis=-1) > 0)
            dsa_1 = dsa_1[dimuon_cut][event_cut]
            dsa_2 = dsa_2[dimuon_cut][event_cut]
            events = events[event_cut]
            print("Den events after DSA selection:", len(events))
            out["n_ev_dsa_selection"][ds] += len(events)

        # Require unique matching of each DSA pair with the gen muons
        gen_muons = events.gen_muon
        cut = self.unique_matching_selection(dsa_1, dsa_2, gen_muons)
        event_cut = (ak.sum(cut, axis=-1) > 0)
        events = events[event_cut]
        dsa_1 = dsa_1[cut][event_cut]
        dsa_2 = dsa_2[cut][event_cut]
        print("Den events after DSA matching with gen muons:", len(events))
        out["n_ev_gen_matching"][ds] += len(events)
        self.fill_histos(events, out, ds, dsa_1, dsa_2)
        self.fill_arrays(events, out, ds, "den")
        return events

    def process_num(self, events, out, ds, do_sel=True):
        # Require at least a DSA pair
        events = events[ak.num(events.DiDSAMuon) >= 1]
        dimuons = events.DiDSAMuon[ak.num(events.DiDSAMuon) >= 1]
        # print("Num events with a 2-DSA fit:", len(events))

        if do_sel:
            cut = self.dsa_selection(events.DSAMuon[dimuons.l1Idx]) & self.dsa_selection(events.DSAMuon[dimuons.l2Idx])
            dimuons = dimuons[cut]
            events = events[ak.num(dimuons) > 0]
            dimuons = dimuons[ak.num(dimuons) > 0]
            # print("Num events with a 2-DSA fit after DSA selection:", len(events))
        print("Num events with a 2-DSA fit:", len(events))

        # Require unique matching of each DSA in a fitted pair with a gen hnl muon
        dsa_1 = events.DSAMuon[dimuons.l1Idx]
        dsa_2 = events.DSAMuon[dimuons.l2Idx]
        gen_muons = events.gen_muon
        cut = self.unique_matching_selection(dsa_1, dsa_2, gen_muons)

        dimuons = dimuons[cut]
        events = events[ak.num(dimuons) > 0]
        dimuons = dimuons[ak.num(dimuons) > 0]
        out["n_ev_converging_fit"][ds] += len(events)
        print("Num events after DSA matching with gen muons:", len(events))

        # Require good vertex prob
        cut = (dimuons.ndof > 0) & (dimuons.svprob >= 0.001)
        dimuons = dimuons[cut]
        events = events[ak.num(dimuons) > 0]
        dimuons = dimuons[ak.num(dimuons) > 0]
        out["n_ev_sv_prob_cut"][ds] += len(events)

        self.fill_arrays(events, out, ds, "num")

        return events, dimuons

    def fill_arrays(self, events, out, ds, stage):
        out[f"{stage}_dxy_gen"][ds] += processor.column_accumulator(ak.to_numpy(events.gen_dxy, False))

    def dsa_selection(self, dsa):
        return (dsa.pt > 5.) \
               & (dsa.pt_error / dsa.pt < 1.) \
               & (dsa.n_valid_hits > 15) \
               & (dsa.chi2 / dsa.ndof < 2.5)

    def unique_matching_selection(self, dsa_1, dsa_2, other_muons, dr_cut=0.7):
        """
        Perform unique matching between a pair of DSA muons and a set of other muons

        :param dsa_1: first muon of DSA pair
        :param dsa_2: second muon of DSA pair
        :param other_muons: muons to be uniquely matched to the dsa
        :return: selection to be applied to the dsa muon arrays
        """

        p_dsa_1, p_hnl_1 = ak.unzip(ak.cartesian([dsa_1, other_muons], nested=True))
        dr_1_pass = delta_r(p_dsa_1, p_hnl_1) < dr_cut

        p_dsa_2, p_hnl_2 = ak.unzip(ak.cartesian([dsa_2, other_muons], nested=True))
        dr_2_pass = delta_r(p_dsa_2, p_hnl_2) < dr_cut

        any_pass = dr_1_pass | dr_2_pass
        cut = (ak.sum(any_pass, axis=-1) >= 2) & (ak.sum(dr_1_pass, axis=-1) >= 1) & (
                ak.sum(dr_2_pass, axis=-1) >= 1)
        return cut

    def fill_histos(self, events, out, ds, dsa_1, dsa_2):
        dsa_1 = dsa_1[:, 0]
        dsa_2 = dsa_2[:, 0]
        out["pt_dsa_1"].fill(ds=ds, pt_dsa=dsa_1.pt)
        out["pt_dsa_2"].fill(ds=ds, pt_dsa=dsa_2.pt)
        out["err_div_pt_dsa_1"].fill(ds=ds, pt_err=(dsa_1.pt_error / dsa_1.pt))
        out["err_div_pt_dsa_2"].fill(ds=ds, pt_err=(dsa_2.pt_error / dsa_2.pt))
        out["nhits_dsa_1"].fill(ds=ds, nhit=dsa_1.n_valid_hits)
        out["nhits_dsa_2"].fill(ds=ds, nhit=dsa_2.n_valid_hits)
        out["chi2ndof_dsa_1"].fill(ds=ds, chi2ndof=dsa_1.chi2/dsa_1.ndof)
        out["chi2ndof_dsa_2"].fill(ds=ds, chi2ndof=dsa_2.chi2/dsa_2.ndof)


    def postprocess(self, accumulator):
        return accumulator