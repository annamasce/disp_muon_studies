from coffea import processor
import hist
from hist import Hist
import awkward as ak
from helpers import *

class MuonAnalysis(processor.ProcessorABC):
    def __init__(self):
        ds_axis = hist.axis.StrCategory([], growth=True, name="ds", label="Primary dataset")
        acc_dict_hist1d = {var: Hist(ds_axis, axis, storage="weight") for var, axis in self.get_var_axis_pairs()}
        acc_dict_hist2d = {var: Hist(ds_axis, axis_x, axis_y, storage="weight") for var, axis_x, axis_y in self.get_var_axis_2d()}
        acc_dict = {**acc_dict_hist1d, **acc_dict_hist2d}
        for stage in self.get_selections():
            acc_dict[f'n_ev_{stage}'] = processor.defaultdict_accumulator(int)
            # acc_dict[f'sumw_{stage}'] = processor.defaultdict_accumulator(float)
        for var in self.get_array_vars():
            acc_dict[f'num_{var}'] = processor.dict_accumulator()
            acc_dict[f'den_{var}'] = processor.dict_accumulator()
        self._accumulator = acc_dict

    @property
    def accumulator(self):
        return self._accumulator

    @staticmethod
    def get_array_vars():
        return [
            "Lxy_gen",
            "dimuon_deltaR_gen",
            "dimuon_deltaEta_gen",
            "dimuon_deltaPhi_gen",
            "dimuon_pt_gen",
            "dimuon_deltaR",
            "dimuon_deltaEta",
            "dimuon_deltaPhi",
            "dimuon_pt",
            "dxy_dsapair",
            "dxy_dsapair_error",
            "dxyz_dsapair",
            "dz_dsapair"
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

        pt_dsa_axis = hist.axis.Regular(20, 0.0, 600.0, name="pt_dsa", label=r"$p_{T}$ [GeV]")
        pt_err_axis = hist.axis.Regular(20, 0.0, 1.2, name="pt_err", label=r"$p_{T}$ error/$p_{T}$")
        nhit_axis = hist.axis.Regular(71, -0.5, 70.5, name="nhit", label=r"N(hits)")
        chi2ndof_axis = hist.axis.Regular(20, 0.0, 5.0, name="chi2ndof", label=r"$\chi^2$/ndof")
        lxy_axis = hist.axis.Regular(20, 0.0, 600.0, name="lxy", label=r"reco $L_{xy}$ [cm]")
        lxy_res_axis = hist.axis.Regular(20, -50.0, 50.0, name="lxy_res",
                                          label=r"$reco L_{xy} - gen L_{xy}$ [cm]")
        lxy_pull_axis = hist.axis.Regular(20, -10.0, 10.0, name="lxy_pull", label=r"$(reco L_{xy} - gen L_{xy})/\sigma_{L_{xy}}$")
        lxy_significance_axis = hist.axis.Regular(100, 0.0, 100.0, name="lxy_significance", label=r"$L_{xy}/\sigma_{L_{xy}}$")

        v_a_pairs = [
            ('pt_dsa_1', pt_dsa_axis),
            ('pt_dsa_2', pt_dsa_axis),
            ('err_div_pt_dsa_1', pt_err_axis),
            ('err_div_pt_dsa_2', pt_err_axis),
            ('nhits_dsa_1', nhit_axis),
            ('nhits_dsa_2', nhit_axis),
            ('chi2ndof_dsa_1', chi2ndof_axis),
            ('chi2ndof_dsa_2', chi2ndof_axis),
            ('dimuon_lxy', lxy_axis),
            ('dimuon_lxy_res', lxy_res_axis),
            ('dimuon_lxy_pull', lxy_pull_axis),
            ('dimuon_lxy_significance', lxy_significance_axis),

        ]

        return v_a_pairs

    @staticmethod
    def get_var_axis_2d():
        lxy_axis = hist.axis.Regular(50, 0.0, 600.0, name="lxy", label=r"reco $L_{xy}$ [cm]")
        lxy_err_axis = hist.axis.Regular(50, 0.0, 100.0, name="lxy_err", label=r"$\sigma_{L_{xy}} [cm]$")

        # (name, x axis, y axis)
        v_a_2d = [
            ('lxy_err_VS_lxy', lxy_axis, lxy_err_axis)
        ]

        return v_a_2d

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator
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
        self.process_ds(events, out, ds)
        # if "2Mu2J" in ds:
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
        # Isolate denominator events
        events_den, dsa_1, dsa_2 = self.process_den(events, out, ds, do_sel_dsa=do_dsa_sel)

        # Get dimuon and event selection for the numerator
        event_cut, dimuon_cut = self.process_num(events_den, out, ds, do_sel=do_dsa_sel)

        # If event passes selection save info for passing DSA muons, otherwise take leading DSA pair in denominator
        dsa_1 = ak.where(event_cut, events_den.DSAMuon[events_den.DiDSAMuon[dimuon_cut].l1Idx], dsa_1)[:, 0]
        dsa_2 = ak.where(event_cut, events_den.DSAMuon[events_den.DiDSAMuon[dimuon_cut].l2Idx], dsa_2)[:, 0]
        self.fill_histos_singleMuon(events_den, out, ds, dsa_1, dsa_2)
        self.fill_arrays_common(events_den, out, ds, "den", dsa_1, dsa_2)

        events_num = events_den[event_cut]
        dimuons_num = events_den.DiDSAMuon[dimuon_cut][event_cut]
        dsa_1 = events_num.DSAMuon[dimuons_num.l1Idx][:, 0]
        dsa_2 = events_num.DSAMuon[dimuons_num.l2Idx][:, 0]
        self.fill_arrays_common(events_num, out, ds, "num", dsa_1, dsa_2)
        self.save_dimuon_info(events_num, dimuons_num, out, ds)


    def process_den(self, events, out, ds, do_sel_dsa=True):
        print("Total den events:", len(events))
        # Select events with at least 2 DSA muons
        events = events[ak.num(events.DSAMuon, axis=-1) >= 2]
        print("Den events with 2 DSA muons:", len(events))
        out["n_ev_two_dsa"][ds] += len(events)
        # Take all possible pairs of DSA muons
        # But first order the DSAMuon collection with decreasing pT
        DSAMuon_ptOrdered = events.DSAMuon[ak.argsort(events.DSAMuon.pt, axis=-1, ascending=False)]
        l1_idx, l2_idx = ak.unzip(ak.argcombinations(DSAMuon_ptOrdered, 2))
        dsa_1 = DSAMuon_ptOrdered[l1_idx]
        # print(dsa_1.pt)
        dsa_2 = DSAMuon_ptOrdered[l2_idx]
        # print(dsa_2.pt)
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
        dimuon_cut = self.unique_matching_selection(dsa_1, dsa_2, gen_muons)
        event_cut = (ak.sum(dimuon_cut, axis=-1) > 0)
        events = events[event_cut]
        dsa_1 = dsa_1[dimuon_cut][event_cut]
        dsa_2 = dsa_2[dimuon_cut][event_cut]
        print("Den events after DSA matching with gen muons:", len(events))
        out["n_ev_gen_matching"][ds] += len(events)
        return events, dsa_1, dsa_2

    def process_num(self, events, out, ds, do_sel=True):
        # Require at least a DSA pair
        dimuons = events.DiDSAMuon
        dimuon_cut = ak.broadcast_arrays(dimuons.pt, True)[1]
        event_cut = (events.event >= 0)
        # print("Num events with a 2-DSA fit:", len(events))

        if do_sel:
            dimuon_cut = dimuon_cut & self.dsa_selection(events.DSAMuon[dimuons.l1Idx]) & self.dsa_selection(events.DSAMuon[dimuons.l2Idx])
            event_cut = event_cut & (ak.sum(dimuon_cut, axis=-1) > 0)
            # print("Num events with a 2-DSA fit after DSA selection:", len(events))
        print("Num events with a 2-DSA fit:", ak.sum(event_cut))

        # Require unique matching of each DSA in a fitted pair with a gen hnl muon
        dsa_1 = events.DSAMuon[dimuons.l1Idx]
        dsa_2 = events.DSAMuon[dimuons.l2Idx]
        gen_muons = events.gen_muon
        dimuon_cut = dimuon_cut & self.unique_matching_selection(dsa_1, dsa_2, gen_muons)
        event_cut = event_cut & (ak.sum(dimuon_cut, axis=-1) > 0)

        # # Require good vertex prob
        # dimuon_cut = dimuon_cut & (dimuons.ndof > 0) & (dimuons.svprob >= 0.001)
        # event_cut = event_cut & (ak.sum(dimuon_cut, axis=-1) > 0)
        # out["n_ev_sv_prob_cut"][ds] += len(events)

        out["n_ev_converging_fit"][ds] += ak.sum(event_cut)
        print("Num events after DSA matching with gen muons:", ak.sum(event_cut))

        return event_cut, dimuon_cut

    def fill_arrays_common(self, events, out, ds, stage, dsa_1, dsa_2):
        gen_muon_1 = events.gen_muon[:, 0]
        gen_muon_2 = events.gen_muon[:, 1]
        out[f"{stage}_dimuon_deltaR_gen"][ds] += processor.column_accumulator(ak.to_numpy(delta_r(gen_muon_1, gen_muon_2), False))
        out[f"{stage}_dimuon_deltaEta_gen"][ds] += processor.column_accumulator(ak.to_numpy((gen_muon_1.eta - gen_muon_2.eta), False))
        out[f"{stage}_dimuon_deltaPhi_gen"][ds] += processor.column_accumulator(ak.to_numpy(delta_phi(gen_muon_1, gen_muon_2), False))
        out[f"{stage}_dimuon_pt_gen"][ds] += processor.column_accumulator(ak.to_numpy(sum_pt(gen_muon_1, gen_muon_2), False))
        out[f"{stage}_Lxy_gen"][ds] += processor.column_accumulator(ak.to_numpy(events.gen_dxy, False))
        out[f"{stage}_dimuon_deltaR"][ds] += processor.column_accumulator(ak.to_numpy(delta_r(dsa_1, dsa_2), False))
        out[f"{stage}_dimuon_deltaEta"][ds] += processor.column_accumulator(ak.to_numpy((dsa_1.eta - dsa_2.eta), False))
        out[f"{stage}_dimuon_deltaPhi"][ds] += processor.column_accumulator(ak.to_numpy(delta_phi(dsa_1, dsa_2), False))
        out[f"{stage}_dimuon_pt"][ds] += processor.column_accumulator(ak.to_numpy(sum_pt(dsa_1, dsa_2), False))

    def save_dimuon_info(self, events, dimuons, out, ds):
        pv = events.PV
        dimuon = dimuons[:, 0]
        dxy2 = (dimuon.vtx_x - pv.x) ** 2 + (dimuon.vtx_y - pv.y) ** 2
        dz2 = (dimuon.vtx_z - pv.z) ** 2
        dxy = np.sqrt(dxy2)
        dz = np.sqrt(dz2)
        dxyz = np.sqrt(dxy2 + dz2)
        dxy_err = np.sqrt((dimuon.vtx_x - pv.x) ** 2 * dimuon.vtx_ex ** 2 + (dimuon.vtx_y - pv.y) ** 2 * dimuon.vtx_ey ** 2) / dxy

        out[f'num_dxy_dsapair'][ds] += processor.column_accumulator(ak.to_numpy(dxy, False))
        out[f'num_dz_dsapair'][ds] += processor.column_accumulator(ak.to_numpy(dz, False))
        out[f'num_dxyz_dsapair'][ds] += processor.column_accumulator(ak.to_numpy(dxyz, False))
        out[f'num_dxy_dsapair_error'][ds] += processor.column_accumulator(ak.to_numpy(dxy_err, False))

        # Fill lxy 1D histos
        out["dimuon_lxy_pull"].fill(ds=ds, lxy_pull=(dxy - events.gen_dxy) / dxy_err)
        out["dimuon_lxy_significance"].fill(ds=ds, lxy_significance=dxy / dxy_err)
        out["dimuon_lxy"].fill(ds=ds, lxy=dxy)
        out["dimuon_lxy_res"].fill(ds=ds, lxy_res=(dxy - events.gen_dxy))

        # Fill lxy 2D histos
        out["lxy_err_VS_lxy"].fill(ds=ds, lxy_err=dxy_err, lxy=dxy)


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

    def fill_histos_singleMuon(self, events, out, ds, dsa_1, dsa_2):
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