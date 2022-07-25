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
        acc_dict = {"ctau": hist.Hist(
            "Counts",
            ds_axis,
            hist.Bin("ctau", "ctau [mm]", 100, 0, 500),
        )}
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
           "ctau",
            "pt",
            "mass",
            "dxy_gen"
        ]

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()
        ds = events.metadata["dataset"]
        for var in self.get_array_vars():
            out[f'num_{var}'][ds] = processor.column_accumulator(np.array([]))
            out[f'den_{var}'][ds] = processor.column_accumulator(np.array([]))

        events["hnl"] = events.GenPart[(events.GenPart.pdgId == 9900012) & (events.GenPart.statusFlags >= 2 ** 13)]
        events["ctau"] = ak.max(events.LHEPart.ctau, axis=-1)
        # To select hnl muons we need to first require distinctParentIdxG != -1 to avoide None pdgId
        hnl_muons = events.GenPart[(np.abs(events.GenPart.pdgId) == 13) & (events.GenPart.status == 1) & (
                events.GenPart.distinctParentIdxG != -1)]
        events["gen_muon"] = hnl_muons[hnl_muons.distinctParent.pdgId == 9900012]

        # Select events with 2 hnl muons in acceptance
        cut = (events.gen_muon.pt > 3) & (np.abs(events.gen_muon.eta) < 2.4)
        events["gen_muon"] = events.gen_muon[cut]
        events = events[ak.num(events.gen_muon.pt, axis=-1) == 2]
        print("events, 2 gen muons:", len(events))

        events = self.process_den(events, out, ds)
        events, dimuons = self.process_num(events, out, ds)

        return out

    def process_den(self, events, out, ds, do_sel_dsa=True):
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
            print("events dsa sel den:", len(events))

        # Require unique matching of each DSA pair with the gen muons
        gen_muons = events.gen_muon
        cut = self.unique_matching_selection(dsa_1, dsa_2, gen_muons)
        event_cut = (ak.sum(cut, axis=-1) > 0)
        events = events[event_cut]
        print("den events after matching:", len(events))
        out["ctau"].fill(
            ds=ds,
            ctau=ak.flatten(events.ctau, axis=-1),
        )
        self.fill_arrays(events, out, ds, "den")
        return events

    def process_num(self, events, out, ds, do_sel=True):
        # Require at least a DSA pair
        events = events[ak.num(events.DiDSAMuon) >= 1]
        dimuons = events.DiDSAMuon[ak.num(events.DiDSAMuon) >= 1]
        print("events 2 dsa fit:", len(events))

        if do_sel:
            cut = self.dsa_selection(events.DSAMuon[dimuons.l1Idx]) & self.dsa_selection(events.DSAMuon[dimuons.l2Idx])
            dimuons = dimuons[cut]
            events = events[ak.num(dimuons) > 0]
            dimuons = dimuons[ak.num(dimuons) > 0]
            print("events dsa sel:", len(events))

        # Require unique matching of each DSA in a fitted pair with a gen hnl muon
        dsa_1 = events.DSAMuon[dimuons.l1Idx]
        dsa_2 = events.DSAMuon[dimuons.l2Idx]
        gen_muons = events.gen_muon
        cut = self.unique_matching_selection(dsa_1, dsa_2, gen_muons)

        dimuons = dimuons[cut]
        events = events[ak.num(dimuons) > 0]
        dimuons = dimuons[ak.num(dimuons) > 0]

        # # Require good vertex prob
        # cut = (dimuons.ndof > 0) & (dimuons.svprob >= 0.001)
        # dimuons = dimuons[cut]
        # events = events[ak.num(dimuons) > 0]
        # dimuons = dimuons[ak.num(dimuons) > 0]

        self.fill_arrays(events, out, ds, "num")

        return events, dimuons

    def fill_arrays(self, events, out, ds, name):
        ctau = ak.to_numpy(ak.flatten(events.ctau, axis=None), False)  # mm
        pt = ak.to_numpy(ak.flatten(events.hnl.pt, axis=None), False)
        mass = ak.to_numpy(ak.flatten(events.hnl.mass, axis=None), False)
        dxy_gen = ctau / 10 * pt / mass  # cm
        out[f"{name}_ctau"][ds] += processor.column_accumulator(ctau)
        out[f"{name}_pt"][ds] += processor.column_accumulator(pt)
        out[f"{name}_mass"][ds] += processor.column_accumulator(mass)
        out[f"{name}_dxy_gen"][ds] += processor.column_accumulator(dxy_gen)

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


    def postprocess(self, accumulator):
        return accumulator