from ..models.rome import ROMEHyperParams, apply_rome_to_model
from ..models.kn import KNHyperParams, apply_kn_to_model
from ..models.mend import MENDHyperParams, MendRewriteExecutor
from ..models.ft import FTHyperParams, apply_ft_to_model
from ..dataset import ZsreDataset, CounterFactDataset
from ..models.ike import IKEHyperParams, apply_ike_to_model
from ..models.grace import GraceHyperParams, apply_grace_to_model
from ..models.wise import WISEHyperParams, apply_wise_to_model
from ..models.alphaedit import AlphaEditHyperParams, apply_AlphaEdit_to_model
from ..models.defer import DeferHyperParams, apply_defer_to_model
from ..models.horen import HORENHyperParams, apply_horen_to_model
from ..models.memit import MEMITHyperParams, apply_memit_to_model
from ..models.ultraedit import UltraEditHyperParams, UltraEditRewriteExecutor

ALG_DICT = {
    "ROME": apply_rome_to_model,
    "FT": apply_ft_to_model,
    "KN": apply_kn_to_model,
    "MEND": MendRewriteExecutor().apply_to_model,
    "IKE": apply_ike_to_model,
    "GRACE": apply_grace_to_model,
    "WISE": apply_wise_to_model,
    "AlphaEdit": apply_AlphaEdit_to_model,
    "DEFER": apply_defer_to_model,
    "HOREN": apply_horen_to_model,
    "MEMIT": apply_memit_to_model,
    "ULTRAEDIT": UltraEditRewriteExecutor().apply_to_model,
}

ALG_MULTIMODAL_DICT = {}

PER_ALG_DICT = {}

DS_DICT = {
    "cf": CounterFactDataset,
    "zsre": ZsreDataset,
}
