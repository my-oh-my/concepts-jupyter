"""
Centralized configuration for the project.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))

# Sub-directories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# API Settings
API_KEY = os.getenv("API_KEY", "")

# Symbols
BENCHMARK_NAME = "WIG20"

WIG20_SYMBOLS = [
    "ALE",
    "ALR",
    "BDX",
    "CDR",
    "CPS",
    "DNP",
    "JSW",
    "KGH",
    "KRU",
    "KTY",
    "LPP",
    "MBK",
    "OPL",
    "PCO",
    "PEO",
    "PGE",
    "PKN",
    "PKO",
    "PZU",
    "SPL",
]

MWIG40_SYMBOLS = [
    "11B",
    "ABE",
    "ACP",
    "APR",
    "ASB",
    "ATC",
    "ATT",
    "BFT",
    "BHW",
    "BNP",
    "CAR",
    "CBF",
    "CCC",
    "CIG",
    "CLN",
    "DOM",
    "DVL",
    "EAT",
    "ENA",
    "EUR",
    "GEA",
    "GPP",
    "GPW",
    "HUG",
    "ING",
    "LWB",
    "MBR",
    "MIL",
    "MRB",
    "NEU",
    "RBW",
    "RVU",
    "SLV",
    "SNT",
    "TEN",
    "TPE",
    "TXT",
    "WPL",
    "XTB",
]

SWIG80_SYMBOLS = [
    "1AT",
    "ABS",
    "AGO",
    "ALL",
    "AMB",
    "AMC",
    "APT",
    "ARH",
    "ASE",
    "AST",
    "BCX",
    "BIO",
    "BLO",
    "BMC",
    "BOS",
    "BRS",
    "CLC",
    "CMP",
    "COG",
    "CRI",
    "CRJ",
    "CTX",
    "DAT",
    "DCR",
    "ECH",
    "ELT",
    "ENT",
    "ERB",
    "FRO",
    "FTE",
    "GRX",
    "INK",
    "KGN",
    "LBW",
    "MAB",
    "MCI",
    "MDG",
    "MGT",
    "MLG",
    "MLS",
    "MNC",
    "MOC",
    "MRC",
    "MSZ",
    "MUR",
    "NWG",
    "OND",
    "OPN",
    "PBX",
    "PCE",
    "PCF",
    "PCR",
    "PEN",
    "PEP",
    "PLW",
    "PUR",
    "PXM",
    "SCP",
    "SEL",
    "SGN",
    "SHO",
    "SKA",
    "SNK",
    "STP",
    "STX",
    "SVE",
    "TAR",
    "TOA",
    "TOR",
    "UNT",
    "VGO",
    "VOT",
    "VOX",
    "VRC",
    "VRG",
    "WLT",
    "WTN",
    "WWL",
    "XTP",
    "ZEP",
]

ALL_SYMBOLS = list(set([BENCHMARK_NAME] + WIG20_SYMBOLS + MWIG40_SYMBOLS + SWIG80_SYMBOLS))
