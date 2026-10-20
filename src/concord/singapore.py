"""Singapore GST treatment flags for CPI 2024-base leaf items.

Standard-rated unless listed. Sources for the rules: IRAS GST guides (exempt supplies:
residential property leases, financial services incl. life insurance), and the
government's announcement that it absorbs GST on subsidised public healthcare and
education. Items marked "review" still need checking against the IRAS guide.

Run:  python -m src.concord.singapore
"""

from __future__ import annotations

import pandas as pd

from src.paths import CONCORDANCES, DATA_PROCESSED

OUT = CONCORDANCES / "singapore_gst_flags.csv"

EXEMPT = {
    "1.03.1.1": "residential lease (exempt supply)",
    "1.03.1.2": "imputed rent (no transaction)",
}
GST_ABSORBED = {  # GST charged but absorbed by government for subsidised services
    "1.05.2.1": "polyclinic fees", "1.05.2.3": "subsidised specialist outpatient",
    "1.09.1.1": "public school fees", "1.09.1.2": "public school fees",
    "1.09.1.3": "public post-secondary fees", "1.09.1.4": "polytechnic fees",
}
ADMINISTERED = {
    "1.03.2.1": "PUB water tariff", "1.03.2.2": "refuse collection fees",
    "1.03.2.3": "electricity tariff (quarterly)", "1.03.2.4": "gas tariff (quarterly)",
    "1.06.2.1": "PTC fare review (Dec 2022, Dec 2023)", "1.06.2.3": "PTC fare review",
    "1.06.1.1": "COE-driven car prices", "1.10.2.2": "tobacco excise rise, Feb 2023",
}
# Most stalls are below the S$1m GST registration threshold, so few charge GST.
LOW_REGISTRATION = {
    "1.11.3.1": "hawker centres", "1.11.3.2": "food courts, coffee shops & kiosks",
}
REVIEW = {"1.05.5": "health insurance: MediShield Life/IP treatment to confirm",
          "1.05.3": "inpatient: mix of subsidised (absorbed) and private wards",
          "1.10.4": "social services: subsidised childcare treatment to confirm"}


def build() -> pd.DataFrame:
    cpi = pd.read_parquet(DATA_PROCESSED / "singapore_cpi.parquet")
    leaves = (cpi[cpi.is_leaf & cpi.series_no.str.startswith("1.")]
              .groupby(["series_no", "name"]).month.agg(first_month="min", last_month="max").reset_index())
    def status(code: str) -> str:
        if code in EXEMPT:
            return "exempt"
        if code in GST_ABSORBED:
            return "gst_absorbed"
        return "standard"
    leaves["gst_status"] = leaves.series_no.map(status)
    leaves["reason"] = leaves.series_no.map({**EXEMPT, **GST_ABSORBED}).fillna("")
    leaves["administered"] = leaves.series_no.map(ADMINISTERED).fillna("")
    leaves["low_registration"] = leaves.series_no.map(LOW_REGISTRATION).fillna("")
    leaves["review"] = leaves.series_no.map(REVIEW).fillna("")
    key = leaves.series_no.map(lambda s: [int(x) for x in s.split(".")])
    return leaves.iloc[key.argsort()].reset_index(drop=True)


def main() -> None:
    df = build()
    df.to_csv(OUT, index=False, lineterminator="\n", encoding="utf-8")
    print(df.gst_status.value_counts().to_string())
    print(f"wrote {OUT.relative_to(OUT.parents[1]).as_posix()} ({len(df)} rows)")


if __name__ == "__main__":
    main()
