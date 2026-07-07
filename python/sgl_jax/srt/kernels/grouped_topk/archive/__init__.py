"""Superseded grouped-topk kernel variants, kept for reference/reproducibility.

The production kernel is `grouped_topk.v1.kernel.grouped_topk_pallas` (token-in-lane [E,BT], stable
lowest-index tie-break, unpadded [topk,BS] output). These variants are not imported by production.
"""
