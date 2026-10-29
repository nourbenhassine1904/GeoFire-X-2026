# GeoFire-X — Project Context

## Scientific Objective

Predict the next-day active-fire map from multimodal spatio-temporal geospatial observations.

## Primary Dataset

WildfireSpreadTS.

## Input Contract

X: Tensor [T, C, H, W]

T = number of context days

C = number of channels/modalities

H x W = spatial grid

## Target Contract

Y: Tensor [1, H, W]

Y represents the binary active-fire map at J+1.

## Main Research Questions

RQ1 — Multimodal fusion

RQ2 — Explicit temporal modeling

RQ3 — Geographic and temporal robustness / OOD generalization

## Phase 1 Objective

Build a reproducible experimental foundation before training advanced Deep Learning models.

## Phase 1 Scope

- dataset version documentation
- dataset manifest
- data audit
- event-level train/validation/test split
- leakage checks
- reproducible DataLoader interface
- Dice / IoU / Precision / Recall
- Persistence baseline
- MLflow experiment tracking

## Out of Scope for Phase 1

- U-Net training
- ConvLSTM
- Transformer
- DANN
- Deep CORAL
- Kafka
- Spark
- cross-dataset evaluation
- final dashboard

## Critical Experimental Rule

A wildfire event must belong to exactly one split.

No day from an event used in training may appear in validation or test.

## Reproducibility Rule

Every experiment must be traceable to:

- dataset version
- split version
- configuration
- random seed
- Git commit
