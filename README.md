# Augmented Self-Attention Pruning (ASAP)

PyTorch re-implementation of the Augmented Self-Attention Pruning (ASAP) model, IUI 2023, https://dl.acm.org/doi/abs/10.1145/3581641.3584081 \
Paper: "ASAP: Endowing Adaptation Capability to Agent in Human-Agent Interaction" [IUI 2023].

## Description of ASAP
In human-human interaction, interlocutors adapt their behaviors reciprocally and dynamically. ASAP models this adaptation mechanism, also referred to as reciprocal adaptation, between a Socially Interactive Agent (SIA) and a human interlocutor.\
The SIA behavior as speaker and listener is fully driven by ASAP. Only its head and upper facial expressions are computed. The voice of the SIA is dubbed from the original video of the human-human data.

## Demo video
A dyadic interaction between a human and a SIA simulated with the GRETA platform. SIA behaviors are generated via our ASAP model at the frame level at 25 fps.

[![ASAP DEMO](https://user-images.githubusercontent.com/44306168/213715354-b1742b06-8df2-45fc-a01c-91dce49e44c6.png)](http://www.youtube.com/watch?v=feojlOrFCIg "ASAP Demo")
Please click to see the full demo video.

## Requirements
- Python 3.12
- PyTorch 2.5.1
- numpy 2.4.4
- pandas 3.0.3
- scipy 1.17.1
- dtaidistance 2.4.0

## Instructions
It is possible to train, predict, and evaluate objectively the ASAP model. The data is read from `../data`.

To train the ASAP model, run:
```
python train_ASAP.py
```
The best weights are saved to `../trainedASAP/weights/best_weights-{epoch}.pth`.

To predict or evaluate the ASAP model, run:
```
python predNeval_ASAP.py
```
You can choose whether to do a prediction, an evaluation, or both by selecting the corresponding mode:
- "pred": predict with ASAP
- "eval": evaluate objectively by loading precomputed predictions of ASAP
- "predNeval": predict and evaluate objectively the predictions of ASAP
