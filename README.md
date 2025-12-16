### Code and data for the ARR January submission "NewsRECON"

This anonymous repository contains the code and data for the anonymous ARR submission "NewsRECON"

## Abstract

> Identifying when and where a news image was taken is crucial for journalists and forensic experts to produce credible stories and debunk misinformation. While many existing methods rely on reverse image search (RIS) engines, these tools often fail to return results, thereby limiting their practical applicability. In this work, we address the challenging scenario where RIS evidence is unavailable. We introduce NewsRECON, a method that links images to relevant news articles to infer their date and location from article metadata. NewsRECON leverages a corpus of over 90,000 articles and integrates: (1) a bi-encoder for retrieving event-relevant articles; (2) two cross-encoders for reranking articles by location and event consistency. Experiments on the TARA and 5Pils-OOC show that NewsRECON outperforms prior work and can be combined with a multimodal large language model to achieve new SOTA results in the absence of RIS evidence. We make our code and data available.

## Environment

Follow these instructions to recreate the environment used for all our experiments.

```
$ conda create --name newsrecon python=3.9
$ conda activate newsrecon
$ pip install -r requirements.txt
```

<p align="center">
  <img width="70%" src="assets/newsrecon.png" alt="header" />
</p>