# Instructions to collect the articles corpus

You will find below the instructions to collect the article corpus that we used in our experiments.

Before starting, you need to obtain your own API keys for the New York Times and The Guardian APIs. 
Replace "YOUR_NYT_API_KEY" and "YOUR_GUARDIAN_API_KEY" in '''download_nyt_articles.py''' and '''download_guardian_articles.py''' by your own API keys, respectively.

Then, run the following scripts as follows

```python
# Download rticles
python download_nyt_articles.py
python download_guardian_articles.py
# Images
python download_nyt_image.py
python download_guardian_image.py
# Remove articles with an image that is identical to one of the input images of the TARA dataset
python remove_duplicates.pyµ
# Remove articles that are not relevant
python preprocessing_qwen.py
# Generate news captions based on the abstracts of the corpus and TARA articles
python caption_generation.py  --input_folder data/processed_articles
python caption_generation.py  --input_folder data/tara_articles
```

Important note: we cannot guarantee that the APIs will provide the exact same set of articles as the one obtained when conducting our experiments. The current scripts automatically remove articles that are not part of the original corpus.

