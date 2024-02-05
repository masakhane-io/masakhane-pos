[MasakhaPOS: Part-of-Speech Tagging for Typologically Diverse African Languages](https://aclanthology.org/2023.acl-long.609/)


The code is based on HuggingFace implementation (License: Apache 2.0).

The license of the POS dataset is in [CC-BY-4.0-NC](https://creativecommons.org/licenses/by-nc/4.0/), the monolingual data have difference licenses depending on the news website license. The monolingual data used for annotation can be found [here](https://github.com/masakhane-io/lacuna_pos_ner/tree/main/language_corpus)

### Required dependencies
* python
  * [transformers](https://pypi.org/project/transformers/) : state-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch.
  * [seqeval](https://pypi.org/project/seqeval/) : testing framework for sequence labeling.
  * [ptvsd](https://pypi.org/project/ptvsd/) : remote debugging server for Python support in Visual Studio and Visual Studio Code.

```bash
pip install transformers seqeval ptvsd
```


If you make use of this dataset, please cite us:

### BibTeX entry and citation info
```
@inproceedings{dione-etal-2023-masakhapos,
    title = "{M}asakha{POS}: Part-of-Speech Tagging for Typologically Diverse {A}frican languages",
    author = "Dione, Cheikh M. Bamba  and Adelani, David Ifeoluwa  and Nabende, Peter  and Alabi, Jesujoba  and Sindane, Thapelo  and Buzaaba, Happy  and Muhammad, Shamsuddeen Hassan  and Emezue, Chris Chinenye  and Ogayo, Perez  and Aremu, Anuoluwapo  and Gitau, Catherine  and Mbaye, Derguene  and Mukiibi, Jonathan  and Sibanda, Blessing  and Dossou, Bonaventure F. P.  and Bukula, Andiswa  and Mabuya, Rooweither  and Tapo, Allahsera Auguste  and Munkoh-Buabeng, Edwin  and Memdjokam Koagne, Victoire  and Ouoba Kabore, Fatoumata  and Taylor, Amelia  and Kalipe, Godson  and Macucwa, Tebogo  and Marivate, Vukosi  and Gwadabe, Tajuddeen  and Elvis, Mboning Tchiaze  and Onyenwe, Ikechukwu  and Atindogbe, Gratien  and Adelani, Tolulope  and Akinade, Idris  and Samuel, Olanrewaju  and Nahimana, Marien  and Musabeyezu, Th{\'e}og{\`e}ne  and Niyomutabazi, Emile  and Chimhenga, Ester  and Gotosa, Kudzai  and Mizha, Patrick  and Agbolo, Apelete  and Traore, Seydou  and Uchechukwu, Chinedu  and Yusuf, Aliyu  and Abdullahi, Muhammad  and Klakow, Dietrich",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.609",
    doi = "10.18653/v1/2023.acl-long.609",
    pages = "10883--10900",
    abstract = "In this paper, we present AfricaPOS, the largest part-of-speech (POS) dataset for 20 typologically diverse African languages. We discuss the challenges in annotating POS for these languages using the universal dependencies (UD) guidelines. We conducted extensive POS baseline experiments using both conditional random field and several multilingual pre-trained language models. We applied various cross-lingual transfer models trained with data available in the UD. Evaluating on the AfricaPOS dataset, we show that choosing the best transfer language(s) in both single-source and multi-source setups greatly improves the POS tagging performance of the target languages, in particular when combined with parameter-fine-tuning methods. Crucially, transferring knowledge from a language that matches the language family and morphosyntactic properties seems to be more effective for POS tagging in unseen languages.",
}
```
