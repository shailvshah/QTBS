# QTBS
Query Based Text Summarizer

Simple python tool which automatically summarizes text based on user-input query based on plain text file. Currently uses wikipedia articles as input files.



This tool uses a combination of different efforts in NLP and Deep Learning to generate results. The questions dataset is trained using CNN in Keras.
Stanford Question-Answering Dataset is used to match patterns in answers based on model created in questions classification, finally this is further processed and textranking is done for the set of solutions that are closest.
