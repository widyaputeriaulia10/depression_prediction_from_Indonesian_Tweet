# depression_prediction_from_Indonesian_Tweet
This project is part of my Thesis, which predict depression based on Indonesian Tweets. you can access adn download the model by using this API :
kaggle kernels output widyaputeriaulia10/xlm-r-multilingual-1e-5-new -p /path/to/dest or URL : https://www.kaggle.com/code/widyaputeriaulia10/xlm-r-multilingual-1e-5-new/notebook

We used 2 approaches : fine-tuning and fine-tuning with metadata linguistic, for fine-tuning approach, we used end-to-end XLM-RoBERTa model with Multilingual Scenario (Indonesian language and English) got the best accuracy. 

For Fine-tuning with metadata linguistic approaches can not outperformed fine-tuning model. Therefore, to predict depression sentence we use the first approaches and after the sentences has been predicted we count the average of depression data for each user. If the average was more than 50% the user will be labeled as depression. You can use API file to predict the data directly.
