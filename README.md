# Predictive Modeling, Deep Learning, NLP, Experimental Data Analysis
The machine learning (NLP, Deep Learning, Predictive Modeling) projects completed in 2023 fall, including 

## Deep Learning
  1. Customized Canday Detector (Resnet50, Manually Labelled Training Images by Myself through Label-Studio) -> The fine-tuned model is also accessible through HuggingFace: https://huggingface.co/tatakea-jin/detr-resnet-50_finetuned_image/tree/main

  2. Anomaly Detection using Autoencoders -> Analyze a short video and detect the frames where something unusual happens by checking if the loss of the model exceeds a pre-defined threshold

## NLP
  1. Customer Review Web Scrapping
  2. Sentiment Classification on Customer Reviews (tf-idf with n-gram BoW representation)
  3. Use a custom-trained Named Entity Recognition (NER) model to identify the dishes in the reviews (i.e. the aspects) and then use a pre-trained ABSA model to analyze the sentiment expressed about those dishes.

-----------
# Causal Inference on Tax Compliance Experimental Data
An in-depth analysis of the data collected in an experiment undertaken in Tanzanian to enhance tax compliance among its citizens. To bolster the tax collection system, three distinct SMS messaging strategies are experimented: sending a SMS tax remainder message (T1), sending the same SMS message with a warning of Mjumbe (local leader) involvement (T2), and sending the same SMS message with the actual Mjumbe involvement. That strategies will help for nuanced examination of the variate effects of SMS message effect and local leader involvement.

The analysis, mainly based on ANOVA and right-tail t-tests, reveals that the three treatments do not yield different impacts on tax compliance overall. However, by comparing the treatment impacts across varying Mjumbe sizes, a profound conclusion is arrived, where strategy with SMS message alone outperforms the strategy with the actual involvement of the local leaders for medium size Mjumbe (11-35 residents) and the strategy with the actual Mjumbe involvement should be preferred for small size Mjumbe (≤ 10 residents). These findings are supported by robust data analytics and visualizations, shedding light on the nuanced responses of taxpayers to different forms of SMS communication across different Mjumbes with different sizes.


The study employed a methodologically rigorous approach, incorporating a randomized control trial design where taxpayers were randomly assigned to receive one of the three SMS strategies or placed in a control group receiving no messages. The effectiveness of each strategy was assessed through subsequent tax compliance rates, with different statistical techniques utilized to ensure the reliability and validity of the findings. These results have significant implications for the Tanzanian government's tax collection strategies, and we recommend a more focused utilization of local leader involvement in tax compliance messages tailored to specific taxpayer segments to optimize compliance rates and enhance the efficiency of tax collection.